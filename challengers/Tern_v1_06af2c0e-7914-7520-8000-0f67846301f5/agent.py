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
LIST_STEPS = (1, 1)
LIST_STEPS_MIXED = (1, 1)
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
GUIDE = 'You are Tern, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store\'s search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Tern knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Tern\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Tern carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Tern\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Tern adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Tern reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: Tern re-reads the cart\'s lines and looks at the cart when you call the checkout, so do not spend a turn on either.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\n3. Last, whatever a hint or the steps above say: questions on these almost never uncover a preference, so they come after every other choice: the product\'s condition or availability, a model, part, series or compatibility code, when or where it will be used (an occasion, a season, a room), and a feature a listing either has or lacks (a yes or no key).\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.'
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
    hit_found = re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))
    return [re.sub('[\\s-]+', ' ', c).strip().lower() for c in hit_found if not re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', c)]

def _code_hit(pattern: str, code: str) -> bool:
    """Whether a code a part names covers a device's code: the same letters and digits, or a series written with trailing
    x's ("abc12xx" covers "abc1234"); spaces and hyphens do not count."""
    (wide, plain) = (re.sub('[\\s-]+', '', pattern), re.sub('[\\s-]+', '', code))
    stem = wide.rstrip('x')
    return plain.startswith(stem) and len(plain) == len(wide) if stem != wide and stem[-1:].isdigit() else wide == plain

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    code_bag: list[str] = []
    for hit_at in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + FIT_CODE + ')\\b', text, re.I):
        code_bag += _codes(hit_at.group(1))
    for hit_at in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        code_bag += _codes(hit_at.group(1))
    works: list[str] = []
    bars: list[str] = []
    for hit_at in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit_at.start() - 40):hit_at.start()], re.I):
            cut = re.split('(?i)\\b(?:except|excluding|but not)\\b', hit_at.group(1), maxsplit=1)
            works += _codes(cut[0])
            bars += _codes(cut[1]) if len(cut) > 1 else []
    also = [(hit_at.group(1).upper(), _codes(hit_at.group(2))) for hit_at in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    further = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    (fam, cited) = _series_marks(text)
    return {'own': own.group(1).upper() if own else '', 'codes': code_bag, 'works': works, 'bars': bars, 'also': also, 'more': further.group(1).upper() if further else '', 'fam': fam, 'named': cited}

def _series_marks(text: str) -> tuple[list[str], list[str]]:
    """The series a record's own codes start with ("AB1" of "AB1-2345") and the bare series-like words it names outside a clause that
    excludes them. Which bare words are series is decided against the codes other records carry (Tern.link_system)."""
    fam = {hit_at.upper() for hit_at in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)}
    leftover = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text)
    leftover = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', leftover)
    return (sorted(fam), sorted({hit_at.upper() for hit_at in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', leftover)}))
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
    if any((raw_val and re.search(FAIL_KEY, str(fkey), re.I) for (fkey, raw_val) in observation.items())):
        return True
    only = next(iter(observation.values()))
    return len(observation) == 1 and isinstance(only, str) and (0 < len(only) <= 40) and (' ' not in only)

def _plain(text: str) -> Any:
    """A field's text as the value a structured result would hold: a boolean, nothing, a number, else the text."""
    bare = text.strip()
    lowered = bare.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('none', 'null', ''):
        return None
    money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
    if money and (not re.match('0\\d', money.group(1))):
        goods_no = float(money.group(1).replace(',', ''))
        return int(goods_no) if goods_no.is_integer() and '.' not in money.group(1) else goods_no
    return bare

def _block_value(text: str) -> Any:
    """A value inside an attribute block: "attachment method=none" is a value, and a block only keeps text and numbers."""
    hit_found = _plain(text)
    return text.strip() if isinstance(hit_found, bool) or (hit_found is None and text.strip()) else hit_found

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    """(name, value) when the text opens with a field name and this mark."""
    (head_word, hit_found, leftover) = text.partition(mark)
    if not hit_found and text.endswith(mark.strip()):
        (head_word, hit_found, leftover) = (text[:-len(mark.strip())], mark, '')
    nm = re.sub(LIST_BULLET, '', head_word).strip()
    return None if not hit_found or not re.fullmatch(FIELD_NAME, nm) or len(nm.split()) > 5 else (nm, leftover.strip())

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    """The fields of a text cut at `between`, a part with no name joining the field before it (a value may hold the
    separator), with how many parts carried a name. A name closed by a colon with nothing after it heads the bulleted
    or indented lines below it, kept as one field holding those lines."""
    segments = [p for p in text.split(between) if p.strip()]
    fields: list[tuple[str, str]] = []
    cited = 0
    heading = -1
    if segments and (not _pair(segments[0].strip(), mark)):
        return ([], 0)
    for portion in segments:
        bare = portion.strip()
        if between == '\n' and heading >= 0 and (re.match(LIST_BULLET, portion) or portion[:1] in ' \t'):
            (nm, raw_val) = fields[heading]
            fields[heading] = (nm, (raw_val + '\n' if raw_val else '') + bare)
            continue
        heading = -1
        if between == '\n' and bare.endswith(':') and re.fullmatch(FIELD_NAME, bare[:-1].strip()):
            fields.append((bare[:-1].strip(), ''))
            heading = len(fields) - 1
            cited += 1
            continue
        pair = _pair(bare, mark)
        if pair:
            fields.append(pair)
            cited += 1
        elif fields:
            (nm, raw_val) = fields[-1]
            fields[-1] = (nm, raw_val + between + bare)
    return (fields, cited) if len(segments) else ([], 0)

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
            hit_found = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', bare))
            return {key_name.strip(): _plain(item_val) for (key_name, item_val) in hit_found.items()} if len(hit_found) >= 2 else None
    if '=' in bare:
        acc: dict[str, Any] = {}
        last = ''
        segments = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', bare) if p.strip()]
        for portion in segments:
            (nm, hit_found, raw_val) = portion.partition('=')
            if hit_found and 0 < len(nm.strip()) <= BLOCK_NAME and (nm.strip() not in acc):
                last = nm.strip()
                acc[last] = _block_value(raw_val)
            elif last:
                acc[last] = f'{acc[last]}; {portion.strip()}'
            else:
                return None
        return acc if len(acc) >= 2 or (acc and ':' not in last and (len(last) <= 30) and (len(bare) <= 90)) else None
    rowset = [basket_line for basket_line in bare.splitlines() if basket_line.strip()]
    if len(rowset) >= 2 and all((re.match(LIST_BULLET, basket_line) for basket_line in rowset)):
        pairs = [_pair(basket_line.strip(), ': ') for basket_line in rowset]
        if all(pairs):
            return {nm: _plain(raw_val) for (nm, raw_val) in pairs}
    for between in (', ', '; '):
        if between in bare and ': ' in bare:
            pairs = [_pair(portion.strip(), ': ') for portion in bare.split(between)]
            if len(pairs) >= 2 and all(pairs) and all(('.' not in nm and len(raw_val) <= BLOCK_VALUE for (nm, raw_val) in pairs)):
                return {nm: _plain(raw_val) for (nm, raw_val) in pairs}
    return None

def _record_text(text: str) -> dict[str, Any] | None:
    """The record a text spells out field by field, whichever separators it uses; None when it reads as plain prose. The
    separators are found by trying them, the outer field mark first and then the pair mark giving the most named fields,
    and a value spelled the same way becomes a record of its own (a listing's attributes written inside one field)."""
    winner: tuple[int, list[tuple[str, str]]] | None = None
    for between in FIELD_MARKS:
        if not between not in text:
            pieces = len([p for p in text.split(between) if p.strip()])
            for mark in PAIR_MARKS:
                (fields, cited) = _split_fields(text, between, mark)
                if cited < RECORD_FIELDS or cited < (PAIR_SHARE if between == ', ' else FIELD_SHARE) * pieces:
                    continue
                if len({nm for (nm, _) in fields}) * 2 <= len(fields):
                    continue
                if winner is None or cited > winner[0]:
                    winner = (cited, fields)
            if winner:
                break
    if winner is None:
        return None
    rec_obj: dict[str, Any] = {}
    for (nm, raw_val) in winner[1]:
        if not nm in rec_obj:
            inner = _attr_block(raw_val)
            rec_obj[nm] = inner if inner else raw_val.strip() if re.search(ID_FIELD, nm, re.I) else _plain(raw_val)
    return rec_obj

def _table_text(text: str) -> list[dict[str, Any]] | None:
    """Rows of a text table: a heading line of names, then lines cut into as many cells (markdown rules are skipped)."""
    rowset = [basket_line.strip().strip('|').strip() for basket_line in text.splitlines() if basket_line.strip() and (not re.fullmatch('[\\s|:+=-]+', basket_line))]
    if len(rowset) < 2:
        return None
    for between in ('|', '\t', ';', ','):
        cells = [[c.strip() for c in basket_line.split(between)] for basket_line in rowset]
        width_n = len(cells[0])
        if width_n < 3 or any((len(one_row) != width_n for one_row in cells)) or (not all((re.fullmatch(FIELD_NAME, c) for c in cells[0]))):
            continue
        if any((re.fullmatch('[\\d.,$]+', c) for c in cells[0])):
            continue
        return [{nm: _plain(c) for (nm, c) in zip(cells[0], one_row)} for one_row in cells[1:]]
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
    for segments in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
        if len(segments) >= 2:
            rows = [_record_text(p) for p in segments]
            good = [rw for rw in rows if rw]
            if len(good) >= 2 and len(good) * 5 >= len(segments) * 4 and (len({next(iter(rw)) for rw in good}) == 1):
                return good
    hit_found = _table_text(bare) if '\n' in bare and _record_text(bare) is None else _record_text(bare)
    if hit_found is None and len(bare) <= TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
        plain = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', bare, flags=re.I)
        plain = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', plain, flags=re.I)
        plain = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', plain).strip()
        if plain != bare and len(plain) >= 12:
            return _records_text(plain)
    if hit_found is None and '\n' in bare:
        rowset = [re.sub('(?:[ ]{2,}|\t)', ': ', basket_line.strip(), count=1) for basket_line in bare.splitlines() if basket_line.strip()]
        spaced = '\n'.join(rowset)
        if spaced != bare:
            return _record_text(spaced)
    return hit_found

def _block_text(text: str) -> Any:
    """A text field of a row that already has its own fields: read as a record only when wholly written as name=value parts."""
    if len(text) < 5 or '=' not in text or (not re.search('; | \\| |\\n', text)):
        return text
    segments = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text.strip()) if p.strip()]
    return text if len(segments) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in segments)) else _attr_block(text) or text

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    """Attributes written as a list: objects of a name and a value, [name, value] lists, or, inside a record, short "name: value"
    texts. A list of offers, of highlights or of notes is no such thing."""
    if len(rows) < 2:
        return None
    acc: dict[str, Any] = {}
    first = rows[0]
    if isinstance(first, dict):
        if any((not isinstance(rw, dict) or set(rw) != set(first) or len(rw) != 2 for rw in rows)):
            return None
        fkeys = list(first)
        if not re.search(PAIR_KEY, str(fkeys[0]), re.I):
            return None
        for one_row in rows:
            (nm, raw_val) = (one_row[fkeys[0]], one_row[fkeys[1]])
            if not isinstance(nm, str) or not 0 < len(nm) <= BLOCK_NAME or isinstance(raw_val, (dict, list)):
                return None
            acc[nm.strip()] = raw_val
        return acc
    if isinstance(first, (list, tuple)):
        for one_row in rows:
            if not isinstance(one_row, (list, tuple)) or len(one_row) != 2 or (not isinstance(one_row[0], str)) or isinstance(one_row[1], (dict, list)) or (not 0 < len(one_row[0]) <= BLOCK_NAME):
                return None
            acc[one_row[0].strip()] = one_row[1]
        return acc
    if in_record and all((isinstance(rw, str) for rw in rows)):
        for one_row in rows:
            pair = _pair(one_row.strip(), ': ') or _pair(one_row.strip(), '=')
            if not pair or len(pair[1]) > BLOCK_VALUE or len(pair[0].split()) > 3:
                return None
            acc[pair[0]] = _plain(pair[1])
        return acc
    return None

def _column_rows(node: dict[str, Any]) -> dict[str, Any]:
    """A result that names its columns once and gives each row as a list comes back with each row as an object."""
    heads = [key_name for (key_name, item_val) in node.items() if isinstance(item_val, list) and len(item_val) >= 2 and all((isinstance(c, str) and re.fullmatch(FIELD_NAME, c) for c in item_val))]
    for head_word in heads:
        names = node[head_word]
        for (fkey, rows) in node.items():
            if fkey != head_word and isinstance(rows, list) and rows and all((isinstance(rw, (list, tuple)) and len(rw) == len(names) for rw in rows)):
                acc = {key_name: item_val for (key_name, item_val) in node.items() if key_name != head_word}
                acc[fkey] = [dict(zip(names, one_row)) for one_row in rows]
                return acc
    return node

def _id_texts(record: dict[str, Any]) -> dict[str, Any]:
    """A record's numeric identifier as text, the form the agent keys listings by (schemas turn it back when a tool wants a number)."""
    return record if not any((isinstance(item_val, str) and ' ' in item_val for item_val in record.values())) else {key_name: str(item_val) if isinstance(item_val, int) and (not isinstance(item_val, bool)) and re.search(ID_FIELD, key_name, re.I) and (abs(item_val) >= 100) else item_val for (key_name, item_val) in record.items()}

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Rows given as an object keyed by listing code, each a record with the same fields, as a list of rows with their code."""
    rows = list(node.values())
    if len(rows) < 2 or not all((isinstance(rw, dict) and len(rw) >= RECORD_FIELDS for rw in rows)) or (not all((re.search('\\d', str(key_name)) and ' ' not in str(key_name) for key_name in node))):
        return None
    return None if len(set(rows[0]).intersection(*rows[1:])) < RECORD_FIELDS else [rw if any((re.search(ID_FIELD, str(el), re.I) for el in rw)) else {'id': str(key_name), **rw} for (key_name, rw) in node.items()]

def _holds_id(node: dict[str, Any]) -> bool:
    return any((re.search(ID_FIELD, str(key_name), re.I) for key_name in node))

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    """An attribute block with groups inside it, as one level: {"size": {"width": 3}} gives {"size width": 3}."""
    if not any((isinstance(item_val, dict) for item_val in node.values())):
        return node
    acc: dict[str, Any] = {}
    for (fkey, raw_val) in node.items():
        if isinstance(raw_val, dict):
            acc.update({f'{fkey} {inner}': item_val for (inner, item_val) in raw_val.items() if not isinstance(item_val, (dict, list))})
        else:
            acc[fkey] = raw_val
    return acc

def _restore_field(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
    """A field's value restored; a text only becomes a record when it reads as a listing, by an id, by a code among its values
    or by many fields, so that a store's note written with colons and semicolons stays a note."""
    hit_found = _restore(value, depth + 1, carded, page)
    return value if isinstance(value, str) and isinstance(hit_found, dict) and (len(hit_found) < NOTE_FIELDS) and (not _holds_id(hit_found)) and (not any((_id_like(item_val) for item_val in hit_found.values()))) else hit_found

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    """A store result in the one form the rest of the agent reads: lists of row objects, a listing's attributes as a nested
    object. Worlds spell results differently (race 15 gave each record as one text, its attributes "a=b; c=d" inside it, and
    a result the agent cannot read leaves every record-based check blind), so the spelling is undone here, once, for all."""
    if depth > 7:
        return value
    if isinstance(value, str):
        hit_found = _records_text(value)
        return hit_found if hit_found is not None else value
    if isinstance(value, list):
        chunk = _pairs_block(value, in_record and page)
        return chunk if chunk else [_restore_field(item_val, depth, False, page) for item_val in value]
    if isinstance(value, dict):
        node = _id_texts(_column_rows(value))
        carded = len(node) >= RECORD_FIELDS or _holds_id(node)
        keyed = {key_name: _keyed_rows(item_val) if isinstance(item_val, dict) else None for (key_name, item_val) in node.items()}
        node = {key_name: keyed[key_name] or item_val for (key_name, item_val) in node.items()}
        listing = _holds_id(node) or any((_id_like(item_val) for item_val in node.values()))
        acc = {key_name: _restore_field(item_val, depth, carded, page) if not isinstance(item_val, str) or len(node) <= 2 else _block_text(item_val) if listing else item_val for (key_name, item_val) in node.items()}
        if depth <= 3:
            acc = {key_name: _flat_block(item_val) if isinstance(item_val, dict) and any((isinstance(el, dict) for el in item_val.values())) and (not any((isinstance(el, list) for el in item_val.values()))) and isinstance(next(iter(item_val.values()), None), (dict, str, int, float)) and (len(node) >= RECORD_FIELDS) and (not _holds_id(item_val)) else item_val for (key_name, item_val) in acc.items()}
        if len(acc) == 1:
            only = next(iter(acc.values()))
            if isinstance(only, dict) and len(only) >= RECORD_FIELDS and isinstance(next(iter(value.values())), str):
                return only
        return acc
    return value
_FIRST_SPELLING: dict[str, str] = {}

def _join_twins(held: str, value: str) -> str:
    """Two spellings' values as one list, each element once; the first spelling's value stays known, for measures of how a key's values read."""
    between = '(?<!\\d),|,(?!\\d)'
    held_now = {portion.strip().lower() for portion in re.split(between, held)}
    extra: list[str] = []
    for portion in re.split(between, value):
        if portion.strip() and portion.strip().lower() not in held_now:
            held_now.add(portion.strip().lower())
            extra.append(portion.strip())
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
    lowered = text.lower()
    return set(re.findall('[a-z0-9]+', lowered)) | set(re.findall('[a-z0-9]+', _glued(lowered)))

def _value_tokens(value: str) -> list[str]:
    """A value's words a search or a row can hold: long ones, else short ones of two characters holding a letter (XL, 4g, 65w), else its numbers."""
    lowered = _glued(value)
    return re.findall('[a-z0-9]{3,}', lowered) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', lowered) or re.findall('[0-9]+', lowered)

def _fold_twins(node: Any, depth: int=0) -> Any:
    """One line per property: a block spelling a key twice keeps the first spelling, with both values when they differ, so that
    one element of a list is enough ("clock shape: rectangle" and "clock_shape: rectangular, square" hold square). About a tenth
    of the listings of races 13-15 and of a practice world hold such pairs with different values; read as a conflict, they
    lost listings the store's answer accepted. Every repeated line also costs tokens on each later model turn."""
    if depth > 5:
        return node
    if isinstance(node, list):
        return [_fold_twins(item_val, depth + 1) for item_val in node]
    if not isinstance(node, dict):
        return node
    acc: dict[str, Any] = {}
    noticed: dict[str, str] = {}
    for (fkey, raw_val) in node.items():
        nm = _words(fkey)
        first = noticed.get(nm)
        if first is None:
            noticed[nm] = fkey
            acc[fkey] = _fold_twins(raw_val, depth + 1)
        elif acc[first] == raw_val:
            continue
        elif isinstance(acc[first], str) and isinstance(raw_val, str) and (not re.search(ID_FIELD, nm, re.I)):
            acc[first] = _join_twins(acc[first], raw_val)
        else:
            acc[fkey] = _fold_twins(raw_val, depth + 1)
    return acc if len(acc) >= 2 or len(node) < 2 else {key_name: _fold_twins(item_val, depth + 1) for (key_name, item_val) in node.items()}

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
        return {key_name: _clip(item_val, limit, depth + 1) for (key_name, item_val) in value.items()}
    return [_clip(item_val, limit, depth + 1) for item_val in value] if isinstance(value, list) else value

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
    (left, right) = (text.find('{'), text.rfind('}'))
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
    return frozenset((tok[:-1] if len(tok) > 3 and tok.endswith('s') else tok for tok in _words(text).split()))

def _roots(text: str) -> set[str]:
    acc = set()
    for tok in _words(text).split():
        tok = tok[:-2] if re.search('(?:ch|sh|ss|x|z)es$', tok) else tok[:-1] if len(tok) > 3 and tok.endswith('s') and (not tok.endswith('ss')) else tok
        if len(tok) >= 3 and tok not in ('and', 'for', 'the', 'with'):
            acc.add(tok)
    return acc

def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', tok[:-1] if len(tok) > 3 and tok.endswith('s') and (not tok.endswith('ss')) else tok) for tok in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    return False if not prop or not key else prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    sort_kind = schema.get('type')
    return [sort_kind] if isinstance(sort_kind, str) else [key_name for key_name in sort_kind or [] if isinstance(key_name, str)]

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
    return value is None if kind == 'null' else True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, key_name) for key_name in kinds))):
        for sort_kind in kinds:
            if sort_kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if sort_kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                goods_no = float(value)
                value = int(goods_no) if sort_kind == 'integer' or goods_no.is_integer() else goods_no
                break
            if sort_kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if sort_kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if sort_kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, key_name) for key_name in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            lowered = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == lowered]
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
            value = [_shape(item_val, schema['items'], f'{where}[{cur}]', faults) for (cur, item_val) in enumerate(value)]
    if isinstance(value, dict):
        prop_map = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for nm in [idx for idx in value if idx not in prop_map]:
                faults.append(f'{where}.{nm} is not an allowed argument')
        for nm in schema.get('required') or []:
            if nm not in value:
                faults.append(f'{where}.{nm} is required')
        value = {key_name: _shape(item_val, prop_map.get(key_name), f'{where}.{key_name}', faults) if key_name in prop_map else item_val for (key_name, item_val) in value.items()}
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []

def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(float(value)) if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value) else None

def _id_like(value: Any, named: bool=False) -> bool:
    return False if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip() else named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    winner = None
    for fkey in rows[0]:
        value_list = [one_row.get(fkey) for one_row in rows]
        cited = bool(re.search(ID_FIELD, str(fkey), re.I))
        if not all((_id_like(item_val, cited) for item_val in value_list)):
            continue
        value_list = [item_val.strip() for item_val in value_list]
        if len(rows) > 1 and len(set(value_list)) < len(value_list):
            continue
        mark_of = (cited, sum((len(item_val) for item_val in value_list)) / len(value_list))
        if winner is None or mark_of > winner[0]:
            winner = (mark_of, fkey)
    return winner[1] if winner else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(item_val, dict) and ident in [str(el).strip() for el in item_val.values() if isinstance(el, str)] for item_val in node)) or any((_in_rows(item_val, ident, depth + 1) for item_val in node))
    return any((_in_rows(item_val, ident, depth + 1) for item_val in node.values())) if isinstance(node, dict) else False

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
    winner = None
    for raw_val in node.values():
        if isinstance(raw_val, dict) and len(raw_val) >= 2:
            scalars = {key_name: item_val for (key_name, item_val) in raw_val.items() if isinstance(item_val, (str, int, float)) and (not isinstance(item_val, bool))}
            if len(scalars) >= 2 and (winner is None or len(scalars) > len(winner)):
                winner = scalars
    return winner

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
    return False if 'boolean' in kinds else 'question'[:spec.get('maxLength') or 8] if 'string' in kinds else None

def _strings(node: Any) -> list[str]:
    """The texts a call's arguments hold, at any depth."""
    if isinstance(node, dict):
        return [s for item_val in node.values() for s in _strings(item_val)]
    return [s for item_val in node for s in _strings(item_val)] if isinstance(node, list) else [node] if isinstance(node, str) else []

def _says(node: Any, text: str) -> bool:
    """Whether a call's arguments carry `text` whole, however deep."""
    if isinstance(node, dict):
        return any((_says(item_val, text) for item_val in node.values()))
    return any((_says(item_val, text) for item_val in node)) if isinstance(node, list) else node == text

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Clockwork:
    """Model calls, the clock, the step budget and the store connection."""

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

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk_due = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk_due and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        return [HEAVY, LIGHT] if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD else [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            if self.model_room() < CALL_FLOOR:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
            self.calls += 1
            tries_n = 0
            while True:
                (msg_dict, http_code) = self.call_model(model, messages, max_tokens, tools)
                if msg_dict is not None:
                    return msg_dict
                passing = http_code is None or http_code in (402, 408, 425, 429) or (isinstance(http_code, int) and http_code >= 500)
                if not passing or tries_n >= (1 if http_code in (None, 408) else len(WAITS)) or self.waited + WAITS[tries_n] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries_n] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries_n] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries_n]
                tries_n += 1
                time.sleep(pause)
            if http_code == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        posted = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            posted = [dict(mo) for mo in messages]
            marks_map = [0] + ([len(posted) - 1] if len(posted) > 1 else [])
            for index in marks_map:
                payload_text = posted[index].get('content')
                if isinstance(payload_text, str) and payload_text:
                    posted[index]['content'] = [{'type': 'text', 'text': payload_text, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': posted, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        res = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        return self.call_model_continued(model, res)

    def call_model_continued(self, model, res) -> tuple[dict[str, Any] | None, Any]:
        """The rest of call_model, from the point where its earlier checks are done."""
        blob = res.data
        if isinstance(blob, dict) and blob.get('choices'):
            usage = blob.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            msg_dict = blob['choices'][0].get('message') or {}
            if msg_dict.get('content') or msg_dict.get('tool_calls'):
                return (msg_dict, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        fault = res.error or {}
        return self.call_model_continued_continued(blob, fault, model)

    def call_model_continued_continued(self, blob, fault, model) -> tuple[dict[str, Any] | None, Any]:
        """The rest of call_model_continued, from the point where its earlier checks are done."""
        http_code = fault.get('status')
        if isinstance(blob, dict) and isinstance(blob.get('error'), dict):
            http_code = _count(blob['error'].get('code'))
            fault = {'body': blob['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': http_code, 'body': str(fault.get('body') or '')[:200]})
        return (None, http_code)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            msg_dict = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((msg_dict or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        return self.act_continued(args, name, payload)

    def act_continued(self, args, name, payload) -> dict[str, Any] | None:
        """The rest of act, from the point where its earlier checks are done."""
        leftover = self.env_at + self.env_gap - time.monotonic()
        if leftover > 0:
            time.sleep(leftover)
        return self.act_continued_continued(args, name, payload)

    def act_continued_continued(self, args, name, payload) -> dict[str, Any] | None:
        """The rest of act_continued, from the point where its earlier checks are done."""
        self.env_at = time.monotonic()
        for _ in range(4):
            began = time.monotonic()
            res = _env.post_verbose('/environment/call', payload)
            self.env_spent += time.monotonic() - began
            self.env_count += 1
            if isinstance(res.data, dict):
                blob = res.data
                self.step = blob['step'] if isinstance(blob.get('step'), int) else self.step + 1
                if blob.get('finished'):
                    self.finished = True
                if name != self.roles.get('ask'):
                    blob['observation'] = self.unspelled(name, blob.get('observation'))
                noticed = blob.get('observation')
                self.note_read(name, args, noticed)
                if name == self.facts_tool() and isinstance(noticed, dict) and (not _failed(noticed)) and self.ident(noticed) and (not _fact_block(noticed)):
                    self.factless.add((name, self.ident(noticed)))
                if name in self.page_tools() and (not (isinstance(noticed, dict) and noticed.get('error'))) and (not _failed(noticed)):
                    with_examples = _dump(noticed)
                    for raw_val in args.values():
                        cited = [item_val.strip() for item_val in (raw_val if isinstance(raw_val, list) else [raw_val]) if isinstance(item_val, str) and (not isinstance(raw_val, list) or item_val.strip() in with_examples)]
                        self.opened.update(cited)
                        for pid in cited:
                            self.views.setdefault(pid, set()).add(name)
                return blob
            fault = res.error or {}
            http_code = fault.get('status')
            body = str(fault.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': http_code, 'body': body[:200]})
            if http_code == 409:
                lowered = body.lower()
                if 'finish' in lowered or 'quarantin' in lowered or 'finaliz' in lowered:
                    self.finished = True
                    return None
                hit_found = re.search('expected (\\d+)', body)
                if hit_found:
                    self.step = int(hit_found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in lowered or 'different action' in lowered:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if http_code in (400, 401, 402, 403, 404, 422):
                if http_code == 402:
                    self.broke = True
                return None
        return None

    def unspelled(self, name: str, observation: Any) -> Any:
        """The result in the form the agent reads (_restore); on a listing view, a record whose attributes sit among its own
        keys gets them gathered into one object: the keys the search rows do not have, holding plain values."""
        views = {self.roles.get('open'), self.roles.get('offer'), self.roles.get('compare')}
        acc = _fold_twins(_restore(observation, page=name in views or name != self.roles.get('search')))
        if name not in views:
            return acc
        self.row_keys.update((key_name for (pid, one_row) in self.rows.items() if pid not in self.opened for key_name in one_row))
        return self.unspelled_continued(acc)

    def unspelled_continued(self, acc) -> Any:
        """The rest of unspelled, from the point where its earlier checks are done."""
        recognized = self.row_keys
        return acc if not recognized else self.unspelled_continued_continued(acc, recognized)

    def unspelled_continued_continued(self, acc, recognized) -> Any:
        """The rest of unspelled_continued, from the point where its earlier checks are done."""

        def gathered(record: Any) -> Any:
            if not isinstance(record, dict) or _fact_block(record) or (not self.ident(record, True)):
                return record
            extra = {key_name: item_val for (key_name, item_val) in record.items() if key_name not in recognized and isinstance(item_val, (str, int, float)) and (not isinstance(item_val, bool))}
            return record if len(extra) < RECORD_FIELDS else {**{key_name: item_val for (key_name, item_val) in record.items() if key_name not in extra}, 'attributes': extra}
        if isinstance(acc, dict) and self.ident(acc):
            return gathered(acc)
        return {key_name: [gathered(rw) for rw in item_val] if isinstance(item_val, list) else item_val for (key_name, item_val) in acc.items()} if isinstance(acc, dict) else acc

    def trim(self) -> None:
        width_n = sum((len(_dump(mo.get('content'))) for mo in self.chat))
        if width_n <= HISTORY_LIMIT:
            return
        return self.trim_continued()

    def trim_continued(self) -> None:
        """The rest of trim, from the point where its earlier checks are done."""
        tools = [cur for (cur, mo) in enumerate(self.chat) if mo.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            payload_text = self.chat[index].get('content') or ''
            if len(payload_text) > 600:
                pids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', payload_text)
                pids = [cur for cur in dict.fromkeys(pids) if cur in self.rows][:20]
                self.chat[index]['content'] = payload_text[:400] + f"...(older result shortened; listings in it: {', '.join(pids)})"

    def show(self, observation: Any) -> str:
        phrase = _dump(_clip(observation, SHOW_CHARS if self.records_in(observation) else TEXT_CHARS))
        if len(phrase) > OBS_LIMIT:
            phrase = phrase[:OBS_LIMIT] + '...(cut)'
        return phrase

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.ready_to_ask():
                recap = self.interview()
                if recap:
                    self.chat.append({'role': 'user', 'content': recap + '\n' + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            msg_dict = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if msg_dict is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS:
                    break
                if self.misses > 2:
                    if self.clock() >= OUTAGE_BY:
                        break
                    time.sleep(BRIEF_GAP)
                continue
            invocations = []
            for (goods_no, invocation) in enumerate(msg_dict.get('tool_calls') or []):
                if isinstance(invocation, dict):
                    fn = invocation.get('function') if isinstance(invocation.get('function'), dict) else {}
                    raw = fn.get('arguments')
                    parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                    invocations.append({'id': str(invocation.get('id') or f'k{self.loop_calls}_{goods_no}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            payload_text = msg_dict.get('content')
            if not isinstance(payload_text, str):
                payload_text = _dump(payload_text) if payload_text else ''
            if not invocations:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': payload_text or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in invocations)) and (self.left() > 8) and (self.clock() <= ASK_BY):
                self.chat.append({'role': 'assistant', 'content': payload_text or None, 'tool_calls': invocations})
                for invocation in invocations:
                    self.chat.append({'role': 'tool', 'tool_call_id': invocation['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                recap = self.interview()
                self.chat.append({'role': 'user', 'content': (recap or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': payload_text or None, 'tool_calls': invocations})
            self.trace.append({'kind': 'turn', 'text': payload_text[:300], 'calls': [c['function'] for c in invocations]})
            replies = self.run_calls(invocations)
            for (index, (cid, phrase)) in enumerate(replies):
                if index == len(replies) - 1:
                    phrase += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': phrase})

    def answer_open_calls(self) -> None:
        """Give every call of the last model turn a reply, so that the conversation stays one the model can continue."""
        at = next((idx for idx in range(len(self.chat) - 1, -1, -1) if self.chat[idx].get('tool_calls')), None)
        if at is None:
            return
        completed_ops = {mo.get('tool_call_id') for mo in self.chat[at + 1:]}
        for invocation in self.chat[at]['tool_calls']:
            if invocation['id'] not in completed_ops:
                self.chat.append({'role': 'tool', 'tool_call_id': invocation['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
        self.chat.append({'role': 'user', 'content': self.status()})

class Brief:
    """The shopper's request, the ledger of requirements and the notes the store or shopper sends."""

    def read_request(self) -> None:
        try:
            rowset = []
            for tool_id in self.tools:
                fn = tool_id['function']
                rowset.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
            instr = BRIEF + '\nTOOLS:\n' + '\n'.join(rowset) + '\n\nREQUEST:\n' + self.request
            if self.view.get('opening') is not None:
                instr += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
            brief = self.ask_json(instr, [LIGHT], 4000) or {}
            if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
                brief = self.ask_json(instr, [HEAVY, LIGHT], 2500) or brief
            elif not self.no_credit and self.shared_shelf(brief):
                again = self.ask_json(instr, [HEAVY], 2500) or {}
                if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                    brief = again
            while not brief and (not self.no_credit) and (self.calls < ALL_CALLS) and (self.clock() < BRIEF_BY):
                time.sleep(BRIEF_GAP)
                brief = self.ask_json(instr, [LIGHT, HEAVY], 2500) or {}
            if brief:
                self.waited = 0
            self.brief = brief
            roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
            self.roles = {key_name: item_val for (key_name, item_val) in roles.items() if isinstance(item_val, str) and item_val in self.schemas}
            return self.read_request_continued(brief)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'read_request', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def read_request_continued(self, brief) -> None:
        """The rest of read_request, from the point where its earlier checks are done."""
        for fkey in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(fkey), dict):
                brief[fkey] = {}
        (buy, refuse) = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            mended = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(mended.get('buy_args'), dict) and isinstance(mended.get('refuse_args'), dict) and (mended['buy_args'] != mended['refuse_args']):
                (brief['buy_args'], brief['refuse_args']) = (mended['buy_args'], mended['refuse_args'])
        for ent in _items(brief.get('items')):
            if isinstance(ent, dict):
                quantity = _count(ent.get('quantity'))
                cnt = quantity if quantity and quantity > 0 else 1
                musts = [str(mo) for mo in _items(ent.get('musts')) if mo]
                self.items.append(Item(str(ent.get('name') or 'the product'), str(ent.get('shelf') or ''), cnt, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for (prior, goods) in enumerate(self.items, 1):
            twin = next((mo for mo in merged if goods.shelf.strip() and mo.qty == goods.qty and (_words(mo.shelf) == _words(goods.shelf)) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in mo.musts for b in goods.musts)))), None)
            if twin is None:
                merged.append(goods)
                became[prior] = len(merged)
            else:
                became[prior] = merged.index(twin) + 1
                twin.musts += [mo for mo in goods.musts if mo not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': goods.name})
        self.items = merged
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                (portion, chief) = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if portion and chief and (portion != chief) and (1 <= portion <= len(self.items)) and (1 <= chief <= len(self.items)) and ((portion, chief) not in self.bound):
                    self.bound.append((portion, chief))
        for lookup in _items(brief.get('lookups')):
            phrase = str(lookup).strip()
            if not any((phrase in goods.musts for goods in self.items)):
                fkey = _words(phrase.split(':')[0])
                phrase = next((mo for goods in self.items for mo in goods.musts if _words(mo.split(':')[0]) == fkey), phrase)
            goods_no = next((idx for (idx, goods) in enumerate(self.items, 1) if phrase in goods.musts), 1)
            if phrase and (goods_no, phrase) not in self.refs:
                self.refs.append((goods_no, phrase))
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
        step_rows = _count(brief.get('steps'))
        if step_rows is not None and 10 <= step_rows < self.max_steps:
            self.max_steps = step_rows
        for ent in _items(brief.get('process')):
            if isinstance(ent, dict) and str(ent.get('step') or '').strip():
                when = str(ent.get('when') or '').strip().lower()
                self.process.append({'step': str(ent['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(ent.get('messages')) or 0)})
        if not self.roles.get('ask'):
            return
        talk_due = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
        self.reserve = min(max(talk_due), self.msg_limit) if talk_due else 0

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for (goods_no, goods) in enumerate(self.items, 1):
            pointers = [req for (idx, req) in self.refs if idx == goods_no and req in goods.musts]
            leftover = [mo for mo in goods.musts if mo not in pointers]
            if not pointers or not leftover:
                continue
            instr = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {goods.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(leftover)
            got = self.ask_json(instr, [HEAVY, LIGHT], 1500) or {}
            relocated = [mo for mo in leftover if mo in [str(el).strip() for el in _items(got.get('describe_other'))]]
            if not relocated:
                continue
            goods.musts = [mo for mo in goods.musts if mo not in relocated]
            joined = pointers[0] + ' (that listing: ' + '; '.join(relocated) + ')'
            goods.musts = [joined if mo == pointers[0] else mo for mo in goods.musts]
            self.refs = [(idx, joined if (idx, req) == (goods_no, pointers[0]) else req) for (idx, req) in self.refs]
            self.trace.append({'kind': 'unmix', 'item': goods_no, 'moved': relocated})

    def described(self, number: int, need: str) -> bool:
        if need in [mo for (_, mo) in self.refs] or need in [str(el) for el in _items(self.brief.get('lookups'))]:
            return False
        (prop_word, _, raw_val) = need.partition(':')
        if not prop_word.strip() or len(re.findall('[a-z0-9]+', raw_val.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', raw_val, re.I):
            return False
        goods = self.items[number - 1]
        return False if any(({tok for tok in _stem(second.name) if len(tok) >= 4} & _stem(raw_val) for second in self.items if second is not goods)) else self.described_continued(goods, prop_word)

    def described_continued(self, goods, prop_word) -> bool:
        """The rest of described, from the point where its earlier checks are done."""
        shelf_field = self.category_key()
        fkeys: set[str] = set()
        readings = 0
        for rec_obj in self.full.values():
            if not (shelf_field and (not self.owns(goods, str(rec_obj.get(shelf_field) or '')))):
                chunk = _fact_block(rec_obj) or {}
                if chunk:
                    readings += 1
                    fkeys |= {_words(key_name) for key_name in chunk}
        return readings >= 3 and (not any((_stem(key_name) & _stem(prop_word) for key_name in fkeys)))

    def ledger(self) -> str:
        segments = []
        many = len(self.items) > 1
        for (goods_no, goods) in enumerate(self.items, 1):
            alive = [mo for mo in goods.musts + goods.told if mo not in goods.dropped]
            told = [mo for mo in alive if self.described(goods_no, mo)]
            needs = [mo + (f' (the page it points at gives: {self.ref_values[mo]})' if mo in self.ref_values else '') for mo in alive if mo not in told]
            head_word = f'item {goods_no}: {goods.name}' if many else f'item: {goods.name}'
            if goods.shelf:
                head_word += f' (store label: {goods.shelf})'
            head_word += f', quantity {goods.qty}'
            phrase = head_word + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                phrase += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            segments.append(phrase)
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            segments.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        selection = brief.get('pick')
        if selection == 'cheapest':
            segments.append('pick: the cheapest listing that fits (lowest price)')
        elif selection in ('most', 'least'):
            segments.append(f"pick: the {selection} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            segments.append('pick: any listing that fits everything')
        return self.ledger_continued(brief, segments)

    def ledger_continued(self, brief, segments) -> str:
        """The rest of ledger, from the point where its earlier checks are done."""
        if brief.get('in_stock_only', True) is not False:
            segments.append('in stock only')
        for wish in _items(brief.get('wishes')):
            segments.append(f'wish (tie-breaker only): {wish}')
        for rule in _items(brief.get('rules')):
            segments.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            segments.append('if nothing fits: end without naming or suggesting any listing')
        return self.ledger_continued_continued(segments)

    def ledger_continued_continued(self, segments) -> str:
        """The rest of ledger_continued, from the point where its earlier checks are done."""
        for ent in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Tern does it when you check out)', 'end': 'at an ending without a purchase (Tern does it then)'}[ent['when']]
            segments.append(f"step the request asks for, {when}: {ent['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            page_count = sorted((t for t in self.page_tools() if t != facts))
            segments.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(page_count)}" if page_count else ''))
        return '\n'.join(('- ' + p for p in segments))

    def status(self) -> str:
        left = self.left()
        phrase = f'[Tern] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            phrase += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            phrase += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            phrase += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            phrase += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            phrase += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        alive = {req for (_, req) in self.needs_now()}
        lookups = [req for (_, req) in self.refs if req in alive and req not in self.ref_values]
        if lookups and self.buys == 0:
            phrase += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        return self.status_continued(phrase)

    def status_continued(self, phrase) -> str:
        """The rest of status, from the point where its earlier checks are done."""
        if self.leads:
            phrase += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            phrase += '\n' + plan
        return self.status_continued_continued(phrase)

    def status_continued_continued(self, phrase) -> str:
        """The rest of status_continued, from the point where its earlier checks are done."""
        if self.notes:
            phrase += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        phrase += '\n[ledger]\n' + self.ledger()
        return phrase

    def needs_now(self) -> list[tuple[int, str]]:
        acc = []
        for (goods_no, goods) in enumerate(self.items, 1):
            for req in goods.musts + goods.told:
                if req not in goods.dropped:
                    acc.append((goods_no, req))
        return acc

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        hit_found: list[str] = []
        return self.spoken_continued(hit_found, observation)

    def spoken_continued(self, hit_found, observation) -> str:
        """The rest of spoken, from the point where its earlier checks are done."""

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                hit_found.append(node)
            elif isinstance(node, dict):
                for raw_val in node.values():
                    walk(raw_val, depth + 1)
            elif isinstance(node, list):
                for raw_val in node:
                    walk(raw_val, depth + 1)
        walk(observation, 0)
        return ' '.join(hit_found)[:800] if hit_found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        rowset: list[str] = []
        if isinstance(observation, dict) and (not self.ident(observation)):
            for (fkey, raw_val) in observation.items():
                if isinstance(raw_val, str) and len(raw_val.split()) >= 5:
                    rowset.append(raw_val)
                elif isinstance(raw_val, list) and raw_val and all((isinstance(item_val, str) for item_val in raw_val)):
                    rowset.extend((item_val for item_val in raw_val if len(item_val.split()) >= 5))
                elif isinstance(raw_val, dict) and (not self.ident(raw_val, False, 1)):
                    for inner in raw_val.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            rowset.append(inner)
        return [basket_line for basket_line in rowset if basket_line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        try:
            self.heard.update(lines)
            instr = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
            got = self.ask_json(instr, [LIGHT, SPARE], 2500) or {}
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
            return self.take_note_continued(changed, gone, got)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'take_note', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def take_note_continued(self, changed, gone, got) -> str:
        """The rest of take_note, from the point where its earlier checks are done."""
        added: set[str] = set()
        drops = [e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for ent in drops:
            goods = self.items[min(max(1, _count(ent.get('item')) or 1), len(self.items)) - 1]
            goal = str(ent['need']).strip()
            alive = [mo for mo in goods.musts + goods.told if mo not in goods.dropped]
            hit_at = [mo for mo in alive if mo == goal]
            if not hit_at:
                hit_at = [mo for mo in alive if _words(mo.split(':')[0]) == _words(goal.split(':')[0])][:1]
            goods.dropped.extend(hit_at or [goal])
            changed.append(f'dropped {goal}')
        for ent in adds:
            goods = self.items[min(max(1, _count(ent.get('item')) or 1), len(self.items)) - 1]
            req = str(ent['need']).strip()
            if req in goods.dropped:
                goods.dropped.remove(req)
            if req not in goods.musts + goods.told:
                goods.told.append(req)
            added.add(req)
            changed.append(f'added {req}')
        return self.take_note_continued_continued(changed, drops, gone, got)

    def take_note_continued_continued(self, changed, drops, gone, got) -> str:
        """The rest of take_note_continued, from the point where its earlier checks are done."""
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        recap = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or recap:
            self.version += 1
            self.recent = []
            self.notes.append(recap or '; '.join(changed))
        return (recap + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def evidence(self, observation: Any) -> str:
        pids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                pid = self.ident(node, False, depth)
                if pid and _fact_block(node) and (pid not in pids):
                    pids.append(pid)
                    return
                for raw_val in node.values():
                    walk(raw_val, depth + 1)
            elif isinstance(node, list):
                for raw_val in node:
                    walk(raw_val, depth + 1)
        walk(observation, 0)
        needs = self.needs_now()
        if not pids or not needs:
            return ''
        many = len(self.items) > 1
        rowset = []
        for pid in pids[:12]:
            chunk = _fact_block(self.full.get(pid) or {}) or {}
            segments = []
            for (goods_no, req) in needs[:8]:
                prop_word = req.split(':', 1)[0].strip()
                hit_list = [f'{key_name}={str(item_val)[:80]}' for (key_name, item_val) in chunk.items() if _names(_stem(prop_word), _stem(key_name))]
                tag = f'(item {goods_no}) ' if many else ''
                if hit_list:
                    segments.append(f'{tag}{prop_word} -> ' + ', '.join(hit_list[:3]))
            if segments:
                rowset.append(f'{pid}: ' + '; '.join(segments))
        return self.evidence_continued(rowset)

    def evidence_continued(self, rowset) -> str:
        """The rest of evidence, from the point where its earlier checks are done."""
        return '' if not rowset else "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(rowset)

class Shelves:
    """Reading listings, pages, attribute keys and search results."""

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing tool called with a code that one seen row's title holds gets that row's id; anything else stays as it was sent."""
        if name not in {self.roles.get(job) for job in ('open', 'offer', 'compare', 'cart')} or not self.rows:
            return args
        title = self.title_key()
        return args if not title else self.row_ids_continued(args, name, title)

    def row_ids_continued(self, args, name, title) -> dict[str, Any]:
        """The rest of row_ids, from the point where its earlier checks are done."""
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}

        def known(value: Any, key: str) -> Any:
            if not isinstance(value, str) or value.strip() in self.rows or (not 3 <= len(value.strip()) <= 30) or (not re.search('\\d', value)) or ('enum' in (prop_map.get(key) or {})):
                return value
            pattern = '(?<![a-z0-9])' + re.escape(value.strip().lower()) + '(?![a-z0-9])'
            hit_list = [cur for (cur, one_row) in self.rows.items() if re.search(pattern, str(one_row.get(title) or '').lower())]
            if len(hit_list) == 1:
                self.trace.append({'kind': 'row_id', 'sent': value, 'id': hit_list[0]})
                return hit_list[0]
            return value
        return {key: [known(item_val, key) for item_val in value] if isinstance(value, list) else known(value, key) for (key, value) in args.items()}

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        recognized = next((t for (t, ok) in self.facts_from.items() if ok and self.single_target(t)), None)
        if recognized:
            return recognized
        tool_id = self.roles.get('open')
        return tool_id if self.facts_from.get(tool_id or '') is not False and self.single_target(tool_id) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        noticed = self.compare_facts()
        return noticed is False or (noticed is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(item_val, dict) for item_val in node.values()))
        return None if not (in_list or depth == 0 or nested) else self.ident_continued(node)

    def ident_continued(self, node) -> str | None:
        """The rest of ident, from the point where its earlier checks are done."""
        hit_found = _learn_id([node])
        return node[hit_found].strip() if hit_found and ('id' in str(hit_found).lower() or node[hit_found].strip() in self.rows) else None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [idx for idx in node if isinstance(idx, dict) and len(idx) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        return True if self.id_key and all((_id_like(rw.get(self.id_key), True) for rw in rows)) else self.learn_rows_continued(rows)

    def learn_rows_continued(self, rows) -> bool:
        """The rest of learn_rows, from the point where its earlier checks are done."""
        hit_found = _learn_id(rows)
        if hit_found:
            self.id_key = hit_found
            return True
        return False

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                pid = self.ident(node, in_list, depth)
                if pid:
                    if pid not in self.rows:
                        self.seen_order.append(pid)
                        self.rows[pid] = {}
                    prior = self.rows[pid]
                    if any((prior.get(key_name) is True and item_val is False for (key_name, item_val) in node.items())):
                        self.unavailable.add(pid)
                    elif any((prior.get(key_name) is False and item_val is True for (key_name, item_val) in node.items())):
                        self.unavailable.discard(pid)
                    self.rows[pid].update(node)
                    if _fact_block(node):
                        self.full[pid] = dict(self.rows[pid])
                    elif pid in self.full:
                        self.full[pid].update(node)
                    return
                for raw_val in node.values():
                    walk(raw_val, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for raw_val in node:
                    walk(raw_val, depth + 1, rows)
        walk(observation, 0)

    def labels(self) -> set[str]:
        hit_found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    hit_found.add(option.strip().lower())
            for raw_val in schema.values():
                if isinstance(raw_val, dict):
                    walk(raw_val, depth + 1)
        for schema in self.schemas.values():
            walk(schema, 0)
        return hit_found

    def category_key(self) -> str | None:
        cnts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        return self.category_key_continued(cnts, present)

    def category_key_continued(self, cnts, present) -> str | None:
        """The rest of category_key, from the point where its earlier checks are done."""
        for one_row in self.rows.values():
            for (fkey, raw_val) in one_row.items():
                if isinstance(raw_val, str) and 0 < len(raw_val) <= 40 and (not re.search('\\d{3}', raw_val)):
                    cnts.setdefault(fkey, set()).add(raw_val)
                    present[fkey] = present.get(fkey, 0) + 1
        total = max(1, len(self.rows))
        recognized = self.labels()
        winner = None
        for (fkey, value_list) in cnts.items():
            if not (present[fkey] < 0.8 * total or len(value_list) > 40):
                listed = sum((1 for item_val in value_list if item_val.strip().lower() in recognized)) / len(value_list)
                mean = sum((len(item_val) for item_val in value_list)) / len(value_list)
                mark_of = (-listed, len(value_list), -present[fkey], -mean)
                if winner is None or mark_of < winner[0]:
                    winner = (mark_of, fkey)
        return winner[1] if winner else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        shelf_field = self.category_key()
        batch_groups: dict[str, dict[str, Any]] = {}
        for pid in self.seen_order:
            rec_obj = self.full.get(pid)
            if not rec_obj:
                continue
            chunk = _fact_block(rec_obj) or {}
            prop_label = str(rec_obj.get(shelf_field) or 'listings') if shelf_field else 'listings'
            batch_group = batch_groups.setdefault(prop_label, {'n': 0, 'keys': {}})
            batch_group['n'] += 1
            completed_ops: set[str] = set()
            for (fkey, raw_val) in chunk.items():
                nm = _words(fkey)
                if not nm or nm in completed_ops:
                    continue
                completed_ops.add(nm)
                slot = batch_group['keys'].setdefault(nm, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                slot['n'] += 1
                phrase = str(raw_val).strip().lower()
                slot['per'][pid] = phrase
                slot['plain'][pid] = _FIRST_SPELLING.get(str(raw_val).strip(), phrase).strip().lower()
                if phrase not in slot['values']:
                    slot['values'].append(phrase)
        sought = [(_words(goods.shelf or ''), _words(goods.name), {tok for tok in _stem(goods.shelf or '') | _stem(goods.name) if len(tok) >= 4}) for goods in self.items]
        matched = {prop_label for prop_label in batch_groups if any((_words(prop_label) in (shelf, nm) or {tok for tok in _stem(prop_label) if len(tok) >= 4} & long for (shelf, nm, long) in sought)) or self.owner_of(prop_label)}
        if matched:
            batch_groups = {prop_label: batch_group for (prop_label, batch_group) in batch_groups.items() if prop_label in matched}
        table = []
        for (prop_label, batch_group) in sorted(batch_groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for (nm, slot) in sorted(batch_group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                value_list = list(slot['plain'].values())
                if not value_list:
                    continue
                mean = sum((len(item_val) for item_val in value_list)) / len(value_list)
                if mean > 40 or sum((1 for item_val in value_list if ' / ' in item_val)) * 2 >= len(value_list):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', item_val)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in value_list if o is not item_val))))) > 0 for item_val in value_list))
                if len(value_list) >= 4 and len(set(value_list)) == len(value_list) and (mean > 12) and (recur * 2 < len(value_list)):
                    continue
                twin = next((slot_spec for (second, slot_spec) in kept if self.twins(second, slot_spec, nm, slot)), None)
                if twin is not None:
                    for (pid, phrase) in slot['per'].items():
                        twin['per'].setdefault(pid, phrase)
                        twin['plain'].setdefault(pid, slot['plain'][pid])
                        if phrase not in twin['values']:
                            twin['values'].append(phrase)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((nm, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for (nm, slot) in kept[:TABLE_KEYS]:
                table.append((prop_label, nm, slot['n'], [item_val[:40] for item_val in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        goods_no = self.owner_of(label)
        return '' if not goods_no else self.not_stated_continued(goods_no, name, values)

    def not_stated_continued(self, goods_no, name, values) -> str:
        """The rest of not_stated, from the point where its earlier checks are done."""
        for must in self.items[goods_no - 1].musts:
            (prop_word, _, raw_val) = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', raw_val, re.I):
                continue
            tokens = set(re.findall('[a-z0-9]{3,}', raw_val.lower()))
            with_examples = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop_word.strip() and _words(prop_word) != name and _stem(prop_word) & _stem(name) and tokens and (not tokens <= with_examples):
                return f' [not the stated {prop_word.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        (a, b) = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            (wx, wy) = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            (short, long) = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [cur for cur in spec['plain'] if cur in slot['plain'] and spec['plain'][cur].strip() and slot['plain'][cur].strip()]
        same = sum((1 for cur in both if agree(spec['plain'][cur], slot['plain'][cur])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        return same * 2 > len(both) if both else bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        shelf_field = self.category_key()
        batch_groups: dict[str, list[str]] = {}
        for pid in self.seen_order:
            prop_label = str(self.rows[pid].get(shelf_field) or '') if shelf_field else ''
            batch_groups.setdefault(prop_label, []).append(pid)
        sought = [(_words(goods.shelf or ''), _words(goods.name), {tok for tok in _stem(goods.shelf or '') | _stem(goods.name) if len(tok) >= 4}) for goods in self.items]
        matched = [ids for (prop_label, ids) in batch_groups.items() if any((_words(prop_label) in (shelf, nm) or {tok for tok in _stem(prop_label) if len(tok) >= 4} & long for (shelf, nm, long) in sought)) or self.owner_of(prop_label)]
        shortlist = matched or list(batch_groups.values())
        return sorted(shortlist, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        shape_spec = self.schemas.get(tool or '') or {}
        prop_map = shape_spec.get('properties') or {}
        text_fields = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val)]
        goal = next((key_name for key_name in shape_spec.get('required') or [] if key_name in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
        return None if not tool or not goal or any((key_name != goal for key_name in shape_spec.get('required') or [])) else goal

    def price_of(self, ident: str) -> float | None:
        raw_val = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(raw_val, dict):
            raw_val = next((item_val for item_val in raw_val.values() if isinstance(item_val, (int, float)) and (not isinstance(item_val, bool))), None)
        if isinstance(raw_val, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', raw_val)
            raw_val = float(figure.group(1).replace(',', '')) if figure else None
        return float(raw_val) if isinstance(raw_val, (int, float)) and (not isinstance(raw_val, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        raw_val = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return False if isinstance(raw_val, str) and re.search(STOCK_OUT, raw_val.strip(), re.I) else not (raw_val is False or (isinstance(raw_val, (int, float)) and (not isinstance(raw_val, bool)) and (raw_val <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            pid = self.ident(observation)
            if pid:
                return [(pid, observation)]
            hit_found: list[tuple[str, dict[str, Any]]] = []
            for raw_val in observation.values():
                if isinstance(raw_val, list):
                    hit_found += [(cur, rw) for rw in raw_val if isinstance(rw, dict) for cur in [self.ident(rw, True)] if cur]
                elif isinstance(raw_val, dict):
                    inner = self.ident(raw_val, False, 1)
                    hit_found += [(inner, raw_val)] if inner else [(cur, rw) for item_val in raw_val.values() if isinstance(item_val, list) for rw in item_val if isinstance(rw, dict) for cur in [self.ident(rw, True)] if cur]
            return hit_found
        return [(cur, rw) for rw in observation if isinstance(rw, dict) for cur in [self.ident(rw, True)] if cur] if isinstance(observation, list) else []

    def note_view(self, name: str, observation: Any) -> None:
        for (pid, rec_obj) in self.records_in(observation):
            fkeys = set(rec_obj)
            self.tool_keys.setdefault(name, set()).update(fkeys)
            (version, held_now) = self.view_keys.get(pid, (-1, set()))
            self.view_keys[pid] = (self.version, (held_now if version == self.version else set()) | fkeys)

    def shown_already(self, name: str, ident: str) -> bool:
        (version, held_now) = self.view_keys.get(ident, (-1, set()))
        fkeys = self.tool_keys.get(name)
        return bool(fkeys) and version == self.version and (fkeys <= held_now) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [key_name for (key_name, item_val) in observation.items() if isinstance(item_val, list) and len(item_val) >= 2 and all((isinstance(rw, dict) for rw in item_val))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {key_name: item_val[:3] if key_name in lists else item_val for (key_name, item_val) in observation.items()}
        instr = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(instr, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        return None if not rows else self.read_shape_continued(got, name, observation, rows)

    def read_shape_continued(self, got, name, observation, rows) -> dict[str, Any] | None:
        """The rest of read_shape, from the point where its earlier checks are done."""
        first = observation[rows][0]
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), prop_map), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        return self.read_shape_continued_continued(name, shape)

    def read_shape_continued_continued(self, name, shape) -> dict[str, Any] | None:
        """The rest of read_shape_continued, from the point where its earlier checks are done."""
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        (rows, cnt, raw_val) = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(cnt, bool) or (not isinstance(cnt, int)):
            return None
        return None if raw_val is not None and (isinstance(raw_val, bool) or not isinstance(raw_val, (str, int))) else (cnt, len(rows), shape['next_arg'], raw_val)

    def price_sorted(self, observation: Any) -> bool:
        costs = [self.price_of(cur) for (cur, _) in self.records_in(observation)]
        return len(costs) >= 3 and None not in costs and all((a <= b for (a, b) in zip(costs, costs[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        selection = self.brief.get('pick')
        plan = self.paging(name, observation) if selection in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        (cnt, with_examples, param, raw_val) = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if raw_val is None or later or cnt <= with_examples:
            return ''
        return '' if selection == 'cheapest' and self.price_sorted(observation) else self.more_pages_continued(args, cnt, name, param, raw_val, with_examples)

    def more_pages_continued(self, args, cnt, name, param, raw_val, with_examples) -> str:
        """The rest of more_pages, from the point where its earlier checks are done."""
        page_count = -(-(cnt - with_examples) // with_examples)
        if page_count > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12):
            page_count = 2
        if page_count > 2 or self.pages_read + page_count > AUTO_PAGES or self.left() - self.tail() - page_count < 12:
            return f"\n[Tern] This search matched {cnt} listings and showed {with_examples}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(raw_val)}."
        text_fields = []
        for _ in range(page_count):
            issues: list[str] = []
            invocation = _shape({**args, param: raw_val}, self.schemas.get(name), 'args', issues)
            if issues:
                break
            answer_obj = self.act(name, invocation)
            if answer_obj is None or self.finished:
                break
            self.pages_read += 1
            noticed = answer_obj.get('observation')
            self.observe(name, noticed)
            self.paged_obs.append(noticed)
            shown_text = self.show(noticed)
            self.read_cache[name + _dump(invocation)] = (self.version, self.step, shown_text)
            text_fields.append(f'{param}={_dump(raw_val)}: {shown_text}')
            after = self.paging(name, noticed)
            raw_val = after[3] if after else None
            if raw_val is None:
                break
        return self.more_pages_continued_continued(cnt, name, raw_val, text_fields, with_examples)

    def more_pages_continued_continued(self, cnt, name, raw_val, text_fields, with_examples) -> str:
        """The rest of more_pages_continued, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(text_fields)})
        leftover = f"\n[Tern] {cnt} listings matched and Tern has read {with_examples * (len(text_fields) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if raw_val is not None and text_fields else ''
        return '\n[Tern read the next pages of this search itself, one step a page]\n' + '\n'.join(text_fields) + leftover if text_fields else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            slot_spec = prop_map.get(key) if isinstance(prop_map.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(slot_spec.get('description') or ''), re.I))
        limits = self.cap_keys(name)
        caps = [k for (k, item_val) in args.items() if k in prop_map and isinstance(item_val, (int, float)) and (not isinstance(item_val, bool)) and (re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k) or (k in limits[:1] and len(limits) == 1))]
        return '' if not caps or not self.price_key else self.cap_ignored_continued(args, caps, observation, prop_map)

    def cap_ignored_continued(self, args, caps, observation, prop_map) -> str:
        """The rest of cap_ignored, from the point where its earlier checks are done."""
        caps.sort(key=lambda k: args[k])
        limit_val = args[caps[0]]
        over = [cur for (cur, _) in self.records_in(observation) if (self.price_of(cur) or 0) > limit_val]
        if not over:
            return ''
        modes = {k: item_val for (k, item_val) in args.items() if isinstance(item_val, str) and isinstance((prop_map.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in prop_map[k]['enum'] if o != item_val] for (k, item_val) in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for (k, o) in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Tern] This search did not apply {caps[0]}={limit_val:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        hit_found = {cur for noticed in observations for (cur, _) in self.records_in(noticed)}
        rows = [(self.price_of(cur), cur) for cur in hit_found if self.buyable(cur) and self.price_of(cur) is not None]
        return '' if len(rows) < 2 else '\n[Tern] buyable rows here, cheapest first: ' + ', '.join((f'{cur} ({p:g})' for (p, cur) in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        rowset = [f'{cur} at {self.price_of(cur):g}' for cur in self.cart_ids() if self.price_of(cur) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(rowset)})" if rowset else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

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
        goal = self.single_target(tool) or ''
        return bool(re.search(SHELF_ARG, _snake(goal), re.I) or re.search(QUERY_WORDS, _snake(goal), re.I)) or tool in self.shapes

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        try:
            tool_id = self.facts_tool()
            goal = self.single_target(tool_id)
            if (not tool_id or not goal) and self.compare_facts() is not True:
                return []
            text_fields = []
            sought = [cur for cur in ids if cur in self.rows and cur not in self.full and (not self.sold_out(cur))]
            side = self.roles.get('compare')
            many = self.compare_target()
            unread: list[str] = []
            while side and many and (self.compare_facts() is True) and (len(sought) >= 2) and (self.left() > keep) and (not self.finished) and (not self.rushed()):
                room = self.compare_room()
                (sent_batch, sought) = (sought[:room], sought[room:])
                faults_side: list[str] = []
                args_side = _shape({many: sent_batch}, self.schemas.get(side), 'args', faults_side)
                if faults_side:
                    sought = sent_batch + sought
                    break
                answer_obj = self.act(side, args_side)
                if answer_obj is None:
                    sought = sent_batch + sought
                    break
                noticed = answer_obj.get('observation')
                self.observe(side, noticed)
                self.note_view(side, noticed)
                rows = self.records_in(noticed)
                for (pid, rec_obj) in rows:
                    text_fields.append(f'{pid}: ' + self.show(rec_obj)[:1500])
                if rows:
                    self.compare_ok = max(self.compare_ok, len(rows))
                    unread += [cur for cur in sent_batch if cur not in self.full]
                    continue
                self.compare_bad = min(self.compare_bad or 999, len(sent_batch))
                sought = sent_batch + sought
                if self.compare_room() >= len(sent_batch):
                    break
            if not tool_id or not goal:
                return text_fields
            for pid in unread + sought:
                if self.left() <= keep or self.finished or self.facts_from.get(tool_id) is False or self.rushed():
                    break
                if (tool_id, pid) in self.factless:
                    continue
                issues: list[str] = []
                call_args = _shape({goal: pid}, self.schemas.get(tool_id), 'args', issues)
                if issues:
                    break
                answer_obj = self.act(tool_id, call_args)
                if answer_obj is None:
                    break
                noticed = answer_obj.get('observation')
                self.observe(tool_id, noticed)
                self.note_view(tool_id, noticed)
                if isinstance(noticed, dict) and self.ident(noticed):
                    self.note_facts(tool_id, bool(_fact_block(noticed)))
                text_fields.append(f'{pid}: ' + self.show(noticed)[:1500])
            return self.fill_facts_continued(text_fields)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'fill_facts', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def fill_facts_continued(self, text_fields) -> list[str]:
        """The rest of fill_facts, from the point where its earlier checks are done."""
        if text_fields:
            self.trace.append({'kind': 'fill_facts', 'n': len(text_fields)})
        return text_fields

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        return self.show_links_continued(ids, keep, plan)

    def show_links_continued(self, ids, keep, plan) -> None:
        """The rest of show_links, from the point where its earlier checks are done."""
        (tool_id, base, fld) = plan
        for pid in ids:
            marks_map = self.fit_index()['marks']
            code_str = (marks_map.get(pid) or {}).get('more')
            if not code_str or code_str in self.link_tried or any((mark.get('own') == code_str for mark in marks_map.values())):
                continue
            if self.left() <= keep or self.finished or self.rushed():
                return
            self.link_tried.add(code_str)
            issues: list[str] = []
            call_args = _shape({**base, fld: code_str}, self.schemas.get(tool_id), 'args', issues)
            if issues:
                return
            answer_obj = self.act(tool_id, call_args)
            if answer_obj is None:
                return
            self.observe(tool_id, answer_obj.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': pid, 'code': code_str})
            marks_map = self.fit_index()['marks']
            self.fill_facts([cur for (cur, mark) in marks_map.items() if mark.get('own') == code_str], max(keep, self.tail() + 6))

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for (tool_id, shape_spec) in self.schemas.items():
            prop_map = (shape_spec or {}).get('properties') or {}
            caps = self.cap_keys(tool_id)
            shelves = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val) and re.search(SHELF_ARG, _snake(key_name), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            learned = [(dict(mode), held) for ((fkey, mode), held) in self.cap_modes(tool_id).items() if fkey == caps[0]]
            options_args = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and isinstance(item_val.get('enum'), list) and (len(item_val['enum']) >= 2)]
            single = len(options_args) == 1 and options_args[0] in ((shape_spec or {}).get('required') or [])
            base: dict[str, Any] = {}
            for (fkey, slot_spec) in prop_map.items():
                options = slot_spec.get('enum') if isinstance(slot_spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                selection = next((mode[fkey] for (mode, held) in learned if held and fkey in mode), None)
                if selection is None:
                    selection = next((a[fkey] for a in reversed(self.searched) if shelves[0] in a and fkey in a and (not any((key_name != shelves[0] and (not (isinstance(prop_map.get(key_name), dict) and 'enum' in prop_map[key_name])) and isinstance(item_val, str) and item_val.strip() and re.search(QUERY_WORDS, _snake(key_name), re.I) for (key_name, item_val) in a.items())))), None)
                    cited = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                    worded = {a[fkey] for a in self.searched if fkey in a and any((isinstance(item_val, str) and item_val.strip() and re.search(QUERY_WORDS, _snake(key_name), re.I) for (key_name, item_val) in a.items()))}
                    unworded = [o for o in options if o not in worded]
                    selection = selection if selection is not None else cited[0] if len(cited) == 1 else unworded[0] if single and worded and (len(unworded) == 1) else None
                spoiled = {mode[fkey] for (mode, held) in learned if not held and fkey in mode} if single else set()
                if spoiled and (selection is None or selection in spoiled):
                    leftover = [o for o in options if o not in spoiled]
                    selection = leftover[0] if len(leftover) == 1 else None
                if selection is None:
                    return None
                base[fkey] = selection
            return (tool_id, base, shelves[0], caps[0])
        return None

    def cap_keys(self, tool: str) -> list[str]:
        """The numeric arguments of `tool` that may cap a listing's price: those a search has shown holding a limit, else those whose
        name says a maximum, else those whose name says a price or budget."""
        prop_map = (self.schemas.get(tool) or {}).get('properties') or {}
        numbers = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and {'integer', 'number'} & set(_kinds(item_val)) and (not re.search(CAP_NOT, key_name, re.I)) and (not re.search(CAP_LOW, _snake(key_name), re.I))]
        held = {fkey for ((fkey, _), ok) in self.cap_modes(tool).items() if ok}
        return self.cap_keys_continued(held, numbers)

    def cap_keys_continued(self, held, numbers) -> list[str]:
        """The rest of cap_keys, from the point where its earlier checks are done."""
        cited = [key_name for key_name in numbers if re.search(CAP_ARG, _snake(key_name), re.I)]
        priced = [key_name for key_name in numbers if re.search(CAP_PRICED, _snake(key_name), re.I)]
        for batch_group in ([key_name for key_name in numbers if key_name in held], [key_name for key_name in cited if key_name in priced], priced, cited):
            if batch_group:
                return batch_group
        return []

    def cap_modes(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """What the searches of `tool` so far show of each numeric argument as a price limit, by argument and mode (the call's option
        arguments): False once a call's rows held one priced above its value; True when a call's rows all sat at or below it and rows drawn
        at random from the known rows of their labels would rarely have (CAP_CHANCE), so the limit, not the query, kept dearer rows out."""
        prop_map = (self.schemas.get(tool) or {}).get('properties') or {}
        shelf_field = self.category_key()
        return self.cap_modes_continued(prop_map, shelf_field, tool)

    def cap_modes_continued(self, prop_map, shelf_field, tool) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """The rest of cap_modes, from the point where its earlier checks are done."""
        acc: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        costs: dict[str, list[float]] = {}
        return self.cap_modes_continued_continued(acc, costs, prop_map, shelf_field, tool)

    def cap_modes_continued_continued(self, acc, costs, prop_map, shelf_field, tool) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """The rest of cap_modes_continued, from the point where its earlier checks are done."""
        for (pid, one_row) in self.rows.items():
            price = self.price_of(pid)
            if price is not None:
                costs.setdefault(str(one_row.get(shelf_field) or '').strip().lower() if shelf_field else '', []).append(price)
        for (mark, (nm, call_args)) in self.read_args.items():
            if not nm != tool:
                rows = [cur for (cur, _) in self.records_in(self.read_calls.get(mark))]
                with_examples = [self.price_of(cur) for cur in rows]
                if not rows or None in with_examples:
                    continue
                mode = tuple(sorted(((key_name, item_val) for (key_name, item_val) in call_args.items() if isinstance(prop_map.get(key_name), dict) and isinstance(prop_map[key_name].get('enum'), list) and isinstance(item_val, (str, int, float)))))
                prop_labels = {str((self.rows.get(cur) or {}).get(shelf_field) or '').strip().lower() for cur in rows} if shelf_field else set()
                for (fkey, raw_val) in call_args.items():
                    if not (fkey not in prop_map or isinstance(raw_val, bool) or (not isinstance(raw_val, (int, float)))):
                        shortlist = [p for prop_label in prop_labels for p in costs.get(prop_label, [])]
                        if max(with_examples) > raw_val:
                            acc[fkey, mode] = False
                        elif acc.get((fkey, mode)) is not False and shortlist and ((sum((p <= raw_val for p in shortlist)) / len(shortlist)) ** len(rows) < CAP_CHANCE):
                            acc[fkey, mode] = True
        return acc

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        shelf_field = self.category_key()
        return self.list_below_continued(cart, plan, shelf_field)

    def list_below_continued(self, cart, plan, shelf_field) -> None:
        """The rest of list_below, from the point where its earlier checks are done."""
        if not plan or not shelf_field:
            return
        (tool_id, base, shelf, limit_val) = plan
        for basket_line in cart:
            price = self.price_of(basket_line)
            prop_label = str((self.rows.get(basket_line) or {}).get(shelf_field) or '')
            if price is None or price < 2 or (not prop_label) or ((basket_line, price) in self.capped):
                continue
            self.capped.add((basket_line, price))
            limit = int(price) - 1 if float(price).is_integer() else int(price)
            tried: set[int] = set()
            for _ in range(WALK_CALLS):
                if self.left() <= self.tail() + (WALK_ROOM if tried else 7) or self.finished or self.rushed():
                    return
                if limit < 1 or limit in tried:
                    break
                tried.add(limit)
                issues: list[str] = []
                call_args = _shape({**base, shelf: prop_label, limit_val: limit}, self.schemas.get(tool_id), 'args', issues)
                if issues:
                    return
                answer_obj = self.act(tool_id, call_args)
                if answer_obj is None:
                    return
                noticed = answer_obj.get('observation')
                self.observe(tool_id, noticed)
                rows = self.records_in(noticed)
                self.trace.append({'kind': 'list_below', 'line': basket_line, 'price': price, 'limit': limit, 'rows': len(rows)})
                page = self.paging(tool_id, noticed)
                if not rows or any(((self.price_of(cur) or 0) > limit for (cur, _) in rows)) or (not page) or (page[0] <= page[1]) or self.price_sorted(noticed):
                    break
                if page[0] <= 2 * page[1]:
                    invocation = _shape({**call_args, page[2]: page[3]}, self.schemas.get(tool_id), 'args', issues) if page[3] is not None else None
                    further = None if issues or invocation is None or self.left() <= self.tail() + 7 or self.rushed() else self.act(tool_id, invocation)
                    if further is not None:
                        self.observe(tool_id, further.get('observation'))
                    break
                lowered = min((p for (cur, _) in rows for p in [self.price_of(cur)] if p is not None), default=None)
                if lowered is None:
                    break
                limit = int(lowered) - 1 if float(lowered).is_integer() else int(lowered)

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        shelf_field = self.category_key()
        if mark in self.cheaper_marks or not shelf_field:
            return []
        self.cheaper_marks.add(mark)
        return self.cheaper_rows_continued(shelf_field)

    def cheaper_rows_continued(self, shelf_field) -> list[str]:
        """The rest of cheaper_rows, from the point where its earlier checks are done."""
        cart = self.cart_ids()
        self.list_below(cart)
        return self.cheaper_rows_continued_continued(cart, shelf_field)

    def cheaper_rows_continued_continued(self, cart, shelf_field) -> list[str]:
        """The rest of cheaper_rows_continued, from the point where its earlier checks are done."""
        selected: list[str] = []
        room = self.compare_room() if self.compare_facts() is True else 0
        for basket_line in cart:
            price = self.price_of(basket_line)
            sort_kind = str((self.rows.get(basket_line) or {}).get(shelf_field) or '').strip().lower()
            if price is None or not sort_kind:
                continue
            shortlist = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(shelf_field) or '').strip().lower() == sort_kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            shortlist.sort(key=lambda i: self.price_of(i) or 0.0)
            lowest = [i for i in shortlist if i in self.value_rows][:1]
            shortlist = lowest + [i for i in shortlist if i not in lowest]
            selected += [i for i in shortlist[:room or 2] if i not in selected]
        return self.fill_facts(selected[:room * len(cart) if room else min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 6)

    @_soft('')
    def below_cart(self) -> str:
        """The buyable rows priced below a cart line, of its label, that nobody has read yet."""
        shelf_field = self.category_key()
        cart = self.cart_ids()
        rowset = []
        for basket_line in cart:
            price = self.price_of(basket_line)
            sort_kind = str((self.rows.get(basket_line) or {}).get(shelf_field) or '').strip().lower() if shelf_field else ''
            if price is None or not sort_kind:
                continue
            shortlist = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(shelf_field) or '').strip().lower() == sort_kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            shortlist.sort(key=lambda i: self.price_of(i) or 0.0)
            rowset += [f'{i} at {self.price_of(i):g}' for i in shortlist[:CHEAPER_SHOWN]]
        return 'Still unread and below a cart line: ' + ', '.join(rowset) + '.' if rowset else ''

    def title_key(self) -> str | None:
        """The row key holding a listing's name: of the short texts most rows carry and few rows share, with more than one word
        on average and about as many different values as any such key has (a seller's name repeats where a listing's does not),
        the shortest."""
        shelf_field = self.category_key()
        totals: dict[str, list[Any]] = {}
        return self.title_key_continued(shelf_field, totals)

    def title_key_continued(self, shelf_field, totals) -> str | None:
        """The rest of title_key, from the point where its earlier checks are done."""
        for one_row in self.rows.values():
            for (fkey, raw_val) in one_row.items():
                if fkey != shelf_field and isinstance(raw_val, str) and (len(raw_val) <= TITLE_CHARS):
                    slot = totals.setdefault(fkey, [0, 0, 0, set()])
                    slot[0] += len(raw_val.split())
                    slot[1] += 1
                    slot[2] += len(raw_val)
                    slot[3].add(raw_val)
        kept = [(slot[2] / slot[1], fkey, len(slot[3])) for (fkey, slot) in totals.items() if slot[1] * 5 >= len(self.rows) * 4 and slot[0] >= slot[1] * 2 and (len(slot[3]) * 2 >= slot[1])]
        most = max((distinct for (_, _, distinct) in kept), default=0)
        sorted_ids = [(mean, fkey) for (mean, fkey, distinct) in kept if distinct * 10 >= most * 9]
        return min(sorted_ids)[1] if sorted_ids else None

    def rows_of_item(self, number: int) -> int:
        shelf_field = self.category_key()
        goods = self.items[number - 1]
        return sum((1 for one_row in self.rows.values() if not shelf_field or self.owns(goods, str(one_row.get(shelf_field) or ''))))

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        """One spelling per call: words in any case and spacing, a number at its first value left out."""
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}
        kept: dict[str, Any] = {}
        return self.call_mark_continued(args, kept, name, prop_map)

    def call_mark_continued(self, args, kept, name, prop_map) -> str:
        """The rest of call_mark, from the point where its earlier checks are done."""
        for (fkey, raw_val) in sorted(args.items()):
            slot_spec = prop_map.get(fkey) if isinstance(prop_map.get(fkey), dict) else {}
            first = raw_val in (slot_spec.get('minimum'), slot_spec.get('default')) or (raw_val == 1 and fkey == self.page_args.get(name) and (slot_spec.get('minimum') is None) and (slot_spec.get('default') is None))
            if isinstance(raw_val, int) and (not isinstance(raw_val, bool)) and first:
                continue
            kept[fkey] = ' '.join(raw_val.lower().split()) if isinstance(raw_val, str) else raw_val
        return name + _dump(kept)

    def note_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """Remember what a search or listing call showed, whoever made it."""
        if name in self.page_tools() or name in {self.roles.get(rw) for rw in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.query_is_new(name, args)
        if not (isinstance(seen, (dict, list)) and (not _failed(seen))):
            return
        self.read_calls[self.call_mark(name, args)] = seen
        self.read_args[self.call_mark(name, args)] = (name, dict(args))

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this search (its arguments apart from numbers) is the first of its kind, and remember it."""
        mark = _dump({key_name: item_val for (key_name, item_val) in args.items() if isinstance(item_val, bool) or not isinstance(item_val, int)})
        noticed = self.page_queries.setdefault(name, set())
        fresh = mark not in noticed
        noticed.add(mark)
        return fresh

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A new search that opened on a later page shows few rows and no next page, which reads as all there is: race 15's models
        asked the browse tool for its highest page in 1716 of 2845 calls, seeing 7 of 342 rows and "next: none". When such a call
        reports more rows than it shows, the same call at the lowest value of its numeric argument tells: rows with other ids,
        at least as many, mean that argument is the page."""
        if name in self.page_args or not isinstance(observation, dict):
            return ''
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}
        numeric = [key_name for (key_name, item_val) in args.items() if isinstance(item_val, int) and (not isinstance(item_val, bool)) and (key_name in prop_map) and (item_val > (prop_map[key_name].get('minimum') if isinstance(prop_map[key_name].get('minimum'), int) else 1))]
        if len(numeric) != 1 or self.left() <= self.tail() + 4:
            return ''
        shape = self.read_shape(name, observation)
        if not shape or not shape.get('count') or shape.get('next') is None or (observation.get(shape['next']) is not None):
            return ''
        rows = observation.get(shape['rows'])
        cnt = observation.get(shape['count'])
        if not isinstance(rows, list) or not isinstance(cnt, int) or cnt <= len(rows):
            return ''
        fkey = numeric[0]
        lowered = prop_map[fkey].get('minimum') if isinstance(prop_map[fkey].get('minimum'), int) else 1
        issues: list[str] = []
        invocation = _shape({**args, fkey: lowered}, self.schemas.get(name), 'args', issues)
        if issues:
            return ''
        answer_obj = self.act(name, invocation)
        if answer_obj is None:
            return ''
        noticed = answer_obj.get('observation')
        self.observe(name, noticed)
        fresh_needs = noticed.get(shape['rows']) if isinstance(noticed, dict) else None
        if not isinstance(fresh_needs, list) or len(fresh_needs) < len(rows) or {cur for (cur, _) in self.records_in(noticed)} & {cur for (cur, _) in self.records_in(observation)}:
            return ''
        self.page_args[name] = fkey
        self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': fkey})
        return f'\n[Tern] {fkey} is the page number of this search: the call above showed its last page. Page {lowered} of the same search:\n' + self.show(noticed)

    @_soft(False)
    def names_code(self, args: dict[str, Any]) -> bool:
        """Whether a search names a model or part code some listing's text gave as a code."""
        spoken_text = {c for item_val in args.values() if isinstance(item_val, str) for c in _codes(item_val)}
        if not spoken_text:
            return False
        recognized: set[str] = set()
        for mark in self.fit_index()['marks'].values():
            recognized.update(mark.get('codes') or [])
            recognized.update(mark.get('works') or [])
        return bool(spoken_text & recognized)

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        """A new search starts on its first page once the tool's page argument is known."""
        fkey = self.page_args.get(name)
        if not fkey or not fresh or (not isinstance(args.get(fkey), int)):
            return args
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}
        lowered = prop_map.get(fkey, {}).get('minimum') if isinstance(prop_map.get(fkey, {}).get('minimum'), int) else 1
        return {**args, fkey: lowered}

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A long words search that found nothing is run again with its first words, no model turn needed. Race 15: 54% of
        six-word queries and 42% of five-word ones came back empty (search wants every word), 10% of one-word ones."""
        plan = self.value_template()
        if not plan or plan[0] != name or self.left() <= self.tail() + SHORT_KEEP or self.finished or (not isinstance(observation, dict)):
            return ''
        fld = plan[2]
        phrase = args.get(fld)
        if not isinstance(phrase, str) or self.records_in(observation) or len(phrase.split()) < 3:
            return ''
        for raw_val in observation.values():
            if isinstance(raw_val, (list, str)) and raw_val or (isinstance(raw_val, (int, float)) and (not isinstance(raw_val, bool)) and (raw_val > 0)):
                return ''
        short = ' '.join([tok for tok in phrase.split() if not _codes(tok)][:2])
        if not short:
            return ''
        issues: list[str] = []
        invocation = _shape({key_name: item_val for (key_name, item_val) in {**args, fld: short}.items() if isinstance(item_val, bool) or not isinstance(item_val, int)}, self.schemas.get(name), 'args', issues)
        if issues:
            return ''
        answer_obj = self.act(name, invocation)
        if answer_obj is None:
            return ''
        noticed = answer_obj.get('observation')
        self.observe(name, noticed)
        self.trace.append({'kind': 'shorter', 'from': phrase, 'to': short, 'rows': len(self.records_in(noticed))})
        return f'\n[Tern] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(noticed)

    @_soft('')
    def first_moves(self) -> str:
        """Search each item once and read a few of its listings before the model's first turn, so the interview can start at once.

        A race model turn costs 5-10 s (race 15: 207 s mean per task, 16 of 180 and 30 of 180 episodes of two agents killed at 300 s,
        all on 3+ items) and the first turns only do this; one-item tasks have time, and keep the model's own first searches. Each
        item's search is its required values with its noun, then its name when that finds nothing. Returns what the store showed; '' when the search tool cannot be used without a model's reading of its
        modes, and the model then starts as usual."""
        try:
            plan = self.value_template()
            if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
                return ''
            (tool_id, base, fld) = plan
            self.env_gap = OPEN_GAP
            with_examples: list[str] = []
            stop_at = self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2
            for (goods_no, goods) in enumerate(self.items, 1):
                sought = [mo.split(':', 1)[1].strip() for mo in goods.musts if ':' in mo and mo not in [req for (_, req) in self.refs]]
                sought = [tok for tok in sought if tok and len(tok.split()) <= 3 and (not re.search(OPEN_NOT, tok, re.I))]
                noun = goods.name if len(goods.name.split()) <= 3 else ' '.join(goods.name.split()[-2:])
                tries_n = list(dict.fromkeys(([' '.join(sought[:2] + [noun])] if sought else []) + [noun]))
                for q_text in tries_n:
                    issues: list[str] = []
                    call_args = _shape({**base, fld: q_text}, self.schemas.get(tool_id), 'args', issues)
                    if issues or self.finished or self.step >= stop_at:
                        break
                    answer_obj = self.act(tool_id, call_args)
                    if answer_obj is None:
                        return '\n'.join(with_examples)
                    noticed = answer_obj.get('observation')
                    self.observe(tool_id, noticed)
                    if self.records_in(noticed):
                        self.searched = (self.searched + [dict(call_args)])[-8:]
                        with_examples.append(f'{q_text}: ' + self.show(noticed))
                        with_examples += self.more_rows_for_keys(tool_id, call_args, noticed, stop_at, goods_no)
                        if self.rows_of_item(goods_no) >= OPEN_EACH:
                            break
                    else:
                        with_examples.append(f"{q_text}: {('the store refused the call: ' + self.show(noticed) if _failed(noticed) else 'no rows')}")
            if self.rows and self.roles.get('ask') and (self.msgs_free() > 0):
                start = self.step
                hit_found = [idx for idx in range(1, len(self.items) + 1) if self.rows_of_item(idx)]
                while self.step - start < self.max_steps // OPEN_SHARE and self.step < stop_at and (not self.finished):
                    if min((self.records_of_item(idx) for idx in hit_found), default=OPEN_EACH) >= OPEN_EACH:
                        break
                    had = len(self.full)
                    self.widen()
                    if len(self.full) == had:
                        break
            self.env_gap = ENV_GAP
            self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
            with_examples += ['read in full: ' + self.page_text(pid, rec_obj) for (pid, rec_obj) in self.full.items()]
            proof = self.evidence(list(self.full.values()))
            return '\n'.join(with_examples + ([proof] if proof else []))
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'first_moves', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        """One read listing for the model: its attributes first, since those are what it judges."""
        chunk = _fact_block(record) or {}
        leftover = {key_name: item_val for (key_name, item_val) in record.items() if item_val is not chunk and (not isinstance(item_val, (dict, list)))}
        return f'{ident}: attributes ' + _dump(_clip(chunk, 120))[:PAGE_ATTR_CHARS] + ' | ' + _dump(_clip(leftover, 200))[:PAGE_REST_CHARS]

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        """One further page when a first page shows too few of the item's listings to tell what they carry; what it showed."""
        page = self.paging(tool, seen)
        if not page or page[3] is None or self.rows_of_item(number) >= OPEN_EACH or self.finished or (self.step >= stop_at):
            return []
        issues: list[str] = []
        invocation = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', issues)
        return [] if issues else self.more_rows_for_keys_continued(invocation, tool)

    def more_rows_for_keys_continued(self, invocation, tool) -> list[str]:
        """The rest of more_rows_for_keys, from the point where its earlier checks are done."""
        answer_obj = self.act(tool, invocation)
        if answer_obj is None:
            return []
        self.observe(tool, answer_obj.get('observation'))
        return [f'next page: ' + self.show(answer_obj.get('observation'))]

    def observe(self, name: str, observation: Any) -> str:
        try:
            self.absorb(observation)
            rowset = self.loose_lines(observation)
            if not rowset:
                return ''
            meaning = self.take_note(rowset)
            if meaning:
                self.queued.append(f'(seen on {name}) {meaning}')
            return meaning
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'observe', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def widen(self) -> None:
        try:
            tool_id = self.roles.get('compare')
            shape_spec = self.schemas.get(tool_id or '') or {}
            prop_map = shape_spec.get('properties') or {}
            goal = next((key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'array' in _kinds(item_val)), None)
            if not tool_id or not goal or any((key_name != goal for key_name in shape_spec.get('required') or [])) or self.compare_bare():
                self.widen_singly()
                return
            if self.left() <= 12:
                return
            limit_val = min(prop_map[goal].get('maxItems') if isinstance(prop_map[goal].get('maxItems'), int) else 4, self.compare_room())
            sorted_ids = sorted(self.item_groups(), key=lambda ids: sum((1 for cur in ids if cur in self.full)))
            lanes = [[cur for cur in ids if cur not in self.full] for ids in sorted_ids]
            return self.widen_continued(goal, lanes, limit_val, shape_spec, tool_id)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'widen', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def widen_continued(self, goal, lanes, limit_val, shape_spec, tool_id) -> None:
        """The rest of widen, from the point where its earlier checks are done."""
        selected: list[str] = []
        for round_of in range(limit_val):
            selected += [track[round_of] for track in lanes if round_of < len(track)]
        selected = selected[:limit_val]
        if len(selected) < 2:
            self.fill_facts(selected, 12)
            return
        return self.widen_continued_continued(goal, selected, shape_spec, tool_id)

    def widen_continued_continued(self, goal, selected, shape_spec, tool_id) -> None:
        """The rest of widen_continued, from the point where its earlier checks are done."""
        issues: list[str] = []
        call_args = _shape({goal: selected}, shape_spec, 'args', issues)
        if issues:
            return
        before = len(self.full)
        answer_obj = self.act(tool_id, call_args)
        if not answer_obj is not None:
            return
        noticed = answer_obj.get('observation')
        self.observe(tool_id, noticed)
        self.note_view(tool_id, noticed)
        rows = self.records_in(noticed)
        if rows:
            self.facts_from[tool_id] = self.facts_from.get(tool_id, False) or any((_fact_block(rw) for (_, rw) in rows))
            self.compare_ok = max(self.compare_ok, len(rows))
        else:
            self.compare_bad = min(self.compare_bad or 999, len(selected))
        self.trace.append({'kind': 'widen', 'ids': selected, 'opened': len(self.full) - before})
        if rows and self.facts_from.get(tool_id) is False:
            self.widen_singly()
            return
        if len(self.full) == before and len(selected) > 2 and (self.left() > 12):
            issues = []
            call_args = _shape({goal: selected[:2]}, shape_spec, 'args', issues)
            if not issues:
                answer_obj = self.act(tool_id, call_args)
                if answer_obj is not None:
                    self.observe(tool_id, answer_obj.get('observation'))
                    self.compare_ok = max(self.compare_ok, len(self.records_in(answer_obj.get('observation'))))

    def widen_singly(self) -> None:
        try:
            tool_id = self.facts_tool()
            goal = self.single_target(tool_id)
            if not tool_id or not goal:
                return
            sought = {tok for (_, req) in self.needs_now() for tok in re.findall('[a-z0-9]{3,}', req.split(':', 1)[-1].lower())}
            return self.widen_singly_continued(sought)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'widen_singly', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def widen_singly_continued(self, sought) -> None:
        """The rest of widen_singly, from the point where its earlier checks are done."""
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for pids in self.item_groups():
            opened = sum((1 for i in pids if i in self.full))
            shortlist = [i for i in pids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                shortlist.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            shortlist.sort(key=lambda i: -len(sought & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(shortlist[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [track[key_name] for key_name in range(4) for (track, head_word) in zip(lanes, heads) if key_name < min(head_word, len(track))]
        further = [track[key_name] for key_name in range(each) for (track, head_word) in zip(lanes, heads) if head_word <= key_name < len(track)]
        limit_val = max(5, 2 * len(lanes) + 2)
        readings = self.fill_facts(first[:limit_val][:8], 22)
        self.fill_facts((first[limit_val:] + further)[:max(0, 10 - len(readings))], 30)

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool_id = self.roles.get('search')
        shape_spec = self.schemas.get(tool_id or '') or {}
        return self.search_template_continued(shape_spec, tool_id)

    def search_template_continued(self, shape_spec, tool_id) -> tuple[str, dict[str, Any], str] | None:
        """The rest of search_template, from the point where its earlier checks are done."""
        prop_map = shape_spec.get('properties') or {}
        text_fields = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val)]
        cited = [key_name for key_name in text_fields if re.search(QUERY_WORDS, _snake(key_name), re.I)]
        for call_args in reversed(self.searched):
            fields = [key_name for key_name in cited or text_fields if isinstance(call_args.get(key_name), str) and re.search('[a-z]', call_args[key_name], re.I)]
            if not cited:
                prop_labels = {str(rw.get(self.category_key()) or '').strip().lower() for rw in self.rows.values()} - {''} if self.category_key() else set()
                fields = [key_name for key_name in fields if len({str(a.get(key_name)) for a in self.searched if a.get(key_name)}) >= 2 and any((len(str(a.get(key_name)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(key_name) or '').strip().lower() in prop_labels for a in self.searched)))]
            if fields and tool_id:
                fld = fields[0]
                reserve_n = {key_name: item_val for (key_name, item_val) in call_args.items() if key_name != fld and (isinstance((prop_map.get(key_name) or {}).get('enum'), list) or key_name in (shape_spec.get('required') or []))}
                shelf_field = self.category_key()
                shelves = {_words(goods.shelf or '') for goods in self.items} | {_words(goods.name) for goods in self.items}
                shelves |= {_words(str(one_row.get(shelf_field) or '')) for one_row in self.rows.values()} if shelf_field else set()
                reserve_n = {key_name: item_val for (key_name, item_val) in reserve_n.items() if key_name in (shape_spec.get('required') or []) or not isinstance(item_val, str) or (_words(item_val) not in shelves and (not any(({tok for tok in _stem(item_val) if len(tok) >= 4} & {tok for tok in _stem(s) if len(tok) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(fld), str) and re.search('[a-z]', a[fld], re.I)]
                for key_name in list(reserve_n):
                    if isinstance((prop_map.get(key_name) or {}).get('enum'), list):
                        used = [_dump(a[key_name]) for a in worded if key_name in a]
                        if used:
                            winner = max(used, key=used.count)
                            reserve_n[key_name] = next((a[key_name] for a in worded if key_name in a and _dump(a[key_name]) == winner))
                return (tool_id, reserve_n, fld)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        acc = []
        told = self.items[number - 1].told
        return self.value_words_continued(acc, number, told)

    def value_words_continued(self, acc, number, told) -> list[tuple[str, bool]]:
        """The rest of value_words, from the point where its earlier checks are done."""
        for (idx, req) in self.needs_now():
            if not (idx != number or req in [mo for (_, mo) in self.refs] or ':' not in req):
                raw_val = req.split(':', 1)[1].strip()
                edge = re.search(FLOOR, raw_val, re.I)
                if edge:
                    if req in told:
                        acc.append((edge.group(1).strip(), True))
                    continue
                if not raw_val or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', raw_val, re.I):
                    continue
                acc.append((raw_val, req in told))
        return acc

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.search_template()
        tool_id = self.roles.get('search')
        if plan or not tool_id:
            return plan
        shape_spec = self.schemas.get(tool_id) or {}
        prop_map = shape_spec.get('properties') or {}
        required = shape_spec.get('required') or []
        return self.value_template_continued(prop_map, required, tool_id)

    def value_template_continued(self, prop_map, required, tool_id) -> tuple[str, dict[str, Any], str] | None:
        """The rest of value_template, from the point where its earlier checks are done."""
        told = self.brief.get('search_words') if isinstance(self.brief.get('search_words'), dict) else {}
        (spoken_text, mended) = (told.get('field') if isinstance(told.get('field'), str) else None, told.get('args') if isinstance(told.get('args'), dict) else {})
        named_tool = told.get('tool') if isinstance(told.get('tool'), str) and told.get('tool') in self.schemas else tool_id
        slot_spec = (self.schemas.get(named_tool) or {}).get('properties') or {}
        if spoken_text in slot_spec and 'string' in _kinds(slot_spec[spoken_text]) and ('enum' not in slot_spec[spoken_text]):
            issues: list[str] = []
            base = _shape({**mended, spoken_text: 'x'}, self.schemas.get(named_tool), 'args', issues)
            if not issues:
                base.pop(spoken_text, None)
                return (named_tool, base, spoken_text)
        cited = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val) and re.search(QUERY_WORDS, _snake(key_name), re.I)]
        last = self.searched[-1] if self.searched else {}
        enums = {key_name: item_val['enum'] for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and isinstance(item_val.get('enum'), list) and (len(item_val['enum']) >= 2)}
        modes = []
        for (fkey, options) in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(fkey)]
            if len(reads) == 1:
                modes.append((fkey, reads[0]))
            elif fkey in required and len(options) == 2 and (len(others) == 1):
                modes.append((fkey, others[0]))
        if len(cited) != 1 or len(modes) > 1 or (enums and (not modes)):
            return None
        base = {key_name: last[key_name] for key_name in required if key_name in last and key_name != cited[0]}
        return self.value_template_continued_continued(base, cited, modes, tool_id)

    def value_template_continued_continued(self, base, cited, modes, tool_id) -> tuple[str, dict[str, Any], str] | None:
        """The rest of value_template_continued, from the point where its earlier checks are done."""
        base.update(modes)
        return (tool_id, base, cited[0])

    def value_search(self, numbers: list[int]) -> str:
        try:
            plan = self.value_template()
            if not plan or self.left() < self.tail() + VALUE_ROOM:
                return ''
            (tool_id, base, fld) = plan
            noticed: list[str] = []
            budget = min(8, self.left() - self.tail() - VALUE_KEEP)
            for goods_no in numbers:
                pairs = self.value_words(goods_no)
                goods = self.items[goods_no - 1]
                value_list = [item_val for (item_val, _) in pairs]
                fresh = [item_val for (item_val, told) in pairs if told]
                if not fresh or budget <= 2:
                    continue
                lead = [item_val for item_val in fresh if len(item_val.split()) == 1 or re.fullmatch(AMOUNT, item_val, re.I)]
                value_list = lead + [item_val for item_val in value_list if item_val not in lead]
                stated = [i for (i, item_val) in enumerate(value_list) if item_val not in fresh]
                drop = stated[-1] if stated else 0
                tries_n = [value_list, value_list[:drop] + value_list[drop + 1:]] if len(value_list) > 1 else [[value_list[0], goods.name], value_list] if lead else [value_list]
                for tokens in tries_n[:2]:
                    if tokens:
                        issues: list[str] = []
                        call_args = _shape({**base, fld: ' '.join(tokens)}, self.schemas.get(tool_id), 'args', issues)
                        if issues or self.finished:
                            break
                        answer_obj = self.act(tool_id, call_args)
                        budget -= 1
                        if answer_obj is None:
                            break
                        observation = answer_obj.get('observation')
                        self.observe(tool_id, observation)
                        shelf_field = self.category_key()
                        hit_found = [i for (i, _) in self.records_in(observation)]
                        limit_val = self.budget if isinstance(self.budget, (int, float)) else None
                        costs = [self.price_of(i) for i in hit_found]
                        if limit_val is not None and costs and all((p is not None and p > 10 * limit_val for p in costs)):
                            limit_val = None
                        rows = [i for i in hit_found if self.buyable(i) and (not shelf_field or self.owns(goods, str((self.rows.get(i) or {}).get(shelf_field) or ''))) and (limit_val is None or (self.price_of(i) or 0) <= limit_val)]
                        home = self.home_label(goods_no)
                        if shelf_field and home:
                            rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(shelf_field) or '')) != home)
                        noticed.append(f'{_dump(call_args.get(fld))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                        if rows:
                            opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + VALUE_KEEP)
                            budget -= len(opened)
                            if opened:
                                noticed.append('opened: ' + ' | '.join(opened)[:2400])
                            break
            return self.value_search_continued(noticed)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'value_search', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def value_search_continued(self, noticed) -> str:
        """The rest of value_search, from the point where its earlier checks are done."""
        if not noticed:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in noticed]})
        return "[Tern searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(noticed)

    def home_label(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        shelf_field = self.category_key()
        return '' if not shelf_field else self.home_label_continued(number, shelf_field)

    def home_label_continued(self, number, shelf_field) -> str:
        """The rest of home_label, from the point where its earlier checks are done."""
        goods = self.items[number - 1]
        for shortlist in (self.full, self.rows):
            cnts: dict[str, int] = {}
            for rec_obj in shortlist.values():
                label = str(rec_obj.get(shelf_field) or '')
                if label and self.owns(goods, label):
                    cnts[_words(label)] = cnts.get(_words(label), 0) + 1
            for nm in (goods.shelf, goods.name):
                if _words(nm) in cnts:
                    return _words(nm)
            if cnts:
                return max(cnts, key=lambda label: cnts[label])
        return ''

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        recognized = self.owners_seen.get((label, len(self.items)))
        return recognized if recognized is not None else self.owner_of_continued(label)

    def owner_of_continued(self, label) -> int:
        """The rest of owner_of, from the point where its earlier checks are done."""
        tokens = _words(label)
        long = _roots(label)
        (winner, top) = (0, 0.0)
        for (goods_no, goods) in enumerate(self.items, 1):
            if tokens == _words(goods.shelf or ''):
                mark_of = 3.0
            elif tokens == _words(goods.name):
                mark_of = 2.0
            else:
                mine = _roots(goods.shelf or '') | _roots(goods.name)
                mark_of = len(long & mine) / max(1, len(long | mine))
            if mark_of > top:
                (winner, top) = (goods_no, mark_of)
        return self.owner_of_continued_continued(label, winner)

    def owner_of_continued_continued(self, label, winner) -> int:
        """The rest of owner_of_continued, from the point where its earlier checks are done."""
        self.owners_seen[label, len(self.items)] = winner
        return winner

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1 or _words(label) == _words(item.shelf or ''))

    def records_of_item(self, number: int) -> int:
        shelf_field = self.category_key()
        goods = self.items[number - 1]
        return sum((1 for rec_obj in self.full.values() if shelf_field and self.owns(goods, str(rec_obj.get(shelf_field) or '')) and _fact_block(rec_obj)))

    def adopt_labels(self) -> None:
        shelf_field = self.category_key()
        if len(self.items) < 2 or not shelf_field:
            return
        self.part_shelves()
        return self.adopt_labels_continued(shelf_field)

    def adopt_labels_continued(self, shelf_field) -> None:
        """The rest of adopt_labels, from the point where its earlier checks are done."""
        prop_labels = {str(rec_obj.get(shelf_field) or '') for rec_obj in self.full.values()} - {''}
        strays = [prop_label for prop_label in prop_labels if not self.owner_of(prop_label)]
        orphans = [idx for idx in range(1, len(self.items) + 1) if not any((self.owner_of(prop_label) == idx and prop_label not in self.adopted for prop_label in prop_labels))]
        if not (strays and len(orphans) == 1):
            return
        self.adopted.update({prop_label: orphans[0] for prop_label in strays})
        self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        not_names = self.not_maker_keys()
        lowered = about.lower()
        acc = []
        for pid in ids:
            one_row = self.rows.get(pid) or {}
            names = [str(item_val).strip() for (key_name, item_val) in one_row.items() if isinstance(item_val, str) and 2 < len(item_val.strip()) <= 40 and (key_name not in not_names) and (item_val.strip().lower() not in NO_NAME)]
            if pid not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(idx.lower()) + '(?![a-z0-9])', lowered) for idx in names)):
                acc.append(pid)
        return acc

    def opened_shop(self, about: str) -> list[str]:
        not_names = self.not_maker_keys()
        lowered = about.lower()
        acc = []
        for pid in self.opened:
            one_row = self.rows.get(pid) or {}
            names = [str(item_val).strip() for (key_name, item_val) in one_row.items() if isinstance(item_val, str) and 2 < len(item_val.strip()) <= 40 and (key_name not in not_names) and (item_val.strip().lower() not in NO_NAME)]
            if any((re.search('(?<![a-z0-9])' + re.escape(idx.lower()) + '(?![a-z0-9])', lowered) for idx in names)):
                acc.append(pid)
        return acc

    def find_link(self, about: str, words: str) -> list[str]:
        try:
            not_names = self.not_maker_keys()
            recognized: list[str] = []
            cited = {tok for pid in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for (key_name, item_val) in (self.rows.get(pid) or {}).items() if isinstance(item_val, str) and len(item_val.strip()) <= 40 and (key_name not in not_names) and (item_val.strip().lower() not in NO_NAME) for tok in re.findall('[a-z0-9]{3,}', item_val.lower())}
            value_list = {tok for tok in re.findall('[a-z0-9]+', (words or about).lower()) if len(tok) >= 3 or tok.isdigit()} - cited - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
            for pid in self.opened_shop(about):
                (stack, leaves) = ([self.full.get(pid) or self.rows.get(pid) or {}], [])
                while stack:
                    node = stack.pop()
                    if isinstance(node, (dict, list)):
                        stack += list(node.values()) if isinstance(node, dict) else node
                    else:
                        leaves.append(str(node))
                phrase = ' '.join(leaves).lower()
                if value_list and all((re.search('(?<![a-z0-9])' + re.escape(item_val) + '(?![a-z0-9])', phrase) for item_val in value_list)):
                    self.trace.append({'kind': 'link_known', 'id': pid})
                    return []
            return self.find_link_continued(about, cited, recognized, words)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'find_link', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def find_link_continued(self, about, cited, recognized, words) -> list[str]:
        """The rest of find_link, from the point where its earlier checks are done."""
        plan = self.value_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            recognized = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            (tool_id, base, fld) = plan
            for q_text in dict.fromkeys((words, ' '.join(words.split()[:2]))):
                issues: list[str] = []
                call_args = _shape({**base, fld: q_text}, self.schemas.get(tool_id), 'args', issues)
                if issues:
                    break
                self.link_searches += 1
                answer_obj = self.act(tool_id, call_args)
                if answer_obj is None:
                    break
                observation = answer_obj.get('observation')
                self.observe(tool_id, observation)
                self.link_seen[memo] = [cur for (cur, _) in self.records_in(observation)]
                recognized = self.shop_rows(about, self.link_seen[memo])
                self.trace.append({'kind': 'link_search', 'words': q_text[:80], 'rows': recognized[:4]})
                if self.link_seen[memo]:
                    break
        return self.find_link_continued_continued(about, cited, recognized, words)

    def find_link_continued_continued(self, about, cited, recognized, words) -> list[str]:
        """The rest of find_link_continued, from the point where its earlier checks are done."""
        sort_kind = {tok for tok in re.findall('[a-z0-9]{3,}', (words or about).lower())} - cited - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [cur for cur in ids if not sort_kind or sort_kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(cur) or {}).lower()))]
        return (of_kind(recognized) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        try:
            tool_id = self.offer_tool()
            goal = self.single_target(tool_id)
            if not tool_id or not goal:
                return ''
            since = max(self.step - 2, self.refused_at)
            stale = [cur for cur in self.cart_ids() if self.offer_seen.get(cur, -99) < since and cur not in self.gone]
            return self.reread_offers_continued(goal, stale, tool_id)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'reread_offers', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def reread_offers_continued(self, goal, stale, tool_id) -> str:
        """The rest of reread_offers, from the point where its earlier checks are done."""
        if not stale or self.left() < len(stale) + 5:
            return ''
        rowset = []
        for pid in stale:
            issues: list[str] = []
            call_args = _shape({goal: pid}, self.schemas.get(tool_id), 'args', issues)
            if issues or self.rushed():
                break
            before = dict(self.rows.get(pid) or {})
            answer_obj = self.act(tool_id, call_args)
            if answer_obj is None or self.finished:
                break
            observation = answer_obj.get('observation')
            self.offer_seen[pid] = self.step
            self.observe(tool_id, observation)
            after = self.rows.get(pid) or {}
            relocated = [key_name for (key_name, item_val) in after.items() if key_name in before and before[key_name] != item_val and (not isinstance(item_val, (dict, list)))]
            if relocated:
                rowset.append(f"{pid} changed ({', '.join(relocated[:4])}): {_dump(_clip(observation, 300))[:900]}")
        return self.reread_offers_continued_continued(rowset, stale)

    def reread_offers_continued_continued(self, rowset, stale) -> str:
        """The rest of reread_offers_continued, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(rowset)})
        if rowset:
            self.version += 1
        return '\n'.join(rowset)

    def open_rows(self, rows: list[str]) -> bool:
        (many, joiner) = (self.compare_target(), self.roles.get('compare'))
        tool_id = self.facts_tool()
        goal = self.single_target(tool_id)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool_id and goal:
            plan = [(tool_id, {goal: cur}) for cur in rows[:2]]
        else:
            return False
        cart_steps = len(self.cart_calls({f'#{idx}': 1 for (idx, _) in enumerate(self.items)}, []) or self.items)
        if self.left() < len(plan) + max(5, self.tail() + cart_steps):
            return False
        before = len(self.full)
        for (nm, call_args) in plan:
            issues: list[str] = []
            call_args = _shape(call_args, self.schemas.get(nm), 'args', issues)
            if issues:
                return False
            answer_obj = self.act(nm, call_args)
            if answer_obj is None or self.finished:
                break
            self.observe(nm, answer_obj.get('observation'))
        return self.open_rows_continued(before, rows)

    def open_rows_continued(self, before, rows) -> bool:
        """The rest of open_rows, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def compare_target(self) -> str | None:
        shape_spec = self.schemas.get(self.roles.get('compare') or '') or {}
        prop_map = shape_spec.get('properties') or {}
        goal = next((key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'array' in _kinds(item_val)), None)
        return None if not goal or any((key_name != goal for key_name in shape_spec.get('required') or [])) else goal

    def compare_room(self) -> int:
        slot_spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(slot_spec.get('maxItems'), int):
            return slot_spec['maxItems']
        return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2))) if self.compare_bad else max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            (opener, joiner, goal) = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
            many = self.compare_target()
            if not opener or not joiner or (not goal) or (not many) or (opener == joiner):
                return calls
            if self.compare_bare():
                return calls
            selected: list[tuple[int, str]] = []
            for (index, invocation) in enumerate(calls):
                fn = invocation.get('function') or {}
                call_args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
                pid = str((call_args or {}).get(goal) or '').strip()
                if fn.get('name') == opener and call_args and (set(call_args) == {goal}) and (pid in self.rows) and (pid not in self.gone):
                    selected.append((index, pid))
            selected = selected[:self.compare_room()]
            if len(selected) < 2:
                return calls
            first = selected[0][0]
            acc = []
            for (index, invocation) in enumerate(calls):
                if index == first:
                    acc.append({'id': invocation['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: [cur for (_, cur) in selected]})}})
                elif index in {cur for (cur, _) in selected}:
                    acc.append({'id': invocation['id'], 'joined': True})
                else:
                    acc.append(invocation)
            return self.join_opens_continued(acc, selected)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'join_opens', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def join_opens_continued(self, acc, selected) -> list[dict[str, Any]]:
        """The rest of join_opens, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'joined', 'ids': [cur for (_, cur) in selected]})
        return acc

    def record_of(self, ident: str) -> str:
        rec_obj = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(rec_obj, 400))[:3000] if rec_obj else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        rowset = []
        order = list(reversed(self.seen_order))
        return self.others_continued(limit, order, rowset, skip)

    def others_continued(self, limit, order, rowset, skip) -> str:
        """The rest of others, from the point where its earlier checks are done."""
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for pid in order:
            if not (pid in skip or self.sold_out(pid)):
                rec_obj = self.full.get(pid) or self.rows.get(pid)
                rowset.append(_dump(_clip(rec_obj, 160))[:900])
                if len(rowset) >= limit:
                    break
        return '\n'.join(rowset)

    def page_read(self, ident: str) -> bool:
        """Whether a listing's page was read: opened, and through the store's page view when it has one apart from the attribute view."""
        page = self.roles.get('offer')
        return ident in self.opened and (not page or not self.single_target(page) or page in self.views.get(ident, ()))

    def not_maker_keys(self) -> tuple[Any, ...]:
        """The row keys that cannot hold the name of who sells or makes a listing: its id, its label and its name. Any other
        text field can, and the value must still appear word for word in the description being matched."""
        return (self.id_key, self.category_key(), self.title_key())

class Hunter:
    """Searching on for fits and cheaper listings the first searches did not show."""

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        """The item's noun, a word its store label holds that no other item's label holds (a search reads labels too, so it
        keeps the rows to this item), with the words a further search can add, most promising first: a word on most rows finds
        nothing new and a word on one row finds little else, so the words kept sit between."""
        shelf_field = self.category_key()
        goods = self.items[number - 1]
        title = self.title_key()
        prop_labels = [str(one_row.get(shelf_field) or '') for one_row in self.rows.values() if self.owns(goods, str(one_row.get(shelf_field) or ''))] if shelf_field else []
        names = [str(one_row.get(title) or '') for one_row in self.rows.values() if title and shelf_field and self.owns(goods, str(one_row.get(shelf_field) or ''))]
        for rec_obj in self.full.values():
            if shelf_field and self.owns(goods, str(rec_obj.get(shelf_field) or '')):
                names.append(' '.join((str(item_val) for item_val in (_fact_block(rec_obj) or {}).values() if isinstance(item_val, str) and len(item_val) <= HUNT_VALUE_CHARS)))
        others = {w for (idx, second) in enumerate(self.items, 1) if idx != number for w in re.findall('[a-z]{4,}', ((second.shelf or '') + ' ' + second.name).lower())}
        prop_label = max(prop_labels, key=prop_labels.count).lower() if prop_labels else (goods.shelf or goods.name).lower()
        nouns = [w for w in re.findall('[a-z]{4,}', prop_label) if w not in others]
        return self.hunt_words_continued(goods, names, nouns, number)

    def hunt_words_continued(self, goods, names, nouns, number) -> tuple[str, list[str]]:
        """The rest of hunt_words, from the point where its earlier checks are done."""
        noun = '' if number in self.hunt_no_noun else max(nouns, key=len) if nouns else ''
        if not names:
            return (noun, [])
        cnts: dict[str, int] = {}
        for nm in names:
            for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", nm)}:
                cnts[word] = cnts.get(word, 0) + 1
        return self.hunt_words_continued_continued(cnts, goods, names, noun)

    def hunt_words_continued_continued(self, cnts, goods, names, noun) -> tuple[str, list[str]]:
        """The rest of hunt_words_continued, from the point where its earlier checks are done."""
        own = _roots(goods.shelf or '') | _roots(goods.name)
        ceiling = max(2, int(len(names) * HUNT_COMMON))
        tokens = [w for (w, idx) in cnts.items() if 2 <= idx <= ceiling and w != noun and _roots(w) and (not _roots(w) & own)]
        tokens.sort(key=lambda w: (-cnts[w], w))
        return (noun, tokens)

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
        try:
            mark = _dump([self.needs_now(), self.budget, sorted(self.gone)])
            if self.brief.get('pick') != 'cheapest' or self.hunts >= HUNT_ROUNDS or self.hunt_mark == mark:
                return ''
            plan = self.value_template()
            if not plan or not self.price_key or self.clock() > HUNT_BY:
                return ''
            (tool_id, base, fld) = plan
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
            for goods_no in numbers:
                if spent >= room or self.finished:
                    break
                (used, gained) = self.hunt_item(goods_no, tool_id, base, fld, min(share, room - spent))
                spent += used
                fresh_total += gained
            self.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh_total, 'cart': cart})
            if spent:
                self.wider_held = True
            return f'Tern ran {spent} more searches for the lowest price, one step each, and they showed {fresh_total} more rows.' if spent else ''
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'hunt_cheaper', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Up to `room` searches for one item: all its required values together, then split by further words."""
        try:
            tried = self.hunt_tried.setdefault(number, set())
            (noun, _) = self.hunt_words(number)
            value_list = [raw_val for (raw_val, _) in self.value_words(number)]
            spent = gained = idle = 0
            issued = refused = False

            def run(query: str) -> bool:
                """Search `query` to its last page or the room, reading again no page this episode has read; whether it found any row."""
                nonlocal spent, gained, idle, issued, refused
                tried.add(query.lower())
                issues: list[str] = []
                call_args = _shape({**base, field: query}, self.schemas.get(tool), 'args', issues)
                refused = refused or bool(issues)
                if issues or spent >= room or self.finished:
                    return False
                issued = True
                holds_values = bool(value_list) and all((item_val.lower() in query.lower() for item_val in value_list))
                hit_found = False
                walked: set[str] = set()
                while not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
                    mark = self.call_mark(tool, call_args)
                    if mark in walked:
                        return hit_found
                    walked.add(mark)
                    if mark in self.read_calls:
                        noticed = self.read_calls[mark]
                        fresh_needs = HUNT_PAGE_NEW
                    elif spent >= room:
                        return hit_found
                    else:
                        before = self.rows_of_item(number)
                        answer_obj = self.act(tool, call_args)
                        spent += 1
                        if answer_obj is None:
                            return hit_found
                        noticed = answer_obj.get('observation')
                        self.observe(tool, noticed)
                        fresh_needs = self.rows_of_item(number) - before
                        gained += fresh_needs
                        idle = 0 if fresh_needs else idle + 1
                    rows = self.records_in(noticed)
                    hit_found = hit_found or bool(rows)
                    if holds_values:
                        self.value_rows.update((cur for (cur, _) in rows))
                    page = self.paging(tool, noticed)
                    if not page or page[3] is None or (fresh_needs < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE):
                        return hit_found
                    issues = []
                    call_args = _shape({**call_args, page[2]: page[3]}, self.schemas.get(tool), 'args', issues)
                    if issues:
                        return hit_found
                return hit_found
            mark = (number, tuple(value_list))
            stem = self.hunt_stem.get(mark)
            if stem is None:
                tries_n = ([' '.join(value_list + [noun]).strip(), ' '.join(value_list)] if value_list else []) + ([' '.join(value_list[:-1] + [noun]).strip()] if len(value_list) >= 2 else []) + ([noun] if noun else [])
                stem = ''
                for query in dict.fromkeys(tries_n):
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
            while spent < room and idle < HUNT_IDLE and (not refused) and (not self.finished) and (self.clock() <= HUNT_BY + HUNT_GRACE):
                (_, tokens) = self.hunt_words(number)
                tokens = [tok for tok in tokens if f'{tok} {stem}'.strip().lower() not in tried and tok not in stem.lower().split()]
                if not tokens:
                    break
                run(f'{tokens[0]} {stem}'.strip())
            return (spent, gained)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'hunt_item', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    @_soft(list)
    def hunt_fits(self) -> list[str]:
        """Search on, without the model, before ending without a purchase.

        Race 15: 495 of 2340 episodes of 13 leaders ended without a purchase on a task that has a right answer, and on 433 of them
        another agent held a cart the store's answer set accepted, so a fit existed within reach; the same holds in practice worlds
        (a fit past the page cap, "no in-stock jeans list a slim fit"). A requirement's value and its words find listings the
        first searches did not show; the rows naming them are opened so that the audit judges their attributes. Returns what
        was read."""
        try:
            if self.fit_hunted or self.clock() > HUNT_BY:
                return []
            plan = self.value_template()
            if not plan:
                return []
            (tool_id, base, fld) = plan
            reserve_n = self.tail() + len(self.items) + 1
            spare = self.left() - reserve_n
            if spare < 3:
                return []
            self.fit_hunted = True
            shelf_field = self.category_key()
            held = [str((self.rows.get(i) or {}).get(shelf_field) or '') for i in self.cart_ids() if (self.held_units(i) or 0) > 0] if shelf_field else []
            numbers = [idx for (idx, goods) in enumerate(self.items, 1) if not any((self.owns(goods, prop_label) for prop_label in held))] or list(range(1, len(self.items) + 1))
            sought = {goods_no: {tok for (raw_val, _) in self.value_words(goods_no) for tok in _value_tokens(raw_val)} for goods_no in numbers}
            paged = self.unread_pages(tool_id, sought, min(FIT_PAGES, spare // 2))
            room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP)
            spent = 0
            for goods_no in numbers if room >= 3 else []:
                goods = self.items[goods_no - 1]
                pairs = self.value_words(goods_no)
                (noun, tokens) = self.hunt_words(goods_no)
                tries_n = [f'{family} {raw_val} {noun}'.strip() for family in self.series_for(goods_no) for raw_val in [item_val for (item_val, _) in pairs][:1] or ['']]
                tries_n += [f'{raw_val} {noun}'.strip() for (raw_val, _) in pairs] + [raw_val for (raw_val, _) in pairs if noun] + [f'{word} {noun}'.strip() for word in tokens[:HUNT_FIT_WORDS]] + ([noun] if noun else [])
                share = max(2, room // max(1, len(numbers)))
                used = 0
                for q_text in dict.fromkeys(tries_n):
                    if spent >= room or used >= share or self.finished or (self.clock() > HUNT_BY + HUNT_GRACE):
                        break
                    issues: list[str] = []
                    call_args = _shape({**base, fld: q_text}, self.schemas.get(tool_id), 'args', issues)
                    if issues:
                        continue
                    answer_obj = self.act(tool_id, call_args)
                    spent += 1
                    used += 1
                    if answer_obj is None:
                        return []
                    self.observe(tool_id, answer_obj.get('observation'))
            selected: list[str] = []
            for goods_no in numbers:
                goods = self.items[goods_no - 1]
                shortlist = [i for i in self.seen_order if i not in self.full and self.buyable(i) and (not shelf_field or self.owns(goods, str((self.rows.get(i) or {}).get(shelf_field) or '')))]
                limit_val = self.budget if isinstance(self.budget, (int, float)) else None
                shortlist = [i for i in shortlist if limit_val is None or (self.price_of(i) or 0) <= limit_val]
                tokens = sought.get(goods_no) or set()
                shortlist.sort(key=lambda i: (-len(tokens & _text_words(_dump(self.rows.get(i) or {}))), self.price_of(i) if self.price_of(i) is not None else float('inf')))
                selected += shortlist[:HUNT_FIT_OPENS]
            readings: list[str] = []
            for start in range(0, len(selected), HUNT_FIT_OPENS):
                if self.clock() > HUNT_BY + HUNT_GRACE:
                    break
                readings += self.fill_facts(selected[start:start + HUNT_FIT_OPENS], reserve_n)
            self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(readings)})
            return readings
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'hunt_fits', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Up to `room` further pages of the earliest searches that hold a required value, each item its share first, before any
        new words; the steps it took. A search whose rows all belong to another item is left. In a practice world accepted rows
        sat on unread pages of the episode's own searches in nearly every wrong decline, and in races 13-15 about 23% of passing
        carts were first seen on page 2 or later."""
        spent = 0
        shelf_field = self.category_key()
        return self.unread_pages_continued(room, shelf_field, spent, tool, wanted)

    def unread_pages_continued(self, room, shelf_field, spent, tool, wanted) -> int:
        """The rest of unread_pages, from the point where its earlier checks are done."""
        share = max(1, room // max(1, len(wanted)))
        for limit in (share, room):
            for (goods_no, tokens) in wanted.items():
                stop = min(room, spent + limit)
                for (mark, (nm, call_args)) in list(self.read_args.items()):
                    phrase = ' '.join((item_val for item_val in call_args.values() if isinstance(item_val, str))).lower()
                    if nm != tool or not tokens & _text_words(phrase):
                        continue
                    noticed = self.read_calls[mark]
                    owners = {self.owner_of(str(rec_obj.get(shelf_field) or '')) for (_, rec_obj) in self.records_in(noticed)} if shelf_field else set()
                    if owners and 0 not in owners and (goods_no not in owners):
                        continue
                    (invocation, walked) = (call_args, {mark})
                    while spent < stop and (not self.finished):
                        page = self.paging(tool, noticed)
                        if not page or page[3] is None:
                            break
                        issues: list[str] = []
                        invocation = _shape({**invocation, page[2]: page[3]}, self.schemas.get(tool), 'args', issues)
                        now = self.call_mark(tool, invocation) if not issues else ''
                        if issues or now in walked:
                            break
                        walked.add(now)
                        if now in self.read_calls:
                            noticed = self.read_calls[now]
                            continue
                        answer_obj = self.act(tool, invocation)
                        spent += 1
                        if answer_obj is None:
                            return spent
                        noticed = answer_obj.get('observation')
                        self.observe(tool, noticed)
        return spent

    def refuse_trail(self) -> str:
        """Once, before a decline with room left: follow what a requirement points at; what that showed, '' when nothing."""
        alive = {req for (_, req) in self.needs_now()}
        if self.refuse_trailed or self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or (not any((req in alive and req not in self.ref_values for (_, req) in self.refs))):
            return ''
        self.refuse_trailed = True
        (phrase, _) = self.trail(HUNT_BY)
        return phrase + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if phrase else ''

    def early_trail(self) -> str:
        """Follow the chains of requirements that point at another listing, once, while there is room to act on what they say."""
        alive = {req for (_, req) in self.needs_now()}
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or (not any((req in alive and req not in self.ref_values for (_, req) in self.refs))):
            return ''
        kept = (self.trails, self.trail_blocks)
        (self.trails, self.trail_room) = (0, EARLY_TRAILS)
        try:
            (phrase, _) = self.trail(HUNT_BY)
        finally:
            ((self.trails, self.trail_blocks), self.trail_room) = (kept, None)
        self.trace.append({'kind': 'early_trail', 'left': self.left(), 'values': list(self.ref_values.values())[:3]})
        return phrase

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        try:
            alive = {req for (_, req) in self.needs_now()}
            pending = [req for (_, req) in self.refs if req in alive and req not in self.ref_values]
            tool_id = self.roles.get('offer') or self.roles.get('open')
            goal = self.single_target(tool_id)
            noticed: list[str] = []
            dead_ids: list[str] = []
            while pending and self.trails < min(6 if self.left() > 15 else 4, self.trail_room or 6) and (self.cost < SPEND_HARD) and (stop is None or self.clock() <= stop):
                self.trails += 1
                records = []
                for pid in reversed(self.seen_order):
                    rec_obj = self.full.get(pid) or self.rows.get(pid) if self.page_read(pid) else None
                    if rec_obj and len(records) < 45:
                        records.append('OPENED ' + _dump(_clip(rec_obj, 300))[:1400])
                for pid in reversed(self.seen_order):
                    if len(records) >= 60:
                        break
                    if not self.page_read(pid) or not (self.full.get(pid) or self.rows.get(pid)):
                        records.append('ROW ' + (_dump(_clip(self.full[pid], 120))[:700] if pid in self.full else _dump(_clip(self.rows.get(pid), 80))[:300]))
                instr = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
                instr += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
                got = self.ask_json(instr, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
                self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
                reach: list[str] = []
                dead_ids = []
                gaps: list[tuple[str, str]] = []
                for chain in _items(got.get('chains')):
                    if isinstance(chain, dict):
                        req = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                        if not req:
                            continue
                        links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                        pids = [str(l.get('id') or '').strip() for l in links]
                        if chain.get('done') is True and links and all((self.page_read(cur) for cur in pids)) and str(chain.get('value') or '').strip():
                            self.ref_values[req] = str(chain['value']).strip()
                            self.version += 1
                            continue
                        if links and all((self.page_read(cur) for cur in pids)) and (not str(links[-1].get('find') or '').strip()):
                            self.ref_values[req] = str(chain.get('value') or '').strip() or f'as stated on {pids[-1]}'
                            self.version += 1
                            continue
                        closed = [cur for cur in pids if cur in self.rows and (not self.page_read(cur))]
                        if closed:
                            reach += [cur for cur in closed if cur not in reach]
                        else:
                            gap = next((l for (l, cur) in zip(links, pids) if cur not in self.rows), {})
                            about = str(gap.get('about') or '') or 'the listing it points at'
                            tokens = str(gap.get('find') or '').strip()
                            dead_ids.append(f'{req} -> find {about}' + (f' (search: {tokens})' if tokens else ''))
                            gaps.append((about, tokens))
                pending = [p for p in pending if p not in self.ref_values]
                if not reach and gaps and tool_id and goal and (self.left() > 12) and (self.link_searches < 3):
                    reach = self.find_link(*gaps[0])
                    if self.trace and self.trace[-1].get('kind') == 'link_known':
                        noticed.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                        dead_ids = dead_ids[1:]
                if not reach or not tool_id or (not goal) or (self.left() < len(reach) + 4) or self.finished:
                    break
                for pid in reach[:max(0, min(3, 6 - len(noticed)))]:
                    issues: list[str] = []
                    call_args = _shape({goal: pid}, self.schemas.get(tool_id), 'args', issues)
                    if issues or self.rushed():
                        break
                    answer_obj = self.act(tool_id, call_args)
                    if answer_obj is None or self.finished:
                        break
                    self.observe(tool_id, answer_obj.get('observation'))
                    noticed.append(f'{pid}: ' + self.show(answer_obj.get('observation'))[:1500])
            return self.trail_continued(dead_ids, noticed)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'trail', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def trail_continued(self, dead_ids, noticed) -> tuple[str, bool]:
        """The rest of trail, from the point where its earlier checks are done."""
        phrase = ''
        if noticed:
            phrase += '[Tern opened the pages a requirement points at]\n' + '\n'.join(noticed) + '\n'
        if self.ref_values:
            phrase += '[values read from the pages requirements point at] ' + '; '.join((f'{key_name} -> {item_val}' for (key_name, item_val) in self.ref_values.items())) + '\n'
        return self.trail_continued_continued(dead_ids, phrase)

    def trail_continued_continued(self, dead_ids, phrase) -> tuple[str, bool]:
        """The rest of trail_continued, from the point where its earlier checks are done."""
        hold = bool(dead_ids) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            phrase += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(dead_ids[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (phrase, hold)

class Interviewer:
    """Planning the questions to the shopper and reading the replies."""

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        return False if self.left() <= 8 or self.clock() > ASK_BY else self.ready_to_ask_continued()

    def ready_to_ask_continued(self) -> bool:
        """The rest of ready_to_ask, from the point where its earlier checks are done."""
        opened = len(self.full)
        if self.loop_calls < ASK_TURNS and (not all((self.records_of_item(idx) for idx in range(1, len(self.items) + 1)))):
            return False
        return True if opened >= 3 * len(self.items) + 1 else self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        nm = self.roles.get('ask')
        shape_spec = self.schemas.get(nm or '') or {}
        prop_map = shape_spec.get('properties') or {}
        required = shape_spec.get('required') or []
        text_fields = [k for (k, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val) and ('const' not in item_val)]
        shortlist = [k for k in required if k in text_fields] or text_fields
        return self.message_args_continued(prop_map, required, shape_spec, shortlist, text)

    def message_args_continued(self, prop_map, required, shape_spec, shortlist, text) -> dict[str, Any] | None:
        """The rest of message_args, from the point where its earlier checks are done."""
        call_args: dict[str, Any] = {}
        if shortlist:
            call_args[max(shortlist, key=lambda k: (not re.search(NOT_MESSAGE, _snake(k), re.I), bool(re.search(MESSAGE_FIELD, _snake(k), re.I)), prop_map[k].get('maxLength') or 10 ** 6))] = text
        else:
            for (fkey, slot_spec) in prop_map.items():
                inner = next((k for (k, item_val) in (slot_spec.get('properties') or {}).items() if isinstance(item_val, dict) and 'string' in _kinds(item_val)), None) if isinstance(slot_spec, dict) else None
                if isinstance(slot_spec, dict) and 'array' in _kinds(slot_spec):
                    call_args[fkey] = [text]
                    break
                if inner:
                    call_args[fkey] = {inner: text}
                    break
        if not call_args:
            return None
        for fkey in required:
            if fkey not in call_args:
                call_args[fkey] = _filler(prop_map.get(fkey) or {})
                if call_args[fkey] is None:
                    return None
        return self.message_args_continued_continued(call_args, shape_spec)

    def message_args_continued_continued(self, call_args, shape_spec) -> dict[str, Any] | None:
        """The rest of message_args_continued, from the point where its earlier checks are done."""
        issues: list[str] = []
        call_args = _shape(call_args, shape_spec, 'args', issues)
        return None if issues else call_args

    def interview(self) -> str:
        try:
            self.asked = True
            if self.msgs_free() <= 0:
                return ''
            self.widen()
            cnt = self.msg_limit - self.msg_used
            self.lanes = self.plan_lanes(max(cnt, self.list_cap()) if self.listing else cnt)
            self.lane_found = {idx: 0 for idx in self.lanes}
            self.lane_spent = {idx: 0 for idx in self.lanes}
            self.lane_used = {idx: set() for idx in self.lanes}
            self.follow_up = []
            posted: list[tuple[int, str, str]] = []
            told: list[str] = []
            batch_groups = self.groups_for(self.msgs_free()) if self.listing else []
            while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
                sent_batch = []
                while batch_groups and self.listing and (not sent_batch):
                    sent_batch = self.ask_group(batch_groups.pop(0), told)
                sent_batch = sent_batch or self.ask_next(told)
                if not sent_batch:
                    break
                posted += sent_batch
            return self.interview_continued(posted, told)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'interview', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def interview_continued(self, posted, told) -> str:
        """The rest of interview, from the point where its earlier checks are done."""
        found_items = [idx for idx in self.lane_found if self.lane_found[idx] > 0]
        chain = self.early_trail()
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        rowset = [f'Q: {q}\nA: {a}' for (_, q, a) in posted]
        recap = "Tern's interviewer asked the shopper (each question used one step):\n" + '\n'.join(rowset)
        recap += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        return self.interview_continued_continued(chain, recap, searched, told)

    def interview_continued_continued(self, chain, recap, searched, told) -> str:
        """The rest of interview_continued, from the point where its earlier checks are done."""
        recap += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [req for (_, req) in self.refs if any((req in t for t in told))]
        if pointed:
            recap += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if chain:
            recap += '\n' + chain
        if searched:
            recap += '\n' + searched
        if self.queued:
            recap += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return recap

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Tern is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        try:
            if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
                return ''
            if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
                return ''
            if self.left() < LATE_STEPS or self.clock() > LATE_BY or (not self.judges()):
                return ''
            self.late_asked = True
            if not self.lanes:
                self.lanes = self.plan_lanes(self.list_cap() if self.listing else 1)
                self.lane_found = {idx: 0 for idx in self.lanes}
                self.lane_spent = {idx: 0 for idx in self.lanes}
                self.lane_used = {idx: set() for idx in self.lanes}
            told: list[str] = []
            budget = self.budget
            sent_batch = self.ask_next(told, LATE_STEPS - 2)
            if not sent_batch:
                self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
                return ''
            if self.budget != budget:
                told.append(f'budget now {self.budget}')
            return self.late_question_continued(sent_batch, told)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'late_question', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def late_question_continued(self, sent_batch, told) -> str:
        """The rest of late_question, from the point where its earlier checks are done."""
        (goods_no, question, answer) = sent_batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        recap = f'Tern asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        return self.late_question_continued_continued(goods_no, recap, told)

    def late_question_continued_continued(self, goods_no, recap, told) -> str:
        """The rest of late_question_continued, from the point where its earlier checks are done."""
        pointed = [req for (_, req) in self.refs if any((req in t for t in told))]
        if pointed:
            recap += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([goods_no])
        return recap + ('\n' + searched if searched else '')

    def groups_for(self, free: int) -> list[list[int]]:
        """Which items each of the free messages asks about when the items outnumber them: consecutive shares as even as they
        can be, so no item goes unasked. Race 15: 19 of 90 tasks wanted 4 or 5 items and allowed 3 messages, and an agent naming
        two items in one message drew a need on either of them (10 needs in 37 such messages of one agent)."""
        numbers = [idx for idx in self.plan_order if idx in self.lanes] + [idx for idx in self.lanes if idx not in self.plan_order]
        numbers = [idx for idx in numbers if self.lanes.get(idx)]
        return [] if free <= 0 or len(numbers) <= free else self.groups_for_continued(free, numbers)

    def groups_for_continued(self, free, numbers) -> list[list[int]]:
        """The rest of groups_for, from the point where its earlier checks are done."""
        acc: list[list[int]] = []
        start = 0
        for slot in range(free):
            width_n = (len(numbers) - start + (free - slot) - 1) // (free - slot)
            acc.append(numbers[start:start + width_n])
            start += width_n
        return acc

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """One message naming a few planned properties of each item of the group, the same list rules applying: the reply is
        about the first tracked property named, so each item's best keys lead its part and the first item's first label is the
        only one a reply disclosing nothing settles."""
        try:
            share = max(1, self.list_now() // len(group)) if self.list_now() < self.list_cap() else max(MIX_LABELS, self.list_cap() // len(group))
            segments: list[tuple[int, list[str]]] = []
            for goods_no in group:
                left = [prop_label for prop_label in self.lanes.get(goods_no) or [] if _words(prop_label) not in self.lane_used.setdefault(goods_no, set()) and (not self.told_about(goods_no, prop_label))]
                if left:
                    segments.append((goods_no, left[:share]))
            if not segments:
                return []
            if len(segments) == 1:
                (goods_no, prop_labels) = segments[0]
                (fitted, phrase) = self.fit_ask(goods_no, prop_labels, False)
                if not fitted:
                    return []
                sent_batch = self.send([(goods_no, prop_labels[0], phrase)], floor)
                if sent_batch:
                    self.finish_ask(goods_no, fitted, False, sent_batch, told)
                return sent_batch
            phrase = ''
            while True:
                for width_n in range(share, 0, -1):
                    pieces = [f"{', '.join(prop_labels[:width_n][:-1])} or {prop_labels[:width_n][-1]}" if len(prop_labels[:width_n]) >= 2 else prop_labels[0] for (_, prop_labels) in segments]
                    phrase = ', and '.join((f'what {piece} for the {self.item_said(goods_no)}' for (piece, (goods_no, _)) in zip(pieces, segments)))
                    phrase = 'W' + phrase[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                    call_args = self.message_args(phrase)
                    if call_args and _says(call_args, phrase):
                        segments = [(goods_no, prop_labels[:width_n]) for (goods_no, prop_labels) in segments]
                        break
                else:
                    segments = segments[:-1]
                    if not segments:
                        return []
                    continue
                break
            first = segments[0][0]
            sent_batch = self.send([(first, segments[0][1][0], phrase)], floor)
            if not sent_batch:
                return []
            for (goods_no, _) in segments:
                self.lane_spent[goods_no] = self.lane_spent.get(goods_no, 0) + 1
            self.list_asks += 1
            had = [len(goods.told) for goods in self.items]
            readout = self.hear([(0, phrase, sent_batch[0][2])], (LIGHT, HEAVY))
            readout.pop('free', None)
            self.take_heard(readout, told)
            answered = False
            for (goods_no, prop_labels) in segments:
                fresh_needs = self.items[goods_no - 1].told[had[goods_no - 1]:]
                if fresh_needs:
                    answered = True
                    self.lane_found[goods_no] = self.lane_found.get(goods_no, 0) + 1
                    self.settle(goods_no, prop_labels, False, list(fresh_needs), False)
            return self.ask_group_continued(answered, first, readout, segments, sent_batch)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'ask_group', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def ask_group_continued(self, answered, first, readout, segments, sent_batch) -> list[tuple[int, str, str]]:
        """The rest of ask_group, from the point where its earlier checks are done."""
        crowded = any((_count(el) == 1 for el in _items(readout.get('crowded'))))
        if crowded and (not answered):
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': first})
            return sent_batch
        if not answered:
            self.lane_used.setdefault(first, set()).add(_words(segments[0][1][0]))
        return self.ask_group_continued_continued(answered, segments, sent_batch)

    def ask_group_continued_continued(self, answered, segments, sent_batch) -> list[tuple[int, str, str]]:
        """The rest of ask_group_continued, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'mixed_ask', 'parts': segments, 'answered': answered})
        return sent_batch

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Read a sent question's answer into the items' needs and settle the labels it named."""
        try:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
            self.list_asks += 1
            had = [len(goods.told) for goods in self.items]
            readout = self.hear(batch, (LIGHT, HEAVY))
            if len(labels) > 1:
                readout.pop('free', None)
            self.take_heard(readout, told)
            fresh_needs = [req for (goods, width_n) in zip(self.items, had) for req in goods.told[width_n:]]
            for (idx, (goods, width_n)) in enumerate(zip(self.items, had), 1):
                if len(goods.told) > width_n:
                    self.lane_found[idx] = self.lane_found.get(idx, 0) + 1
            return self.finish_ask_continued(fresh_needs, labels, number, readout, shown)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'finish_ask', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def finish_ask_continued(self, fresh_needs, labels, number, readout, shown) -> None:
        """The rest of finish_ask, from the point where its earlier checks are done."""
        crowded = any((_count(el) == 1 for el in _items(readout.get('crowded'))))
        self.settle(number, labels, shown, fresh_needs, crowded)

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def list_now(self) -> int:
        """Labels the next list names: few in the episode's first questions, since nothing shows yet how this shopper takes a list."""
        step_rows = LIST_STEPS if len(self.items) == 1 else LIST_STEPS_MIXED
        return min(step_rows[self.list_asks], self.list_cap()) if self.list_asks < len(step_rows) else self.list_cap()

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: cur for (cur, n) in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for goods_no in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [prop_label for prop_label in lanes[goods_no] if _words(prop_label) not in used[goods_no] and (not self.told_about(goods_no, prop_label))]
            if left:
                return (goods_no, left[:self.list_now() if self.listing else 1])
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
        noticed: dict[str, set[str]] = {}
        for pid in self.item_ids(number):
            for (fkey, raw_val) in (_fact_block(self.full.get(pid) or {}) or {}).items():
                noticed.setdefault(_words(fkey), set()).update((tok for tok in _roots(str(raw_val)) if len(tok) > 1))
        return self.look_alikes_continued(mine, noticed)

    def look_alikes_continued(self, mine, noticed) -> list[str]:
        """The rest of look_alikes, from the point where its earlier checks are done."""
        cnts: dict[str, int] = {}
        for nm in noticed:
            for word in set(nm.split()):
                cnts[word] = cnts.get(word, 0) + 1
        plain = {tok for tok in mine.split() if cnts.get(tok, 0) <= 2}
        scored = []
        for (order, (nm, value_list)) in enumerate(noticed.items()):
            if nm != mine and nm:
                shared = plain & set(nm.split())
                common = noticed.get(mine, set()) & value_list
                if shared or common:
                    scored.append((-(3 * len(shared) + min(len(common), 3)), order, nm))
        return [nm for (_, _, nm) in sorted(scored)[:LOOK_ALIKES]]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        alike = self.apart.get(labels[0], {}).get(number)
        return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {self.item_said(number)}?" if alike else self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for width_n in range(len(labels), 0, -1):
            phrase = self.ask_text(number, labels[:width_n], shown)
            call_args = self.message_args(phrase)
            if call_args and _says(call_args, phrase):
                return (labels[:width_n], phrase)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        try:
            for _ in range(LIST_SIZE):
                self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
                if self.follow_up:
                    (goods_no, prop_labels, with_examples) = self.follow_up.pop(0)
                else:
                    (goods_no, prop_labels) = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                    with_examples = False
                if not prop_labels:
                    return []
                (fitted, phrase) = self.fit_ask(goods_no, prop_labels, with_examples)
                if not fitted and with_examples:
                    (fitted, phrase) = self.fit_ask(goods_no, prop_labels, False)
                if fitted:
                    break
                self.lane_used.setdefault(goods_no, set()).add(_words(prop_labels[0]))
                self.trace.append({'kind': 'ask_too_long', 'item': goods_no, 'label': prop_labels[0]})
            else:
                return []
            prop_labels = fitted
            sent_batch = self.send([(goods_no, prop_labels[0], phrase)], floor)
            if not sent_batch:
                return []
            self.finish_ask(goods_no, prop_labels, with_examples, sent_batch, told)
            return sent_batch
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'ask_next', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

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
        try:
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
            if crowded and (not new):
                self.listing = False
                self.trace.append({'kind': 'lists_off', 'item': number})
                self.follow_up.insert(0, (number, labels[:1], False))
                return
            hit_at = self.answered_label(labels, new) if new else -1
            return self.settle_continued(hit_at, labels, number, used)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'settle', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def settle_continued(self, hit_at, labels, number, used) -> None:
        """The rest of settle, from the point where its earlier checks are done."""
        ahead = labels[hit_at + 1:] if hit_at >= 0 else labels[:0:-1]
        used.update((_words(prop_label) for prop_label in labels[:max(hit_at, 0) + 1]))
        return self.settle_continued_continued(ahead, hit_at, labels, number)

    def settle_continued_continued(self, ahead, hit_at, labels, number) -> None:
        """The rest of settle_continued, from the point where its earlier checks are done."""
        leftover = [prop_label for prop_label in self.lanes.get(number) or [] if prop_label not in labels]
        self.lane_misses[number] = 0 if hit_at >= 0 else self.lane_misses.get(number, 0) + 1
        if self.lane_misses[number] and self.lane_misses[number] % 2 == 0:
            ahead = [prop_label for (_, _, prop_label) in sorted(((abs(cur - (len(ahead) - 1) / 2), cur, prop_label) for (cur, prop_label) in enumerate(ahead)))]
            self.lanes[number] = leftover + ahead
        else:
            self.lanes[number] = ahead + leftover
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit_at + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        noticed: dict[str, set[str]] = {}
        for (_, fkey, _, value_list) in self.key_table():
            noticed.setdefault(_words(fkey), set()).update((tok for item_val in value_list for tok in _roots(item_val)))
        hit_list = []
        for req in new:
            (prop_word, _, raw_val) = req.partition(':')
            scores = [len(_spelled(prop_label) & _spelled(prop_word)) for prop_label in labels]
            if max(scores, default=0) == 0:
                scores = [len(noticed.get(_words(prop_label), set()) & _roots(raw_val)) for prop_label in labels]
            if max(scores, default=0) > 0:
                hit_list.append(scores.index(max(scores)))
        return min(hit_list) if hit_list else -1

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        goods = self.items[number - 1]
        settled = [_stem(mo.split(':', 1)[0]) for mo in goods.musts]
        rows = [(prop_label, fkey, carry_count, len(value_list)) for (prop_label, fkey, carry_count, value_list) in table if self.owner_of(prop_label) == number]
        measured = {fkey: sum((1 for item_val in value_list if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', item_val))) * 2 > len(value_list) for (_, fkey, _, value_list) in table}
        if len(self.items) == 1 and (not rows):
            rows = [(prop_label, fkey, carry_count, len(value_list)) for (prop_label, fkey, carry_count, value_list) in table]
        return self.lane_keys_continued(measured, rows, settled)

    def lane_keys_continued(self, measured, rows, settled) -> list[str]:
        """The rest of lane_keys, from the point where its earlier checks are done."""
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        every = [fkey for (_, fkey, _, _) in rows if not any((_names(prop_word, _stem(fkey)) for prop_word in settled))]
        fkeys = [fkey for fkey in every if not re.search(NOT_PROPERTY, fkey) or re.search('\\w[ _]models?\\b', fkey)]
        return [fkey for fkey in fkeys if not re.search(CATCH_ALL, fkey)] + [fkey for fkey in fkeys if re.search(CATCH_ALL, fkey)] + [fkey for fkey in every if fkey not in fkeys]

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        try:
            self.adopt_labels()
            table = self.key_table()
            settled = []
            for (goods_no, goods) in enumerate(self.items, 1):
                settled.append(f'item {goods_no}: {goods.name} - settled by the request: ' + ('; '.join(goods.musts) or 'nothing'))
            listing = []
            last = None
            for (index, (label, nm, carry_count, value_list)) in enumerate(table, 1):
                if label != last:
                    listing.append(f'category {label!r}:')
                    last = label
                listing.append(f'  {index}. {nm} (on {carry_count}): ' + ' | '.join(value_list) + self.not_stated(label, nm, value_list) + (' [several features in one: ask last]' if re.search(CATCH_ALL, nm) else ''))
            instr = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
            first = [str(el) for el in _items(self.brief.get('ask_first')) if str(el).strip()]
            if first:
                instr += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
            instr += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
            plan = self.ask_json(instr, [HEAVY, LIGHT], 3500) or {}
            lanes: dict[int, list[str]] = {idx: [] for idx in range(1, len(self.items) + 1)}
            hinted: dict[int, set[str]] = {idx: set() for idx in lanes}
            for selection in _items(plan.get('ask')):
                if isinstance(selection, dict):
                    goods_no = min(max(1, _count(selection.get('item') or 1) or 1), len(self.items))
                    label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(selection.get('label') or ''))[0].strip().rstrip('?.')
                    if not label or len(label.split()) > 5:
                        continue
                    if all((_words(label) != _words(second) for second in lanes[goods_no])):
                        lanes[goods_no].append(label)
                    if str(selection.get('hint') or '').strip():
                        hinted[goods_no].add(_words(label))
                        if goods_no not in self.plan_order:
                            self.plan_order.append(goods_no)
            for goods_no in lanes:
                fkeys = self.lane_keys(goods_no, table)
                readings = max(1, self.records_of_item(goods_no))
                own: dict[str, float] = {}
                for (label, fkey, carry_count, _) in table:
                    if self.owner_of(label) == goods_no:
                        own[fkey] = max(own.get(fkey, 0.0), carry_count / readings)

                def spelled(label: str) -> str:
                    plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                    return next((key_name for key_name in own if plain(key_name) == plain(label)), label)

                def share_of(label: str) -> float:
                    if _words(label) in own:
                        return own[_words(label)]
                    near = [s for (key_name, s) in own.items() if _names(_stem(label), _stem(key_name)) or _names(_stem(key_name), _stem(label))]
                    return max(near) if near else 0.0
                hinted[goods_no] = {_words(spelled(h)) for h in hinted[goods_no]}
                track = []
                for selection in lanes[goods_no]:
                    if all((_words(spelled(selection)) != _words(second) for second in track)):
                        track.append(spelled(selection))
                for fkey in [key_name for key_name in fkeys if own.get(key_name, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, key_name))][:GUARD_KEYS]:
                    if not any((_words(fkey) == _words(second) for second in track[:3])):
                        weak = next((cur for (cur, second) in enumerate(track[:3]) if _words(second) not in hinted[goods_no] and share_of(second) + GUARD_GAP <= own.get(fkey, 0.0)), None)
                        if weak is None and len(track) >= 3:
                            continue
                        placed = [second for second in track if _words(second) != _words(fkey)]
                        if weak is not None and len(self.items) > 1:
                            relocated = placed[weak]
                            placed[weak] = fkey
                            placed.insert(min(3, len(placed)), relocated)
                        else:
                            placed.insert(min(len(placed), weak if weak is not None else len(placed)), fkey)
                        if any((_words(el) in hinted[goods_no] and all((_words(el) != _words(y) for y in placed[:3])) for el in track[:3])):
                            continue
                        track = placed
                lanes[goods_no] = track
                for fkey in fkeys:
                    if all((_words(fkey) != _words(second) for second in lanes[goods_no])):
                        lanes[goods_no].append(fkey)
                lanes[goods_no] = [el for el in lanes[goods_no] if not re.search(CATCH_ALL, _words(el))] + [el for el in lanes[goods_no] if re.search(CATCH_ALL, _words(el))]
            return self.plan_lanes_continued(lanes, plan)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'plan_lanes', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def plan_lanes_continued(self, lanes, plan) -> dict[int, list[str]]:
        """The rest of plan_lanes, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {idx: item_val[:8] for (idx, item_val) in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        nm = self.items[number - 1].name
        if not re.search('\\band\\b', nm, re.I):
            return []
        segments = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', nm, flags=re.I) if p.strip()]
        return segments if len(segments) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in segments)) else []

    def item_said(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first)."""
        goods = self.items[number - 1]
        segments = self.name_parts(number)
        if not segments:
            return goods.name
        return re.sub('\\band\\b', 'or', goods.name, flags=re.I) if number not in self.one_product else self.item_said_continued(goods, segments)

    def item_said_continued(self, goods, segments) -> str:
        """The rest of item_said, from the point where its earlier checks are done."""
        shelf_field = self.category_key()
        rows = [one_row for one_row in self.rows.values() if not shelf_field or self.owns(goods, str(one_row.get(shelf_field) or ''))]
        cited = [_roots(' '.join((str(item_val) for (key_name, item_val) in one_row.items() if key_name != shelf_field and isinstance(item_val, str)))) for one_row in rows]
        hit_list = {p: sum((1 for tokens in cited if _roots(p.split()[-1]) & tokens)) for p in segments}
        return max(segments, key=lambda p: (hit_list[p], -segments.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        phrase = f'What {label} would you like for the {self.item_said(number)}'
        return phrase + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for (_, nm, _, value_list) in self.key_table():
            if nm == want:
                return [item_val for item_val in value_list if item_val and item_val not in settled and (len(item_val) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool_id = self.roles['ask']
        posted: list[tuple[int, str, str]] = []
        return self.send_continued(floor, posted, questions, tool_id)

    def send_continued(self, floor, posted, questions, tool_id) -> list[tuple[int, str, str]]:
        """The rest of send, from the point where its earlier checks are done."""
        for (goods_no, _, phrase) in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            call_args = self.message_args(phrase)
            if not call_args:
                break
            answer_obj = self.act(tool_id, call_args)
            if answer_obj is None:
                break
            self.msg_used += 1
            answer = answer_obj.get('observation')
            tokens = self.spoken(answer)
            posted.append((goods_no, phrase, tokens))
            self.done_log.append(f'step {self.step}: message to the shopper {phrase!r} -> reply {tokens[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': phrase, 'a': tokens[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return posted

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        try:
            many = len(self.items) > 1
            for ent in _items(heard.get('needs')):
                if not (not isinstance(ent, dict) or not str(ent.get('need') or '').strip()):
                    goods_no = min(max(1, _count(ent.get('item')) or 1), len(self.items))
                    req = str(ent['need']).strip()
                    if req not in self.items[goods_no - 1].told:
                        self.items[goods_no - 1].told.append(req)
                        told.append(f'item {goods_no}: {req}' if many else req)
                    if ent.get('points') is True and (goods_no, req) not in self.refs:
                        self.refs.append((goods_no, req))
            for ent in _items(heard.get('drop')):
                if isinstance(ent, dict) and str(ent.get('property') or '').strip():
                    goods = self.items[min(max(1, _count(ent.get('item')) or 1), len(self.items)) - 1]
                    gone = _words(str(ent['property']).split(':')[0])
                    hit_at = [mo for mo in goods.musts + goods.told if mo not in goods.dropped and _words(mo.split(':')[0]) == gone]
                    goods.dropped.extend(hit_at)
                    told.extend((f'dropped {mo}' for mo in hit_at))
            for ent in _items(heard.get('free')):
                if isinstance(ent, dict) and ent.get('property'):
                    goods_no = min(max(1, _count(ent.get('item')) or 1), len(self.items))
                    self.items[goods_no - 1].free.append(str(ent['property']))
            if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
                self.budget = heard['budget']
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'take_heard', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        try:
            if self.quiet_reply(sent):
                self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
                return {}
            self.hear_reads += 1
            instr = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {idx}: {it.name}' for (idx, it) in enumerate(self.items, 1)))
            instr += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{cur}. ({(f'about item {idx}' if idx else 'about several items')}) Q: {q}\nA: {a}" for (cur, (idx, q, a)) in enumerate(sent, 1)))
            return self.ask_json(instr, list(models), 2500) or {}
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'hear', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """Whether an answer needs no reading: it names nothing the questions, the items, the listings' keys or their values name
        and holds no number, and it asks nothing about how to ask. Race 15: 5112 of 5160 non-need replies are such (race 14: 5050 of 5055) and
        the check misses few need replies once the keys count (race 15: 35 of 1355 by question words alone). A reading costs a
        model call, and a race model call about 5-10 s of a 300 s episode. The first answer of an episode is always read: it
        tells whether this shopper takes lists."""
        if self.hear_reads < 1:
            return False
        recognized: frozenset[str] = frozenset()
        for (goods_no, goods) in enumerate(self.items, 1):
            recognized |= _spelled(goods.name) | _spelled(goods.shelf or '')
            for req in goods.musts + goods.told:
                recognized |= _spelled(' '.join(re.findall('[A-Za-z0-9]+', req.split(':', 1)[0])))
        for (prop_label, nm, _, _) in self.key_table():
            recognized |= _spelled(nm) | _spelled(prop_label)
        for rec_obj in self.full.values():
            for raw_val in (_fact_block(rec_obj) or {}).values():
                recognized |= {tok for tok in _spelled(' '.join(re.findall('[A-Za-z0-9]+', str(raw_val)))) if len(tok) >= QUIET_WORD}
        for (_, question, answer) in sent:
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            mine = recognized | _spelled(' '.join(re.findall('[A-Za-z0-9]+', asked)))
            spoken_text = _spelled(' '.join(re.findall('[A-Za-z0-9]+', answer)))
            if re.search('\\d', answer) or spoken_text & mine or re.search(ASK_WORDS, answer, re.I):
                return False
        return True

class Fitter:
    """Which parts work with which devices, from codes and series the listings show."""

    def fit_index(self) -> dict[str, Any]:
        state = (len(self.seen_order), len(self.full), sum((len(item_val) for item_val in self.views.values())))
        return self.fit_cached[1] if self.fit_cached and self.fit_cached[0] == state else self.fit_index_continued(state)

    def fit_index_continued(self, state) -> dict[str, Any]:
        """The rest of fit_index, from the point where its earlier checks are done."""
        marks_map: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for pid in self.seen_order:
            rec_obj = self.full.get(pid) or self.rows.get(pid) or {}
            phrase = ' '.join((str(item_val) for item_val in rec_obj.values() if isinstance(item_val, str)))
            phrase += ' ' + '. '.join((f'{key_name} {item_val}' for chunk in rec_obj.values() if isinstance(chunk, dict) for (key_name, item_val) in chunk.items() if isinstance(item_val, str)))
            marks_map[pid] = _fit_marks(phrase)
            for (portion, code_bag) in marks_map[pid]['also']:
                notes.setdefault(portion, set()).update(code_bag)
        self.fit_cached = (state, {'marks': marks_map, 'notes': notes})
        return self.fit_cached[1]

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        hit_found = set(mark.get('works') or [])
        if mark.get('own'):
            hit_found |= index['notes'].get(mark['own'], set())
        return hit_found

    def item_ids(self, number: int) -> list[str]:
        shelf_field = self.category_key()
        goods = self.items[number - 1]
        return [cur for cur in self.seen_order if shelf_field and self.owns(goods, str((self.rows.get(cur) or {}).get(shelf_field) or ''))]

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """(item carrying series codes, item naming them, the series) of two items bound to each other, or None when the rows read so far
        show no such system: most rows of one item carry a code of few series, and rows of the other name at least one of those series."""
        marks_map = index['marks']
        for (carrier, namer) in ((part, main), (main, part)):
            rows = self.item_ids(carrier)
            tagged = [cur for cur in rows if (marks_map.get(cur) or {}).get('fam')]
            families = {f for cur in tagged for f in marks_map[cur]['fam']}
            if len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows) or len(families) > LINK_FAMILIES:
                continue
            if any((f in families for cur in self.item_ids(namer) for f in (marks_map.get(cur) or {}).get('named') or [])):
                return (carrier, namer, families)
        return None

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        """Whether a row's own series is one the other row names; None when either row says nothing about it."""
        own = set((index['marks'].get(carried) or {}).get('fam') or []) & families
        cited = set((index['marks'].get(namer) or {}).get('named') or []) & families
        return bool(own & cited) if own and cited else None

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.link_system(part, main, index)
        return [] if not system else self.link_faults_continued(device, index, lines, main, part, system)

    def link_faults_continued(self, device, index, lines, main, part, system) -> list[str]:
        """The rest of link_faults, from the point where its earlier checks are done."""
        (carrier, _, families) = system
        marks_map = index['marks']
        issues = []
        for basket_line in lines:
            (carry_count, namer) = (basket_line, device) if carrier == part else (device, basket_line)
            if self.link_holds(carry_count, namer, families, index) is not False:
                continue
            sought = set(marks_map[namer].get('named') or []) & families
            good = [cur for cur in self.item_ids(carrier) if set(marks_map[cur].get('fam') or []) & sought and (not self.sold_out(cur)) and (cur != carry_count)][:4]
            issues.append(f"{carry_count} is of series {', '.join(sorted(set(marks_map[carry_count].get('fam') or []) & families))}, while {namer} takes {', '.join(sorted(sought))}" + (f"; seen listings of that series: {', '.join(good)}" if good else f"; search that series with the item's words, or take another {self.items[main - 1].name} whose series the wanted lines carry"))
        return issues

    def series_for(self, number: int) -> list[str]:
        """The series to search a linked item by, most useful first: those the cart's lines of the linked items hold, then those most rows of the
        linked items carry or name. [] when the item is linked by no series system."""
        index = self.fit_index()
        marks_map = index['marks']
        return self.series_for_continued(index, marks_map, number)

    def series_for_continued(self, index, marks_map, number) -> list[str]:
        """The rest of series_for, from the point where its earlier checks are done."""
        cnts: dict[str, int] = {}
        held: list[str] = []
        cart = self.cart_ids()
        for (portion, chief) in self.bound:
            system = self.link_system(portion, chief, index) if number in (portion, chief) else None
            if not system:
                continue
            for member in (portion, chief):
                for cur in self.item_ids(member):
                    tags = set((marks_map.get(cur) or {}).get('fam') or []) | set((marks_map.get(cur) or {}).get('named') or [])
                    for family in tags & system[2]:
                        cnts[family] = cnts.get(family, 0) + 1
                        if cur in cart and family not in held:
                            held.append(family)
        return (held + [f for f in sorted(cnts, key=lambda f: (-cnts[f], f)) if f not in held])[:LINK_SEARCHES]

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        return self.fit_faults_continued(index)

    def fit_faults_continued(self, index) -> list[str]:
        """The rest of fit_faults, from the point where its earlier checks are done."""
        cart = self.cart_ids()
        issues = []
        return self.fit_faults_continued_continued(cart, index, issues)

    def fit_faults_continued_continued(self, cart, index, issues) -> list[str]:
        """The rest of fit_faults_continued, from the point where its earlier checks are done."""
        for (portion, chief) in self.bound:
            devices = [i for i in cart if i in self.item_ids(chief)]
            segments = [i for i in cart if i in self.item_ids(portion)]
            if len(devices) != 1 or not segments:
                continue
            linked = self.link_faults(portion, chief, devices[0], segments, index)
            if linked:
                issues += linked
                continue
            code_bag = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not code_bag:
                continue
            barred = lambda i: set(index['marks'].get(i, {}).get('bars') or [])
            covers = lambda i: any((_code_hit(p, c) for p in self.fit_set(i, index) for c in code_bag)) and (not any((_code_hit(p, c) for p in barred(i) for c in code_bag)))
            for basket_line in segments:
                fit_list = self.fit_set(basket_line, index)
                if fit_list and (not covers(basket_line)):
                    good = [i for i in self.item_ids(portion) if covers(i) and (not self.sold_out(i)) and (i not in cart)][:4]
                    issues.append(f"{basket_line} ({self.items[portion - 1].name}) works with {', '.join(sorted(fit_list)[:4])}, not with {devices[0]} ({', '.join(sorted(code_bag))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
        return issues

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        return self.fit_plan_continued(index, short)

    def fit_plan_continued(self, index, short) -> str:
        """The rest of fit_plan, from the point where its earlier checks are done."""
        rowset = []
        for chief in sorted({mo for (_, mo) in self.bound}):
            segments = [p for (p, mo) in self.bound if mo == chief]
            tally: dict[str, dict[int, int]] = {}
            for p in segments:
                for cur in self.item_ids(p):
                    if not self.sold_out(cur):
                        for code_str in self.fit_set(cur, index):
                            tally.setdefault(code_str, {}).setdefault(p, 0)
                            tally[code_str][p] += 1
            for p in segments:
                system = self.link_system(p, chief, index)
                for cur in self.item_ids(p) if system and system[0] == p else []:
                    for code_str in set((index['marks'].get(cur) or {}).get('fam') or []) & system[2] if not self.sold_out(cur) else ():
                        tally.setdefault(code_str, {}).setdefault(p, 0)
                        tally[code_str][p] += 1
            sorted_ids = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
            devices = {code_str: [cur for cur in self.item_ids(chief) if code_str in ((index['marks'].get(cur) or {}).get('named') or []) or any((_code_hit(code_str, c) for c in index['marks'].get(cur, {}).get('codes') or []))][:2] for (code_str, _) in sorted_ids}
            phrase = '; '.join((f'{code_str}: ' + ', '.join((f'{idx} {self.items[p - 1].name}' for (p, idx) in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code_str])})" if devices[code_str] else ' (device not seen yet)') for (code_str, per) in sorted_ids))
            if phrase:
                rowset.append(f"codes the {self.items[chief - 1].name} candidates' parts name -> {phrase}")
            sets = self.cheapest_sets(chief, segments, index) if self.brief.get('pick') == 'cheapest' else ''
            if sets:
                rowset.append(sets)
        return self.fit_plan_continued_continued(rowset, short)

    def fit_plan_continued_continued(self, rowset, short) -> str:
        """The rest of fit_plan_continued, from the point where its earlier checks are done."""
        return '' if not rowset else ' | '.join(rowset) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code; try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] " + ' | '.join(rowset)

    def part_shelves(self) -> None:
        """An item that shares its shelf with another requested item takes the store label that is its own name, when one is seen
        and no item holds it yet."""
        shelf_field = self.category_key()
        prop_labels = sorted({str(one_row.get(shelf_field) or '').strip() for one_row in self.rows.values()} - {''}) if shelf_field else []
        return self.part_shelves_continued(prop_labels)

    def part_shelves_continued(self, prop_labels) -> None:
        """The rest of part_shelves, from the point where its earlier checks are done."""
        relocated = False
        for goods in self.items:
            nm = _roots(goods.name)
            if not nm or not goods.shelf.strip() or (not any((second is not goods and _words(second.shelf) == _words(goods.shelf) for second in self.items))):
                continue
            if any((second is not goods and _roots(second.name) == nm for second in self.items)):
                continue
            own = [prop_label for prop_label in prop_labels if _roots(prop_label) == nm and _words(prop_label) != _words(goods.shelf)]
            if len(own) == 1 and (not any((_words(second.shelf) == _words(own[0]) for second in self.items))):
                self.trace.append({'kind': 'shelf_parted', 'item': goods.name, 'from': goods.shelf, 'to': own[0]})
                goods.shelf = own[0]
                relocated = True
        if not relocated:
            return
        self.owners_seen = {}
        self.fit_cached = None

class Basket:
    """The cart: its lines, quantities, the calls that set it and the checks before an order."""

    def cart_text(self) -> str:
        return _dump(_clip(self.cart_view, 300))[:3000] if self.cart_view is not None and self.cart_seen_at >= self.cart_at else "lines by Tern's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [item_val.strip() for item_val in node.values() if isinstance(item_val, str)]:
                        cnts = [_count(item_val) for (key_name, item_val) in node.items() if not isinstance(item_val, str) and _count(item_val) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(key_name))]
                        if len(cnts) == 1:
                            return cnts[0]
                    stack += [item_val for item_val in node.values() if isinstance(item_val, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            phrase = _dump(self.cart_view)
            return [cur for cur in self.rows if cur in phrase]
        return list(self.cart_lines)

    def missing_items(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        shelf_field = self.category_key()
        pids = [cur for cur in self.cart_ids() if (self.held_units(cur) or 0) > 0]
        if not shelf_field or len(pids) >= len(self.items):
            return []
        prop_labels = [str(rw.get(shelf_field) or '') for rw in self.rows.values()]
        held = [str((self.rows.get(cur) or {}).get(shelf_field) or '') for cur in pids]
        return [idx for (idx, goods) in enumerate(self.items, 1) if any((self.owns(goods, prop_label) for prop_label in prop_labels)) and (not any((self.owns(goods, prop_label) for prop_label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        chunk = _fact_block(self.full.get(ident) or {}) or {}
        for (owner, req) in self.needs_now():
            if not owner != number:
                (prop_word, _, raw_val) = req.partition(':')
                split = lambda text: set(re.sub('[,/;|]', ' ', _words(text)).split())
                tokens = split(raw_val)
                if not tokens or tokens & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                    return False
                if not any((_names(_spelled(prop_word), _spelled(key_name)) and tokens <= split(str(item_val)) for (key_name, item_val) in chunk.items())):
                    return False
        return True

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {idx for pair in self.bound for idx in pair}
        return self.fill_short_cart_continued(bound, cart)

    def fill_short_cart_continued(self, bound, cart) -> bool:
        """The rest of fill_short_cart, from the point where its earlier checks are done."""
        for goods_no in self.missing_items():
            goods = self.items[goods_no - 1]
            fit_list = [i for i in self.item_ids(goods_no) if i in self.full and i not in self.cart_lines and self.buyable(i) and (self.price_of(i) is not None) and self.states_needs(i, goods_no)]
            fit_list.sort(key=lambda i: self.price_of(i) or 0.0)
            selection = fit_list[0] if fit_list and goods_no not in bound else None
            if selection and isinstance(self.budget, (int, float)):
                held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
                cost = (self.price_of(selection) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else goods.qty)
                if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                    selection = None
            invocations = self.cart_calls({**{i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}, selection: goods.qty}, []) if selection else None
            if not invocations or len(invocations) + 1 > self.left():
                return False
            for call_args in invocations:
                issues: list[str] = []
                call_args = _shape(call_args, self.schemas.get(cart), 'args', issues)
                if issues or self.act(cart, call_args) is None:
                    return False
                self.note_cart(cart, call_args)
            self.trace.append({'kind': 'filled_line', 'item': goods_no, 'line': selection})
        return True

    def fuller_label(self, number: int, label: str) -> str:
        shelf_field = self.category_key()
        mine = _roots(self.items[number - 1].name)
        return '' if len(mine) < 2 or mine <= _roots(label) or (not shelf_field) or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)) else next((noticed for noticed in sorted({str(rw.get(shelf_field) or '') for rw in self.rows.values()}) if noticed and mine <= _roots(noticed)), '')

    def named_label(self, number: int, label: str) -> str:
        """The store label the request named for this item, when a listing seen carries it and `label` is another one."""
        goods = self.items[number - 1]
        shelf_field = self.category_key()
        sought = _words(goods.shelf or '')
        if not sought or not shelf_field or _words(label) == sought:
            return ''
        here = next((str(rw.get(shelf_field)) for rw in self.rows.values() if _words(str(rw.get(shelf_field) or '')) == sought), '')
        return here

    def line_faults(self) -> list[str]:
        try:
            self.adopt_labels()
            shelf_field = self.category_key()
            pids = [cur for cur in self.cart_ids() if (self.held_units(cur) or 0) > 0]
            if not pids or not shelf_field:
                return []
            prop_label = {cur: str((self.rows.get(cur) or {}).get(shelf_field) or '') for cur in pids}
            recognized = {idx for (idx, goods) in enumerate(self.items, 1) if any((self.owns(goods, str(rw.get(shelf_field) or '')) for rw in self.rows.values()))}
            owner = {cur: [idx for (idx, goods) in enumerate(self.items, 1) if self.owns(goods, prop_label[cur])] for cur in pids}
            issues: list[str] = []
            for (idx, goods) in enumerate(self.items, 1):
                if not idx not in recognized:
                    rowset = [cur for cur in pids if idx in owner[cur]]
                    if not rowset:
                        issues.append(f'item {idx} ({goods.name}) has no line in the cart')
                    elif len([cur for cur in rowset if owner[cur] == [idx]]) > 1:
                        issues.append(f"item {idx} ({goods.name}) has {len(rowset)} lines ({', '.join(rowset)}); keep one")
                    for cur in rowset:
                        held_now = self.held_units(cur)
                        if owner[cur] == [idx] and held_now != goods.qty:
                            issues.append(f'{cur} is held at {held_now} unit(s); item {idx} ({goods.name}) wants {goods.qty}')
                        fuller = self.fuller_label(idx, prop_label[cur]) or self.named_label(idx, prop_label[cur])
                        if fuller:
                            issues.append(f'{cur} is from {prop_label[cur]}, but the store has the label {fuller}, which names item {idx} ({goods.name}) in full: take the line from there')
            if len(recognized) == len(self.items):
                for cur in pids:
                    if not owner[cur]:
                        issues.append(f"{cur} ({prop_label[cur] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
            for cur in pids:
                if self.sold_out(cur):
                    issues.append(f'{cur} is sold out')
            costs = {cur: self.price_of(cur) for cur in pids}
            if isinstance(self.budget, (int, float)) and all((p is not None for p in costs.values())):
                if self.brief.get('budget_scope') == 'each':
                    over = [cur for cur in pids if self.budget < costs[cur] <= 10 * self.budget]
                    issues += [f'{cur} costs {costs[cur]:g}, over the {self.budget:g} limit for each item' for cur in over]
                else:
                    total = sum((costs[cur] * self.held_units(cur) for cur in pids))
                    if self.budget + 1e-09 < total <= 10 * self.budget:
                        issues.append(f'the total is {total:g}, over the budget of {self.budget:g}')
            return issues
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'line_faults', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def count_fixes(self) -> list[tuple[str, int, int]]:
        shelf_field = self.category_key()
        return [] if not shelf_field else self.count_fixes_continued(shelf_field)

    def count_fixes_continued(self, shelf_field) -> list[tuple[str, int, int]]:
        """The rest of count_fixes, from the point where its earlier checks are done."""
        acc = []
        for pid in self.cart_ids():
            held_now = self.held_units(pid) or 0
            owners = [idx for (idx, goods) in enumerate(self.items, 1) if self.owns(goods, str((self.rows.get(pid) or {}).get(shelf_field) or ''))]
            mine = [cur for cur in self.cart_ids() if (self.cart_lines.get(cur, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(cur) or {}).get(shelf_field) or ''))]
            if held_now > 0 and len(owners) == 1 and (len(mine) == 1) and (held_now != self.items[owners[0] - 1].qty) and (not self.sold_out(pid)):
                acc.append((pid, held_now, self.items[owners[0] - 1].qty))
        return acc

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        try:
            nm = self.roles.get('cart') or ''
            shape_spec = self.schemas.get(nm) or {}
            prop_map = shape_spec.get('properties') or {}
            text_fields = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'string' in _kinds(item_val) and ('enum' not in item_val)]
            goal = next((key_name for (t, key_name) in self.id_args if t == nm and key_name in text_fields), None) or next((key_name for key_name in shape_spec.get('required') or [] if key_name in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
            cnt = next((key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and 'integer' in _kinds(item_val)), None)
            mode = next((key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and isinstance(item_val.get('enum'), list) and (len(item_val['enum']) >= 2)), None)
            if not nm or not goal or (not cnt):
                return []
            invocations: list[dict[str, Any]] = []
            for (pid, held_now, want) in wrong:
                if self.cart_sets(nm):
                    invocations.append({goal: pid, cnt: want})
                elif mode:
                    options = [str(o) for o in prop_map[mode]['enum']]
                    off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                    on = next((o for o in options if o != off), options[0])
                    invocations += [{goal: pid, mode: off, cnt: held_now}, {goal: pid, mode: on, cnt: want}]
                else:
                    return []
            return self.fix_counts_continued(invocations, nm, shape_spec)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'fix_counts', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def fix_counts_continued(self, invocations, nm, shape_spec) -> list[str]:
        """The rest of fix_counts, from the point where its earlier checks are done."""
        if len(invocations) + 3 > self.left():
            return []
        completed_ops = []
        for call_args in invocations:
            issues: list[str] = []
            call_args = _shape(call_args, shape_spec, 'args', issues)
            if issues:
                break
            answer_obj = self.act(nm, call_args)
            if answer_obj is None or self.finished:
                break
            self.note_cart(nm, call_args)
            self.cart_said = _dump(answer_obj.get('observation'))
            self.observe(nm, answer_obj.get('observation'))
            completed_ops.append(_dump(call_args) + ' -> ' + self.show(answer_obj.get('observation'))[:300])
        return self.fix_counts_continued_continued(completed_ops)

    def fix_counts_continued_continued(self, completed_ops) -> list[str]:
        """The rest of fix_counts_continued, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'count_fix', 'calls': completed_ops})
        return completed_ops

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        try:
            lead = ids is not None
            ids = ids if lead else self.cart_ids()
            fkey = self.check_key(ids)
            if fkey in self.checked or not self.judges(limit) or (not ids):
                return ''
            self.checked.add(fkey)
            instr = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
            if lead:
                quantity = '; '.join((f'item {idx}: quantity {it.qty}' for (idx, it) in enumerate(self.items, 1)))
                instr += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
            else:
                instr += '\n\nCART:\n' + self.cart_text()
            instr += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{cur}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(cur) else '') + self.record_of(cur) for cur in ids))
            if self.brief.get('pick') in ('cheapest', 'most', 'least'):
                instr += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
            judges = self.judges(limit)
            got = None
            if len(judges) > 1 and (not lead) and (not self.audited) and (self.brief.get('pick') not in ('cheapest', 'most', 'least')):
                got = self.ask_json(instr, [LIGHT], 3000)
                if got:
                    judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
            if judges and (not self.no_credit):
                got = self.ask_json(instr, judges, 3000) or got
            return self.check_cart_continued(fkey, got, ids)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'check_cart', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def check_cart_continued(self, fkey, got, ids) -> str:
        """The rest of check_cart, from the point where its earlier checks are done."""
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[fkey] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [cur for cur in self.leads if cur not in ids]
        return self.check_cart_continued_continued(got)

    def check_cart_continued_continued(self, got) -> str:
        """The rest of check_cart_continued, from the point where its earlier checks are done."""
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        phrase = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            phrase += '. Listings that look better: ' + ', '.join(better)
        return phrase + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """The lowest totals of one device of the item with one part of each bound item its codes fit, among buyable rows
        seen, requirements not checked: the pick counts the whole order, not each line on its own."""
        options = []
        systems = {portion: self.link_system(portion, main, index) for portion in parts}
        shelves = {portion: sorted(((self.price_of(cur), cur, self.fit_set(cur, index), index['marks'].get(cur, {}).get('bars') or []) for cur in self.item_ids(portion) if self.price_of(cur) is not None and self.buyable(cur)), key=lambda o: (o[0], o[1])) for portion in parts}
        for device in self.item_ids(main):
            code_bag = index['marks'].get(device, {}).get('codes') or []
            price = self.price_of(device)
            if not (code_bag or index['marks'].get(device, {}).get('named')) or price is None or (not self.buyable(device)):
                continue
            total = price * self.items[main - 1].qty
            selected = [device]
            for portion in parts:
                system = systems[portion]
                fitting = next((cur for (_, cur, fit_list, bars) in shelves[portion] if any((_code_hit(p, c) for p in fit_list for c in code_bag)) and (not any((_code_hit(p, c) for p in bars for c in code_bag))) or (system and system[0] == portion and (self.link_holds(cur, device, system[2], index) is True))), None)
                if not fitting:
                    selected = []
                    break
                selected.append(fitting)
                total += (self.price_of(fitting) or 0.0) * self.items[portion - 1].qty
            if selected:
                options.append((total, selected))
        options.sort(key=lambda o: o[0])
        return '' if not options else 'lowest whole sets seen, requirements not yet checked: ' + '; '.join((f"{' + '.join(p)} = {t:g}" for (t, p) in options[:SET_SHOWN]))

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        try:
            nm = self.roles.get('cart') or ''
            shape_spec = self.schemas.get(nm) or {}
            prop_map = shape_spec.get('properties') or {}
            for (fkey, slot_spec) in prop_map.items():
                shape = slot_spec.get('items') if isinstance(slot_spec, dict) and 'array' in _kinds(slot_spec) else None
                if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                    inner = shape['properties']
                    id_key = next((key_name for (key_name, s) in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                    count_key = next((key_name for (key_name, s) in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                    return None if not id_key else [{fkey: [{id_key: cur, **({count_key: idx} if count_key else {})} for (cur, idx) in lines.items()]}]
            text_fields = [key_name for (key_name, s) in prop_map.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)]
            goal = next((key_name for (t, key_name) in self.id_args if t == nm and key_name in text_fields), None) or next((key_name for key_name in shape_spec.get('required') or [] if key_name in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
            return self.cart_calls_continued(drop, goal, lines, nm, prop_map)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'cart_calls', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def cart_calls_continued(self, drop, goal, lines, nm, prop_map) -> list[dict[str, Any]] | None:
        """The rest of cart_calls, from the point where its earlier checks are done."""
        cnt = next((key_name for (key_name, s) in prop_map.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((key_name for (key_name, s) in prop_map.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        return None if not goal else self.cart_calls_continued_continued(cnt, drop, goal, lines, mode, nm, prop_map)

    def cart_calls_continued_continued(self, cnt, drop, goal, lines, mode, nm, prop_map) -> list[dict[str, Any]] | None:
        """The rest of cart_calls_continued, from the point where its earlier checks are done."""
        invocations: list[dict[str, Any]] = []
        for pid in drop:
            held_now = self.cart_lines.get(pid, 1)
            if mode:
                invocations.append({goal: pid, mode: prop_map[mode]['enum'][1], **({cnt: held_now} if cnt else {})})
            elif cnt and self.cart_sets(nm):
                invocations.append({goal: pid, cnt: 0})
            elif cnt and isinstance(prop_map[cnt].get('minimum'), int) and (prop_map[cnt]['minimum'] <= -held_now):
                invocations.append({goal: pid, cnt: -held_now})
            else:
                return None
        for (pid, want) in lines.items():
            held_now = self.cart_lines.get(pid, 0) if pid in self.cart_ids() else 0
            if held_now == want:
                continue
            if held_now and (not self.cart_sets(nm)):
                return None
            invocations.append({goal: pid, **({mode: prop_map[mode]['enum'][0]} if mode else {}), **({cnt: want} if cnt else {})})
        return invocations

    def cart_sets(self, name: str) -> bool:
        prop_map = self.schemas.get(name, {}).get('properties') or {}
        return False if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in prop_map.values())) else any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in prop_map.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        slot_spec = self.schemas.get(name, {}).get('properties') or {}
        cnt = next((key_name for (key_name, item_val) in slot_spec.items() if isinstance(item_val, dict) and 'integer' in _kinds(item_val)), None)
        removing = any((isinstance((slot_spec.get(key_name) or {}).get('enum'), list) and item_val in slot_spec[key_name]['enum'][1:] for (key_name, item_val) in args.items()))
        pid = next((item_val for item_val in args.values() if isinstance(item_val, str) and item_val in self.cart_lines), None)
        return {**args, cnt: min(self.cart_lines[pid], slot_spec[cnt].get('maximum', self.cart_lines[pid]))} if cnt and removing and pid and (cnt not in args) else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        try:
            self.cart_touched = True
            self.cart_at = self.step
            slot_spec = self.schemas.get(name, {}).get('properties') or {}
            removing = False
            return self.note_cart_continued(args, name, removing, slot_spec)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'note_cart', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def note_cart_continued(self, args, name, removing, slot_spec) -> None:
        """The rest of note_cart, from the point where its earlier checks are done."""
        for (fkey, raw_val) in args.items():
            options = (slot_spec.get(fkey) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (raw_val in options[1:]):
                removing = True
        loose_count = next((_count(item_val) for item_val in args.values() if _count(item_val) is not None and (not isinstance(item_val, str))), None)
        rowset: list[tuple[str, int | None]] = []
        complete = False
        for (fkey, raw_val) in args.items():
            if isinstance(raw_val, list):
                shape = (slot_spec.get(fkey) or {}).get('items') or {}
                if not raw_val and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    complete = True
                for basket_line in raw_val:
                    if isinstance(basket_line, dict):
                        complete = True
                        pid = next((item_val for item_val in basket_line.values() if isinstance(item_val, str) and item_val in self.rows), None)
                        cnt = next((_count(item_val) for item_val in basket_line.values() if _count(item_val) is not None and (not isinstance(item_val, str))), 1)
                        if pid:
                            rowset.append((pid, cnt))
                    elif isinstance(basket_line, str) and basket_line in self.rows:
                        rowset.append((basket_line, loose_count))
            elif isinstance(raw_val, str) and raw_val in self.rows:
                rowset.append((raw_val, loose_count))
        if complete:
            self.cart_lines = {cur: c for (cur, c) in rowset if c and c > 0}
            return
        sets = self.cart_sets(name)
        for (pid, cnt) in rowset:
            held_now = self.cart_lines.get(pid, 0)
            held_now = (held_now - (cnt or 1) if any(('integer' in _kinds(item_val) for item_val in slot_spec.values() if isinstance(item_val, dict))) else held_now - cnt if cnt else 0) if removing else (cnt if cnt is not None else 1) if sets else held_now + (cnt if cnt is not None else 1)
            if held_now > 0:
                self.cart_lines[pid] = held_now
            else:
                self.cart_lines.pop(pid, None)

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(rw) for rw in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        prop_map = (self.schemas.get(name) or {}).get('properties') or {}
        free = [key_name for (key_name, item_val) in prop_map.items() if isinstance(item_val, dict) and ('string' in _kinds(item_val) or 'array' in _kinds(item_val)) and ('enum' not in item_val)]
        return (args, '') if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args) else self.resolve_ids_continued(args, free)

    def resolve_ids_continued(self, args, free) -> tuple[dict[str, Any], str]:
        """The rest of resolve_ids, from the point where its earlier checks are done."""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
            lowered = value.strip().lower()
            hit_list = [cur for (cur, one_row) in self.rows.items() if any((key_name != self.id_key and isinstance(item_val, str) and item_val.strip().lower().startswith(lowered) for (key_name, item_val) in one_row.items()))]
            if len(hit_list) == 1:
                return (hit_list[0], '')
            return (value, f"several listings carry the text {value!r} ({', '.join(hit_list[:6])}); pass one listing id") if hit_list else (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        acc = dict(args)
        for fkey in free:
            if not fkey not in args:
                mended = [fix(item_val) for item_val in args[fkey]] if isinstance(args[fkey], list) else [fix(args[fkey])]
                problem = next((p for (_, p) in mended if p), '')
                if problem:
                    return (args, problem)
                acc[fkey] = [item_val for (item_val, _) in mended] if isinstance(args[fkey], list) else mended[0][0]
        return (acc, '')

class Closer:
    """Process gates, sell-outs, reviews before a decline, rescues and the finishing calls."""

    def gate(self, stage: str) -> tuple[bool, str]:
        try:
            step_rows = [p for p in self.process if p['when'] == stage]
            return (True, '') if not step_rows or self.finished or (not self.gate_due(stage)) else self.gate_continued(stage, step_rows)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'gate', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def gate_continued(self, stage, step_rows) -> tuple[bool, str]:
        """The rest of gate, from the point where its earlier checks are done."""
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        return self.gate_continued_continued(ending, stage, step_rows, tools)

    def gate_continued_continued(self, ending, stage, step_rows, tools) -> tuple[bool, str]:
        """The rest of gate_continued, from the point where its earlier checks are done."""
        spoken_text: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            instr = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in step_rows)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{cur}. {basket_line}' for (cur, basket_line) in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(instr, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if spoken_text:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            invocation = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(invocation, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Tern did the steps the request asks for:\n' + '\n'.join(spoken_text) + '\n' if spoken_text else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            nm = str((invocation or {}).get('tool') or '')
            with_examples = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk_due = stage == 'order' and any((p['messages'] > 0 for p in step_rows))
            rowset = range(1, len(self.done_log) + 1)
            proof = {cur for cur in rowset if not talk_due or (cur <= len(self.done_steps) and self.done_steps[cur - 1] > self.cart_at)}
            unproven = [e for e in with_examples if e.get('done') is not True or not any(((_count(idx) or 0) in proof for idx in _items(e.get('lines'))))]
            if not invocation and unproven and (not spoken_text) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Tern: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not invocation or nm not in self.schemas or nm in ending:
                if not invocation:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (nm == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            issues: list[str] = []
            call_args = _shape(invocation.get('args') if isinstance(invocation.get('args'), dict) else {}, self.schemas[nm], 'args', issues)
            if issues:
                break
            if nm == self.roles.get('ask'):
                phrase = max(_strings(call_args), key=len, default='')
                if not phrase:
                    break
                sent_batch = self.send([(1, '', phrase)], 1)
                if not sent_batch:
                    break
                self.gate_msgs += 1
                spoken_text.append(f'Q: {phrase}\nA: {sent_batch[0][2]}')
                continue
            answer_obj = self.act(nm, call_args)
            if answer_obj is None:
                break
            noticed = answer_obj.get('observation')
            self.absorb(noticed)
            self.done_log.append(f'step {self.step}: {nm} {_dump(call_args)} -> {self.show(noticed)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            spoken_text.append(f'{nm} {_dump(call_args)} -> {self.show(noticed)[:400]}')
        return (True, 'Before going on, Tern did the steps the request asks for:\n' + '\n'.join(spoken_text) if spoken_text else '')

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        try:
            dead_ids = [cur for cur in self.cart_ids() if cur not in self.gone and self.sold_out(cur)]
            if not dead_ids:
                return ''
            for pid in dead_ids:
                self.gone[pid] = self.step
            self.leads = [cur for cur in self.leads if cur not in dead_ids]
            self.version += 1
            return self.swap_hint_continued(dead_ids)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'swap_hint', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def swap_hint_continued(self, dead_ids) -> str:
        """The rest of swap_hint, from the point where its earlier checks are done."""
        selected = self.replacements(dead_ids)
        phrase = f"[Tern] Sold out now: {', '.join(dead_ids)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if selected and self.left() <= 6:
            return phrase + f" Few steps are left: in this turn add {selected[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(selected[1:]) or 'none'}."
        if selected and self.brief.get('pick') == 'cheapest':
            return phrase + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(selected)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {selected[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        return phrase + f" Listings already read that appear to fit that item instead: {', '.join(selected)}. In this turn take the sold-out line out of the cart and add {selected[0]} at the requested quantity, then check out." if selected else phrase + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        try:
            pids = [cur for cur in self.cart_ids() if cur not in self.gone]
            if any((self.sold_out(cur) for cur in pids)):
                return ''
            if len(pids) == 1:
                self.unavailable.add(pids[0])
                return ''
            return self.reread_offers() if pids else ''
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'find_sold', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def replacements(self, lost: list[str]) -> list[str]:
        try:
            if not self.judges():
                return []
            records = []
            for pid in reversed(self.seen_order):
                if not (pid in lost or pid in self.gone or self.sold_out(pid)):
                    records.append(_dump(_clip(self.full.get(pid) or self.rows.get(pid), 200))[:1200])
                    if len(records) >= 25:
                        break
            if not records:
                return []
            instr = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
            instr += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{cur}: {self.record_of(cur)}' for cur in lost)) + '\n\nCART NOW:\n' + self.cart_text()
            instr += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
            got = self.ask_json(instr, self.judges()[::-1], 3500)
            fit_list: list[tuple[float, str]] = []
            for ent in _items((got or {}).get('fits')):
                pid = str(ent.get('id') if isinstance(ent, dict) else ent)
                price = ent.get('price') if isinstance(ent, dict) else None
                if pid in self.rows and pid not in lost and (pid not in self.gone) and (pid not in self.cart_ids()) and (pid not in [cur for (_, cur) in fit_list]):
                    fit_list.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), pid))
            return self.replacements_continued(fit_list, lost)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'replacements', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def replacements_continued(self, fit_list, lost) -> list[str]:
        """The rest of replacements, from the point where its earlier checks are done."""
        sorted_ids = [cur for (_, cur) in sorted(fit_list, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [cur for (_, cur) in fit_list]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fit_list})
        return sorted_ids[:4]

    def audit(self) -> str:
        try:
            if self.audited or not self.judges():
                return ''
            self.audited = True
            got = self.audit_call()
            return self.audit_continued(got)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'audit', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def audit_continued(self, got) -> str:
        """The rest of audit, from the point where its earlier checks are done."""
        cited = self.audit_fits(got)
        if not cited:
            rows = [str(cur) for cur in _items((got or {}).get('open')) if str(cur) in self.rows and str(cur) not in self.opened and (not self.sold_out(str(cur)))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                cited = self.audit_fits(got)
        if not cited:
            return ''
        self.lead_items = cited
        self.leads = [cur for pids in cited.values() for cur in pids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {key_name}: {', '.join(item_val)}" for (key_name, item_val) in cited.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for pid in sorted(reversed(self.seen_order), key=lambda i: i not in self.full):
            if not self.sold_out(pid):
                rec_obj = self.full.get(pid) if pid in self.full else None
                records.append('OPENED ' + _dump(_clip(rec_obj, 200))[:1200] if rec_obj else 'ROW ' + _dump(_clip(self.rows.get(pid), 120))[:500])
                if len(records) >= 30:
                    break
        if not records:
            return None
        instr = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        instr += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        return self.audit_call_continued(instr)

    def audit_call_continued(self, instr) -> dict[str, Any] | None:
        """The rest of audit_call, from the point where its earlier checks are done."""
        judges = self.judges()
        got = self.ask_json(instr, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and (not self.no_credit) and (not self.audit_fits(got)):
            got = self.ask_json(instr, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fit_list = (got or {}).get('fits')
        if not isinstance(fit_list, dict):
            return {}
        cited = {str(key_name): [str(cur) for cur in _items(item_val) if str(cur) in self.rows and (not self.sold_out(str(cur)))][:4] for (key_name, item_val) in fit_list.items()}
        return {key_name: item_val for (key_name, item_val) in cited.items() if item_val}

    def complete_cart(self) -> str:
        try:
            (cart, buy) = (self.roles.get('cart'), self.roles.get('buy'))
            if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
                return ''
            self.completed = True
            shelf_field = self.category_key()

            def kind(ident: str) -> str:
                return str((self.rows.get(ident) or {}).get(shelf_field) or '').strip().lower() if shelf_field else ident
            plan: dict[str, int] = {}
            for (goods_no, pids) in self.lead_items.items():
                hit_found = re.search('\\d+', str(goods_no))
                index = int(hit_found.group()) if hit_found else 1 if len(self.items) == 1 else None
                selection = next((cur for cur in pids if not self.sold_out(cur)), None)
                if selection and index and (1 <= index <= len(self.items)) and (kind(selection) not in {kind(p) for p in plan}):
                    plan[selection] = self.items[index - 1].qty
            if self.confirmed not in plan:
                return ''
            now = self.cart_ids()
            reserve_n = {cur: self.cart_lines.get(cur, 1) for cur in now if cur not in plan and (not self.sold_out(cur)) and (kind(cur) not in {kind(p) for p in plan})}
            drop = [cur for cur in now if cur not in plan and cur not in reserve_n]
            if len(plan) + len({kind(cur) for cur in reserve_n}) < len(self.items):
                return ''
            rowset = {**reserve_n, **plan}
            return self.complete_cart_continued(buy, cart, drop, rowset)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'complete_cart', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def complete_cart_continued(self, buy, cart, drop, rowset) -> str:
        """The rest of complete_cart, from the point where its earlier checks are done."""
        invocations = self.cart_calls(rowset, drop)
        if not invocations or len(invocations) + 1 > self.left():
            return ''
        self.fill_facts(list(rowset), len(invocations) + 3)
        self.show_links(list(rowset), len(invocations) + 3)
        self.check_cart(list(rowset), SPEND_HARD)
        return '' if not self.verdicts.get(self.check_key(list(rowset))) else self.complete_cart_continued_continued(buy, cart, drop, invocations, rowset)

    def complete_cart_continued_continued(self, buy, cart, drop, invocations, rowset) -> str:
        """The rest of complete_cart_continued, from the point where its earlier checks are done."""
        completed_ops = []
        for call_args in invocations:
            issues: list[str] = []
            call_args = _shape(call_args, self.schemas.get(cart), 'args', issues)
            if issues:
                return ''
            answer_obj = self.act(cart, call_args)
            if answer_obj is None or self.finished:
                return ''
            self.note_cart(cart, call_args)
            self.observe(cart, answer_obj.get('observation'))
            completed_ops.append(_dump(call_args))
        issues = []
        call_args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', issues)
        if issues:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(rowset), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(rowset))) or self.left() < 1 or self.finished or any((self.sold_out(cur) for cur in rowset)):
                    break
            (go, spoken_text) = self.gate('order')
            if spoken_text:
                replies.append(spoken_text)
            if not go:
                break
            answer_obj = self.act(buy, call_args)
            self.trace.append({'kind': 'completed', 'lines': rowset, 'drop': drop, 'reply': _clip(answer_obj, 200)})
            if answer_obj is None:
                break
            self.buys += 1
            if not answer_obj.get('finished'):
                self.refused_at = self.step
            self.observe(buy, answer_obj.get('observation'))
            replies.append(self.show(answer_obj.get('observation')))
            if self.finished or answer_obj.get('finished'):
                break
        return '' if not replies else 'Not run (Tern acted instead): the independent check confirmed ' + self.confirmed + ', so Tern set the cart (' + '; '.join(completed_ops) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        mended = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(key_name) == item_val for (key_name, item_val) in mended.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        mended = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(key_name) == item_val for (key_name, item_val) in mended.items()))

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        return self.offered_continued(ask, left)

    def offered_continued(self, ask, left) -> list[dict[str, Any]]:
        """The rest of offered, from the point where its earlier checks are done."""
        reserve_n: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            reserve_n = set(ends)
        elif left <= self.tail() + 3:
            reserve_n = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            reserve_n = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask or self.message_args('x' * 200) is None]
        if reserve_n is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in reserve_n]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        shape_spec = self.schemas.get(name) or {}
        prop_map = shape_spec.get('properties') or {}
        return self.quiet_continued(args, prop_map, shape_spec)

    def quiet_continued(self, args, prop_map, shape_spec) -> dict[str, Any]:
        """The rest of quiet, from the point where its earlier checks are done."""
        acc = dict(args)
        for (fkey, slot_spec) in prop_map.items():
            if not (not isinstance(slot_spec, dict) or 'string' not in _kinds(slot_spec) or 'enum' in slot_spec):
                raw_val = acc.get(fkey)
                if raw_val is None and fkey not in (shape_spec.get('required') or []):
                    continue
                phrase = str(raw_val or '')
                cited = any((pid in phrase for pid in self.rows)) or self.brief.get('quiet_decline') is not False
                if cited or not phrase:
                    acc[fkey] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:slot_spec.get('maxLength', 200)]
        return acc

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        issues: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', issues)
        return not issues

    def finish(self, again: bool=False) -> str:
        try:
            if self.finished or self.broke or self.left() <= 0:
                return ''
            (buy, refuse) = (self.roles.get('buy'), self.roles.get('refuse'))
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            return self.finish_continued(again, buy, fresh, refuse)
        except Exception as exc:
            if not getattr(exc, '_where_noted', False):
                self.trace.append({'kind': 'failed_in', 'failed_in': 'finish', 'where': str(exc)})
                try:
                    exc._where_noted = True
                except AttributeError:
                    pass
            raise

    def finish_continued(self, again, buy, fresh, refuse) -> str:
        """The rest of finish, from the point where its earlier checks are done."""
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        complete = True
        if holding and len(self.items) >= 2:
            complete = self.fill_short_cart() or not self.can_refuse()
            if complete and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not complete:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        return self.finish_continued_continued(again, buy, complete, fresh, holding, refuse)

    def finish_continued_continued(self, again, buy, complete, fresh, holding, refuse) -> str:
        """The rest of finish_continued, from the point where its earlier checks are done."""
        if self.finished or self.left() <= 0:
            return ''
        if complete and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            issues: list[str] = []
            call_args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', issues)
            if not issues:
                if self.left() >= 2:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                answer_obj = self.act(buy, call_args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(answer_obj, 200)})
                if self.finished or self.left() <= 0:
                    return ''
                if answer_obj is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, answer_obj.get('observation'))
                    return 'the checkout was refused: ' + self.show(answer_obj.get('observation'))
        if refuse:
            issues = []
            call_args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', issues)
            if not issues:
                if self.left() >= 2:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                answer_obj = self.act(refuse, call_args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(answer_obj, 200)})
                if answer_obj is not None and (not self.finished):
                    self.observe(refuse, answer_obj.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(answer_obj.get('observation'))
        return ''

class Shopper(Closer, Basket, Fitter, Interviewer, Hunter, Shelves, Brief, Clockwork):
    """One shopping episode: the state every role reads, and the tool loop."""

    def __init__(self, problem: dict[str, Any]):
        self.bind_world(problem)
        self.start_ledger()
        self.start_catalog()
        self.start_reading()
        self.start_interview()
        return self.init___continued()

    def init___continued(self):
        """The rest of __init__, from the point where its earlier checks are done."""
        self.start_lanes()
        self.start_basket()
        self.start_checks()
        self.start_endings()
        self.start_trails()
        self.start_misc()

    def bind_world(self, problem: dict[str, Any]) -> None:
        """The episode's binding, tools, schemas, brief and limits."""
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        return self.bind_world_continued()

    def bind_world_continued(self) -> None:
        """The rest of bind_world, from the point where its earlier checks are done."""
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        step_rows = _count(self.view.get('max_steps'))
        if step_rows is None and isinstance(self.view.get('max_steps'), float):
            step_rows = int(self.view['max_steps'])
        self.max_steps = step_rows if step_rows and step_rows > 0 else 30

    def start_ledger(self) -> None:
        """Starting state: started, step, finished, broke and more."""
        self.started = time.monotonic()
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        return self.start_ledger_continued()

    def start_ledger_continued(self) -> None:
        """The rest of start_ledger, from the point where its earlier checks are done."""
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = _Stamped(self.clock)
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}

    def start_catalog(self) -> None:
        """Starting state: items, budget, msg_limit, msg_used and more."""
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        return self.start_catalog_continued()

    def start_catalog_continued(self) -> None:
        """The rest of start_catalog, from the point where its earlier checks are done."""
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}

    def start_reading(self) -> None:
        """Starting state: full, seen_order, heard, notes and more."""
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        return self.start_reading_continued()

    def start_reading_continued(self) -> None:
        """The rest of start_reading, from the point where its earlier checks are done."""
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.note_named = False
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False

    def start_interview(self) -> None:
        """Starting state: offer_seen, gone, unavailable, sold_lines and more."""
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        return self.start_interview_continued()

    def start_interview_continued(self) -> None:
        """The rest of start_interview, from the point where its earlier checks are done."""
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []

    def start_lanes(self) -> None:
        """Starting state: dropped, late_held, lead_items, verdicts and more."""
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        return self.start_lanes_continued()

    def start_lanes_continued(self) -> None:
        """The rest of start_lanes, from the point where its earlier checks are done."""
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.adopted: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.factless: set[tuple[str, str]] = set()
        self.trail_room: int | None = None
        self.id_key: str | None = None

    def start_basket(self) -> None:
        """Starting state: facts_from, queued, buys, refusals and more."""
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        return self.start_basket_continued()

    def start_basket_continued(self) -> None:
        """The rest of start_basket, from the point where its earlier checks are done."""
        self.opened: set[str] = set()
        self.views: dict[str, set[str]] = {}
        self.fit_cached: tuple[Any, dict[str, Any]] | None = None
        self.owners_seen: dict[tuple[str, int], int] = {}
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0

    def start_checks(self) -> None:
        """Starting state: trail_blocks, compare_ok, compare_bad, tool_keys and more."""
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        return self.start_checks_continued()

    def start_checks_continued(self) -> None:
        """The rest of start_checks, from the point where its earlier checks are done."""
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []

    def start_endings(self) -> None:
        """Starting state: plan_order, lanes, lane_found, lane_spent and more."""
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        return self.start_endings_continued()

    def start_endings_continued(self) -> None:
        """The rest of start_endings, from the point where its earlier checks are done."""
        self.one_product: set[int] = set()
        self.list_asks = 0
        self.apart: dict[str, dict[int, list[str]]] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []

    def start_trails(self) -> None:
        """Starting state: fit_holds, searched, link_searches, link_seen and more."""
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.hunts = 0
        self.env_gap = ENV_GAP
        return self.start_trails_continued()

    def start_trails_continued(self) -> None:
        """The rest of start_trails, from the point where its earlier checks are done."""
        self.env_spent = 0.0
        self.env_count = 0
        self.read_calls: dict[str, Any] = {}
        self.read_args: dict[str, tuple[str, dict[str, Any]]] = {}
        self.value_rows: set[str] = set()
        self.row_keys: set[str] = set()
        self.refuse_trailed = False

    def start_misc(self) -> None:
        """Starting state: env_at, hunt_stem, hunt_no_noun, bare_pages and more."""
        self.env_at = 0.0
        self.hunt_stem: dict[tuple[int, tuple[str, ...]], str] = {}
        self.hunt_no_noun: set[int] = set()
        self.bare_pages: dict[str, int] = {}
        self.code_tries: list[bool] = []
        self.codes_dead = False
        return self.start_misc_continued()

    def start_misc_continued(self) -> None:
        """The rest of start_misc, from the point where its earlier checks are done."""
        self.hear_reads = 0
        self.fit_hunted = False
        self.hunt_mark = ''
        self.hunt_tried: dict[int, set[str]] = {}
        self.page_args: dict[str, str] = {}
        self.page_queries: dict[str, set[str]] = {}

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        acc: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        return self.run_calls_continued(acc, calls)

    def run_calls_continued(self, acc, calls) -> list[tuple[str, str]]:
        """The rest of run_calls, from the point where its earlier checks are done."""
        for (index, invocation) in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = invocation['id']
            if invocation.get('joined'):
                acc.append((cid, 'Shown in the side-by-side result above: Tern opened these listings together in one step.'))
                continue
            fn = invocation.get('function') or {}
            nm = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                acc.append((cid, 'Not run: the episode is over.'))
                continue
            if nm == self.roles.get('ask') and self.message_args('x' * 200) is not None:
                acc.append((cid, "Not run (no step used): Tern's interviewer handles every message to the shopper."))
                continue
            if nm not in self.schemas:
                acc.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if nm not in allowed:
                acc.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                call_args = raw
            else:
                call_args = _json_obj(raw) if (raw or '').strip() else {}
                if call_args is None:
                    acc.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            issues: list[str] = []
            call_args = _shape(call_args, self.schemas[nm], 'args', issues)
            if issues:
                acc.append((cid, 'Not run (no step used): ' + '; '.join(issues[:6])))
                continue
            (call_args, wrong) = self.resolve_ids(nm, call_args)
            if wrong:
                acc.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if nm == self.roles.get('cart'):
                call_args = self.whole_removal(nm, call_args)
            trimmed = ''
            many = self.compare_target() if nm == self.roles.get('compare') else None
            if many and isinstance(call_args.get(many), list) and (len(call_args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Tern] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, call_args[many][room:]))}."
                call_args = {**call_args, many: call_args[many][:room]}
            goal = self.single_target(nm)
            if goal and nm in (self.roles.get('open'), self.roles.get('offer')) and (str(call_args.get(goal)) in self.gone):
                acc.append((cid, f'Not run (no step used): {call_args[goal]} sold out at step {self.gone[str(call_args[goal])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if nm == self.roles.get('cart') and any(((_count(item_val) or 0) < 0 for item_val in call_args.values() if not isinstance(item_val, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [item_val for item_val in call_args.values() if isinstance(item_val, str) and item_val in self.rows and (item_val not in self.cart_lines) and (item_val not in view) and (item_val not in self.cart_said)]
                if absent:
                    acc.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(nm, call_args) or self.is_refuse(nm, call_args)
            plain_jobs = {tool_id for (job, tool_id) in self.roles.items() if job != 'search'}
            if ending and len(calls) > 1:
                acc.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                acc.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            fresh = nm not in plain_jobs and self.query_is_new(nm, call_args)
            asked_for = call_args
            call_args = self.first_look(nm, call_args, fresh)
            if call_args != asked_for:
                trimmed += f'\n[Tern] This is a new search, so Tern asked for its first page ({_dump(asked_for)} was the call you sent).'
            call_args = self.row_ids(nm, call_args)
            if nm == self.roles.get('search') and self.codes_dead and self.names_code(call_args):
                acc.append((cid, "Not run (no step used): this store's search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows."))
                continue
            mark = nm + _dump(call_args)
            if not ending and nm != self.roles.get('cart') and (mark in self.recent):
                acc.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(rw) for rw in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if nm in readers and cached and (cached[0] == self.version):
                acc.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if goal and nm in readers and (nm != self.roles.get('compare')) and (self.shown_already(nm, str(call_args.get(goal))) or (nm, str(call_args.get(goal))) in self.factless):
                acc.append((cid, f'Not run (no step used): {call_args[goal]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(call_args[goal]))))
                continue
            if many and isinstance(call_args.get(many), list) and call_args[many]:
                reserve_n = [cur for cur in call_args[many] if not self.shown_already(nm, str(cur))]
                if not reserve_n:
                    acc.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{cur}: {self.record_of(str(cur))}' for cur in call_args[many]))[:4000]))
                    continue
                if len(reserve_n) < len(call_args[many]):
                    trimmed += f"\n[Tern] Left out {', '.join((str(cur) for cur in call_args[many] if cur not in reserve_n))}: shown in full already, nothing changed since."
                    call_args = {**call_args, many: reserve_n}
                    mark = nm + _dump(call_args)
            if ending and self.left() > 2:
                if self.is_buy(nm, call_args):
                    hunted = self.hunt_cheaper()
                    wider = self.wider_hold()
                    if wider:
                        acc.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 5) and (not self.rushed()):
                        issues = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', issues)
                        view = self.act(self.roles['cart_view'], view_args) if not issues else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                acc.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.swap_hint()
                    held = [cur for cur in self.cart_ids() if cur in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        acc.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_text()))
                        continue
                    (chain, hold) = self.trail()
                    if hold:
                        acc.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        acc.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    issues = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    mended = []
                    if issues and all((' is held at ' in f for f in issues)):
                        mended = self.fix_counts(self.count_fixes())
                        issues = self.line_faults() if mended else issues
                        if mended and (not issues):
                            self.line_holds += 1
                            acc.append((cid, "Not run yet (no step used). Tern set the cart's quantities itself: " + '; '.join(mended) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if issues:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': issues[:6]})
                        acc.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(issues[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    rowset = self.cart_ids()
                    readings = self.fill_facts(rowset, self.tail() + (1 if len(rowset) >= 2 else 3))
                    self.show_links(rowset, self.tail() + 1)
                    readings += self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        noticed = "Tern read these listings' attributes for the check (one step each):\n" + '\n'.join(readings) + '\n' if readings else ''
                        noticed = (hunted + ' ' + self.below_cart() + '\n' if hunted else '') + noticed
                        acc.append((cid, 'Not run yet. ' + chain + noticed + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        acc.append((cid, "Not run yet (no step used). Tern re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    pointed = self.refuse_trail()
                    if pointed:
                        acc.append((cid, 'Not run yet (no step used). ' + pointed))
                        continue
                    self.hunt_fits()
                    review = self.audit()
                    if review:
                        acc.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        acc.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            acc.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refuse(nm, call_args) and self.left() <= 5 and (not self.late_held):
                rowset = [cur for cur in self.built if not self.sold_out(cur)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [cur for cur in rowset if cur not in self.cart_ids() or (cur in self.dropped and (not fresh))]
                if len(rowset) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': rowset, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    acc.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(nm, call_args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                acc.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(nm, call_args):
                completed_ops = self.complete_cart()
                if completed_ops:
                    acc.append((cid, completed_ops + (' [episode finished]' if self.finished else '')))
                    continue
                call_args = self.quiet(nm, call_args)
            if self.is_buy(nm, call_args):
                late = self.late_question()
                if late:
                    acc.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if ending and self.left() >= 2:
                (go, spoken_text) = self.gate('order' if self.is_buy(nm, call_args) else 'end')
                if not go:
                    acc.append((cid, 'Not run yet. ' + spoken_text + '\nCart now: ' + self.cart_text()))
                    continue
                if spoken_text:
                    trimmed = '\n[Tern] ' + spoken_text + trimmed
            answer_obj = self.act(nm, call_args)
            if answer_obj is None:
                acc.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = answer_obj.get('observation')
            if many and isinstance(call_args.get(many), list) and (len(call_args[many]) > 1):
                with_examples = _dump(observation)
                if not any((str(cur) in with_examples for cur in call_args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(call_args[many]))
                    trimmed += f'\n[Tern] The side-by-side view showed nothing for {len(call_args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(call_args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((nm, key_name) for (key_name, item_val) in call_args.items() if (item_val if isinstance(item_val, list) else [item_val]) and all((isinstance(el, str) and el in self.rows for el in (item_val if isinstance(item_val, list) else [item_val])))))
            self.absorb(observation)
            if nm == self.roles.get('compare'):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[nm] = self.facts_from.get(nm, False) or any((_fact_block(rw) for (_, rw) in rows))
            if nm in readers - {self.roles.get('search')}:
                self.note_view(nm, observation)
            plain = {tool_id for (job, tool_id) in self.roles.items() if job != 'search'}
            paged = self.more_pages(nm, call_args, observation) if nm not in plain and (not ending) else ''
            paged += self.last_page_first(nm, call_args, observation) if fresh and (not ending) else ''
            paged += self.shorter_search(nm, call_args, observation) if nm not in plain and (not ending) else ''
            if self.single_target(nm) and nm not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.note_facts(nm, bool(_fact_block(observation)))
                    if nm == self.roles.get('open') and self.facts_from.get(nm) is False:
                        second = next((t for t in self.schemas if t != nm and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search'))) and (not self.lists_things(t))), None)
                        if second:
                            self.roles['open'] = second
                    if self.facts_from.get(nm) is False and nm != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = nm
                    if nm == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if nm == self.roles.get('cart'):
                self.note_cart(nm, call_args)
                self.cart_said = _dump(observation)
                self.recent = []
                cnts = [_count(item_val) for item_val in call_args.values() if not isinstance(item_val, str) and _count(item_val) is not None]
                for raw_val in call_args.values():
                    if cnts and isinstance(raw_val, str) and (raw_val in self.rows):
                        if cnts[0] <= 0 and (not _in_rows(observation, raw_val)):
                            self.dropped.add(raw_val)
                        elif cnts[0] > 0:
                            self.dropped.discard(raw_val)
                rowset = [cur for cur in self.cart_ids() if not self.sold_out(cur) and cur not in self.dropped]
                if len(rowset) >= len(self.built):
                    self.built = rowset
            if nm == self.roles.get('search') and self.names_code(call_args):
                self.code_tries.append(bool(self.records_in(observation)))
                self.codes_dead = len(self.code_tries) >= 2 and (not any(self.code_tries))
            if nm == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(call_args)])[-8:]
            if nm == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(nm, call_args):
                self.buys += 1
                if not answer_obj.get('finished'):
                    self.refused_at = self.step
            phrase = self.show(observation) + paged + trimmed
            if nm not in plain and (not ending):
                phrase += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(nm, call_args, observation)
            base = phrase
            proof = self.evidence(observation)
            if proof:
                phrase += '\n' + proof
            if nm == self.roles.get('cart') and any((_count(item_val) == 0 for item_val in call_args.values() if not isinstance(item_val, str))):
                if any((isinstance(item_val, str) and item_val in self.rows and _in_rows(observation, item_val) for item_val in call_args.values())):
                    phrase += '\n[Tern] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            rowset = self.loose_lines(observation)
            if rowset:
                meaning = self.take_note(rowset)
                if meaning:
                    phrase += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.is_buy(nm, call_args) and (not answer_obj.get('finished'))
            again = refused and any((basket_line in self.spoken(observation) for basket_line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(nm, call_args))))
            if (loose or again) and self.cart_ids() and (not self.finished):
                noticed = self.find_sold()
                if noticed:
                    phrase += "\n[Tern re-read the cart's lines after the refusal]\n" + noticed
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                phrase += '\n' + swap
            if nm == self.roles.get('cart') and self.bound and (not self.finished):
                misfit = self.fit_faults()
                if misfit:
                    phrase += '\n[Tern] These lines do not work together yet: ' + '; '.join(misfit)
            if self.queued:
                phrase += "\n[notes seen on Tern's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if nm in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if answer_obj.get('finished'):
                phrase += ' [episode finished]'
            acc.append((cid, phrase))
        return acc

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            opening = _restore(opening)
            self.absorb(opening)
        relocated = self.first_moves() if not self.finished else ''
        self.env_gap = ENV_GAP
        return self.run_continued(opening, relocated)

    def run_continued(self, opening, relocated) -> list[dict[str, Any]]:
        """The rest of run, from the point where its earlier checks are done."""
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        if relocated:
            intro += '\n\nTern has searched for each item and read some of its listings (one step each):\n' + relocated + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        return self.run_continued_continued()

    def run_continued_continued(self) -> list[dict[str, Any]]:
        """The rest of run_continued, from the point where its earlier checks are done."""
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Tern] Tern tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
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

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Shopper(problem_data)
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
