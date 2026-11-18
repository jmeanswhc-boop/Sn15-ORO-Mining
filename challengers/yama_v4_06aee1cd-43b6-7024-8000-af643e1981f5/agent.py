

























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
SHOW_CHARS = 450
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
VALUE_PER_ITEM = True
VALUE_SHARED = 8
VALUE_EACH = 5
VALUE_FLOOR = 22
VALUE_NEXT = 12
EMPTY_FILL = True
TEXTROW_ON = True
BUY_UNTIL_TOLD = 2
SCARCE_MESSAGES = 2
LIST_SIZE = 8
DRY_LANE = True
REFUSE_REJECTED = True
ASK_ROLE = True
REPUT = True
DECLNAME = True
IDMATCH = True
FINDEND = True
LONGTEXT = True
LONG_KEYS = 3
LONG_CLIP = 420
LONG_LINES = 9
PARTIALFIRST = False  
LONGREAD = True
SRCLINE = True
ALLREC = True
SECONDEND = True
SECONDEND_TOLD = 2
FULLONLY = True
NEARFILL = True
NEARRANK = True
STATERANK = True
GAPFILL = True
ROWCUE = True
NEARASK = True
NEARASK_EACH = 12
NEARASK_CHARS = 36000
RECSEARCH = True
RECSEARCH_EACH = 2
RECSEARCH_RUNS = 3
RECSEARCH_ROWS = 8
RECSEARCH_CHARS = 12000
RECSEARCH_KEEP = 8
RETRY_SWAP = 4
RECARGS = True
RECARGS_SCHEMA = 3000
RESEEK_ARGS = 'Some items below still have no fitting listing. The fitting product is most often on a later page of the search already run for that item, or hidden by number limits or other filters that search added and the item does not need; less often the store names it with words no search used yet. For each item, write one or two new searches, preferring in this order: (1) that item\'s earlier search, same words, repeated on its next page (page 3 at most); when that search had any number limit, drop all number limits, and drop the other extra filters as well; (2) only when none of the item\'s earlier searches had a number limit: the same search with the single number limit the request states, when it states one and the tool takes it; (3) a two to five word query in the wording the listings seen for the item use for this kind of product, never used before. Only use argument fields the search tool below defines, and do not repeat any recent search exactly.\nReply with ONE JSON object only: {"searches": [{"item": item number, "args": {all arguments for the search tool}}]}'
RESEEK = 'Some items below still have no listing that fits, and a store often files the right product under words the searches so far never used. For each item, look at the listings already seen for it (if any) and suggest one or two fresh search phrases built from the wording those listings use for this kind of product: its type, category or feature words. Skip any phrase already searched. Each phrase is two to five words about the item itself, never a different kind of product.\nReply with ONE JSON object only: {"searches": [{"item": item number, "query": "search phrase"}]}'
NEARPICK = 'A shopping agent must check out now, and the requested items below still have no line in its cart. For each item choose one listing from that item\'s candidates, judging only from the records shown (OPENED: the agent read the listing\'s page; ROW: a search result row only). A requirement on a property holds when the record states that value for the property (any key spelling, case, singular or plural, plain synonyms; a value that contains the wanted words counts; units converted; limits include the edge), and fails when an attribute naming that property states another value. The listing\'s category label must be the item itself, not a neighbouring kind. Take a listing that meets every requirement of its item; when none does, the one that meets the most. Never take one shown sold out or priced above the budget. When the request asks for the cheapest, take the lowest price among those that meet everything. Candidates come in the agent\'s own order: when two are equally good, keep the earlier one.\nReturn ONE JSON object and nothing else: {"picks": [{"item": item number, "id": the listing id copied exactly}]}'
CARTREADY = True
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Yama, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Yama knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Yama\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Yama carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Yama\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Yama adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Yama reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _codes(hit.group(1))
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

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
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
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
    
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

_ROWS_READ = [0]


def _bare_value(text: str) -> Any:
    
    word = text.strip()
    if not word.isascii() or len(word) > 13:
        return word
    digits = word[1:] if word.startswith('-') else word
    if digits.isdecimal() and len(digits) <= 6 and (digits == '0' or not digits.startswith('0')):
        return int(word)
    whole, dot, frac = digits.partition('.')
    if dot and whole.isdecimal() and frac.isdecimal() and len(whole) <= 7 and len(frac) <= 4 and (whole == '0' or not whole.startswith('0')):
        return float(word)
    if len(word) > 5:
        return word
    try:
        flag = json.loads(word.lower())
    except ValueError:
        flag = None
    return flag if isinstance(flag, bool) else word


def _field_ok(name: str) -> bool:
    name = name.strip()
    return len(name) <= 40 and len(name.split()) <= 4 and re.search('[A-Za-z]', name) is not None and re.search('[|=;{}\\[\\]"]', name) is None


def _split_fields(text: str) -> list[list[str]] | None:
    
    for mark in ('|', '\n'):
        pieces = [p.strip() for p in text.split(mark) if p.strip()] if mark in text else []
        if not pieces:
            continue
        fields: list[list[str]] = []
        stray = 0
        for piece in pieces:
            name, colon, rest = piece.partition(':')
            if colon and rest.strip() and _field_ok(name):
                fields.append([name.strip(), rest.strip()])
                continue
            stray += 1
            if fields:
                fields[-1][1] = fields[-1][1] + ' | ' + piece
        if mark == '\n' and stray:
            continue
        if len(fields) >= 3 and 5 * len(fields) >= 3 * len(pieces):
            return fields
    return None


def _eq_pairs(text: str) -> dict[str, Any] | None:
    
    if text.count('=') < 2:
        return None
    for mark in (';', ','):
        chunks = [c for c in text.split(mark) if c.strip()]
        found: list[list[str]] = []
        for chunk in chunks:
            name, eq, rest = chunk.partition('=')
            if eq and len(name.strip()) <= 40 and re.search('[A-Za-z]', name):
                found.append([name.strip(), rest.strip()])
            elif not found:
                break
            else:
                found[-1][1] = found[-1][1] + mark + chunk
        if len(found) >= 2 and 5 * len(found) >= 3 * len(chunks):
            return {name: _bare_value(rest) for name, rest in found}
    return None


def _row_from_text(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or len(value) < 12 or value.count(':') < 3:
        return None
    fields = _split_fields(value)
    if fields is None:
        return None
    record: dict[str, Any] = {}
    for name, rest in fields:
        if name not in record:
            record[name] = _eq_pairs(rest) or _bare_value(rest)
    _ROWS_READ[0] += 1
    return record


def _as_records(node: Any, depth: int=0) -> Any:
    
    if depth > 6:
        return node
    if isinstance(node, str):
        return _row_from_text(node) or node
    if isinstance(node, dict):
        if len(node) == 1:
            record = _row_from_text(next(iter(node.values())))
            if record is not None:
                return record
        return {k: v if k == 'user_message' else _as_records(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        if not node or not all((isinstance(v, str) for v in node)):
            return [_as_records(v, depth + 1) for v in node]
        records = [_row_from_text(v) for v in node]
        if 2 * sum((r is not None for r in records)) < len(node):
            return node
        return [v if r is None else r for v, r in zip(node, records)]
    return node


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

def _long_fields(record: dict[str, Any]) -> list[tuple[str, str]]:
    

    out: list[tuple[str, str]] = []

    def take(key: str, value: Any) -> None:
        if isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
            value = '; '.join(value)
        if isinstance(value, str) and len(value.strip()) > 40 and not re.match('\\s*(?:https?:|www\\.)', value):
            out.append((key, value.strip()))

    for key, value in record.items():
        if isinstance(value, dict):
            for inner, deep in value.items():
                take(str(inner), deep)
        else:
            take(str(key), value)
    return out


def _segments(text: str) -> list[str]:
    

    pieces = re.split('[\\n;|\\u2022]+|(?<=[a-z0-9)])\\.\\s+|,\\s+(?=[A-Za-z][\\w /-]{0,30}\\s*[:=])', text)
    out: list[str] = []
    for piece in pieces:
        piece = re.sub('\\s+', ' ', piece or '').strip(' .,').lower()
        if len(piece) >= 3 and piece not in out:
            out.append(piece[:140])
    return out


def _partial_first(names: list[str], share: dict[str, float]) -> list[str]:
    


    some = [n for n in names if 0.0 < share.get(n, 0.0) < 1.0 and (not re.search(CATCH_ALL, _words(n)))]
    return some + [n for n in names if n not in some]


def _text_names(record: dict[str, Any]) -> list[str]:
    


    names: list[str] = []
    for _, body in _long_fields(record):
        local: list[str] = []
        for piece in _segments(body):
            hit = re.match("([a-z][a-z0-9 '-]{0,39}?)\\s*[:=]\\s*(\\S.{0,59})$", piece)
            if hit and len(hit.group(1).split()) <= 4 and (not hit.group(2).startswith('//')) and _words(hit.group(1)) not in map(_words, local):
                local.append(hit.group(1).strip())
        if len(local) >= 2:
            names += [n for n in local if _words(n) not in map(_words, names)]
    return names


class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Turn:
    


    def __init__(self, call: dict[str, Any]):
        self.call = call
        self.cid = call['id']
        self.name: Any = None
        self.args: dict[str, Any] = {}
        self.note = ''
        self.many: str | None = None
        self.mark = ''
        self.ending = False
        self.readers: set[str] = set()


class GateRun:
    


    def __init__(self, stage: str, steps: list[dict[str, Any]], ending: set[str]):
        self.stage = stage
        self.steps = steps
        self.ending = ending
        self.said: list[str] = []
        self.nudged = False


def _tool_line(tool: dict[str, Any]) -> str:
    fn = tool['function']
    return f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}"


def _has_buyer(brief: dict[str, Any]) -> bool:
    roles = brief.get('roles')
    return bool(_items(brief.get('items'))) and isinstance(roles, dict) and bool(roles.get('buy'))


def _brief_item(entry: dict[str, Any]) -> Item:
    quantity = _count(entry.get('quantity'))
    return Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, [str(m) for m in _items(entry.get('musts')) if m])


def _clash(one: str, other: str) -> bool:
    key, _, value = one.partition(':')
    key2, _, value2 = other.partition(':')
    return _words(key) == _words(key2) and _words(value) != _words(value2)


def _same_line(kept: Item, item: Item) -> bool:
    if not item.shelf.strip() or _words(kept.shelf) != _words(item.shelf):
        return False
    return not any((_clash(a, b) for a in kept.musts for b in item.musts))


def _amount(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        return None
    return value


def _process_step(entry: dict[str, Any]) -> dict[str, Any]:
    when = str(entry.get('when') or '').strip().lower()
    return {'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)}


def _plan_pick(pick: Any, size: int) -> tuple[int, str, bool] | None:
    
    if not isinstance(pick, dict):
        return None
    label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
    if not label or len(label.split()) > 5:
        return None
    return (min(max(1, _count(pick.get('item') or 1) or 1), size), label, bool(str(pick.get('hint') or '').strip()))


def _plain(text: str) -> str:
    return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))


def _record_spelling(own: dict[str, float], label: str) -> str:
    return next((k for k in own if _plain(k) == _plain(label)), label)


def _share_of(own: dict[str, float], label: str) -> float:
    if _words(label) in own:
        return own[_words(label)]
    return max((s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))), default=0.0)


def _guard_place(lane: list[str], key: str, own: dict[str, float], hints: set[str], many: bool) -> list[str]:
    

    top = lane[:3]
    if _words(key) in map(_words, top):
        return lane
    need = own.get(key, 0.0)
    weak = next((i for i, other in enumerate(top) if _words(other) not in hints and _share_of(own, other) + GUARD_GAP <= need), None)
    if weak is None and len(lane) >= 3:
        return lane
    placed = [other for other in lane if _words(other) != _words(key)]
    if weak is not None and many:
        moved = placed[weak]
        placed[weak] = key
        placed.insert(min(3, len(placed)), moved)
    else:
        placed.insert(min(len(placed), len(placed) if weak is None else weak), key)
    front = set(map(_words, placed[:3]))
    return lane if any((_words(x) in hints and _words(x) not in front for x in top)) else placed


def _value_tries(values: list[str], fresh: list[str], lead: bool, name: str, named: bool) -> list[list[str]]:
    


    if len(values) > 1:
        stated = [i for i, v in enumerate(values) if v not in fresh]
        cut = stated[-1] if stated else 0
        tries = [values, values[:cut] + values[cut + 1:]]
    else:
        tries = [[values[0], name], values] if lead else [values]
    if named and name.strip() and not _roots(name) & _roots(' '.join(tries[0])):
        tries = [tries[0] + [name], tries[0]]
    return [t for t in tries if t][:2]


class Yama:

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
        self.stamped = False
        self.stamps = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
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
        self.answer_rounds = 0
        self.buy_urged = False
        self.end_tries = 0
        self.second_end = False
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
        self.found_for: dict[str, int] = {}
        self.src_item = 0
        self.capped: set[tuple[str, float]] = set()
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
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
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.dry_done = False
        self.refuse_rejects = 0
        self.reject_said = False
        self.put_told: tuple[int, ...] | None = None
        self.reput_said = False
        self.open_order: list[str] = []
        self.open_hits: dict[str, int] = {}
        self.open_last: dict[str, int] = {}
        self.list_hits: dict[str, int] = {}
        self.list_top: dict[str, int] = {}
        self.name_said = False
        self.reject_bought = False
        self.line_holds = 0
        self.gap_tried = False
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

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
        self.src_item = 0
        if SRCLINE and len(self.items) > 1 and name == self.roles.get('search'):
            said = ' '.join((v for v in args.values() if isinstance(v, str)))
            self.src_item = self.owner_of(said) if said.strip() else 0
        if self.stamped:
            self.stamp(payload)
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = self.open_wrap(result.data)
                if TEXTROW_ON and 'observation' in data:
                    before = _ROWS_READ[0]
                    data['observation'] = _as_records(data['observation'])
                    if _ROWS_READ[0] > before and (not getattr(self, 'rows_from_text', 0)):
                        self.trace.append({'kind': 'text_rows', 'rows': _ROWS_READ[0] - before, 'step': self.step + 1})
                    self.rows_from_text = getattr(self, 'rows_from_text', 0) + _ROWS_READ[0] - before
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                        self.open_order.extend((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip() not in self.open_order))
                        for v in value if isinstance(value, list) else [value]:
                            if isinstance(v, str) and v.strip():
                                self.open_hits[v.strip()] = self.open_hits.get(v.strip(), 0) + 1
                                self.open_last[v.strip()] = self.step
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
                    if self.stamped:
                        self.stamp(payload)
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    if self.stamped:
                        self.stamp(payload)
                    continue
                return None
            if status == 422:
                low = body.lower()
                if re.search('finished|terminal observation', low):
                    self.finished = True
                    return None
                if not self.stamped and re.search('call_id|idempotency|turn must', low):
                    self.stamped = True
                    self.trace.append({'kind': 'stamp_on', 'step': self.step})
                    self.stamp(payload)
                    continue
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def stamp(self, payload: dict[str, Any]) -> None:
        

        self.stamps += 1
        number = self.step + 1
        name = f'ya-{number}-{self.stamps}'
        payload['step'] = number
        payload['turn'] = number
        payload['call_id'] = name
        payload['idempotency_key'] = name

    @staticmethod
    def open_wrap(data: dict[str, Any]) -> dict[str, Any]:
        


        calls = data.get('calls')
        if not (isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict)):
            return data
        inner = calls[0].get('observation')
        if not isinstance(inner, dict) or 'observation' not in inner or set(inner) - {'done', 'error', 'observation'}:
            return data
        seen = inner['observation']
        fault = inner.get('error')
        if fault:
            seen = dict(seen, error=fault) if isinstance(seen, dict) and 'error' not in seen else {'observation': seen, 'error': fault}
        said = data.get('user_message')
        if isinstance(said, dict):
            said = said.get('content')
        if isinstance(said, str) and said.strip():
            seen = dict(seen, user_message=[said]) if isinstance(seen, dict) else {'observation': seen, 'user_message': [said]}
        out = dict(data)
        out['observation'] = seen
        if not isinstance(out.get('step'), int):
            for key in ('action_count', 'turn'):
                if isinstance(out.get(key), int):
                    out['step'] = out[key]
                    break
        if inner.get('done') is True or calls[0].get('done') is True:
            out['finished'] = True
        return out

    def find_ask(self) -> None:
        


        if not ASK_ROLE or self.roles.get('ask') or self.no_credit:
            return
        taken = set(self.roles.values())
        spare = [t for t in self.tools if t['function']['name'] not in taken]
        if not spare:
            return
        prompt = ('Below are store tools a shopping agent may call. Which one sends a message or a question to the shopper '
                  '(the person the agent shops for)? Return ONE JSON object and nothing else: {"ask": the exact tool name, '
                  'or null when none of them does}.\nTOOLS:\n' + '\n'.join(map(_tool_line, spare)))
        got = self.ask_json(prompt, [LIGHT], 300) or {}
        name = got.get('ask')
        if not isinstance(name, str) or name not in self.schemas or name in taken:
            return
        self.roles['ask'] = name
        if self.message_args('?') is None:
            del self.roles['ask']
            return
        self.trace.append({'kind': 'ask_role_found', 'spare': len(spare)})

    def refused_retry(self) -> str | None:
        



        if not REFUSE_REJECTED or not self.refuse_rejects or self.finished:
            return None
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        buy = self.roles.get('buy')
        if not holding or not buy or self.reject_bought:
            return None
        if not (self.reject_said or self.late_held or self.refuse_rejects >= 2 or self.left() <= 4):
            self.reject_said = True
            self.trace.append({'kind': 'refuse_rejected_hold', 'n': self.refuse_rejects, 'left': self.left()})
            return ('Not run (no step used). The store did not accept the last ending without a purchase (read its reply), '
                    'so sending it again only spends a step. The cart holds lines: check it out, after fixing anything '
                    'the reply names.\nCart now: ' + self.cart_text())
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return None
        if self.left() >= 2:
            self.gate('order')
        if self.finished or self.left() <= 0:
            return 'Not run: the episode is over.'
        self.reject_bought = True
        reply = self.act(buy, args)
        self.trace.append({'kind': 'refuse_rejected_buy', 'n': self.refuse_rejects, 'left': self.left(), 'finished': self.finished})
        if self.finished:
            return 'Yama checked out the cart, since the store had not accepted ending without a purchase. [episode finished]'
        if reply is None:
            return None
        self.buys += 1
        self.observe(buy, reply.get('observation'))
        return ('Not run (no step used). Yama tried to check out the cart, since the store had not accepted ending '
                'without a purchase; the checkout reply: ' + self.show(reply.get('observation')) + '\nFix what it names, '
                'then check out.\nCart now: ' + self.cart_text())

    def read_request(self) -> None:
        brief = self.first_brief(self.brief_prompt())
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.find_ask()
        self.find_end(brief)
        self.split_mode(brief)
        became = self.take_items(brief)
        self.take_fits(brief, became)
        self.take_lookups(brief)
        self.unmix()
        self.take_limits(brief)
        self.take_process(brief)

    def brief_prompt(self) -> str:
        parts = [BRIEF, '\nTOOLS:\n', '\n'.join(map(_tool_line, self.tools)), '\n\nREQUEST:\n', self.request]
        opening = self.view.get('opening')
        if opening is not None:
            parts += ['\n\nOPENING NOTE FROM THE STORE:\n', _dump(_clip(opening, 1500))]
        return ''.join(parts)

    def first_brief(self, prompt: str) -> dict[str, Any]:
        

        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if self.no_credit:
            return brief
        if not _has_buyer(brief):
            return self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        if self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _has_buyer(again):
                return again
        return brief

    def find_end(self, brief: dict[str, Any]) -> None:
        


        buy = self.roles.get('buy')
        if not FINDEND or not buy or self.roles.get('refuse') or self.no_credit:
            return
        prompt = ('A shopping agent finishes every task in one of two ways: it checks out, or it ends the task without '
                  'buying anything. Here the checkout tool is ' + buy + '. Which of the store tools below ends the task '
                  'without a purchase? It may be the checkout tool itself, switched by an argument. Return ONE JSON object '
                  'and nothing else: {"refuse": the exact tool name, or null when no tool does it, "refuse_args": the '
                  'fixed arguments that select ending without a purchase ({} when none), "buy_args": the fixed arguments '
                  'that select the checkout ({} when none)}.\nTOOLS:\n' + '\n'.join(map(_tool_line, self.tools)))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        name = got.get('refuse')
        others = {v for k, v in self.roles.items() if k != 'buy'}
        if not isinstance(name, str) or name not in self.schemas or (name != buy and name in others):
            self.trace.append({'kind': 'end_role', 'found': False})
            return
        self.roles['refuse'] = name
        brief['roles'] = {**(brief.get('roles') if isinstance(brief.get('roles'), dict) else {}), 'refuse': name}
        if isinstance(got.get('refuse_args'), dict):
            brief['refuse_args'] = got['refuse_args']
        if name == buy and isinstance(got.get('buy_args'), dict) and not brief.get('buy_args'):
            brief['buy_args'] = got['buy_args']
        self.trace.append({'kind': 'end_role', 'found': True, 'same': name == buy})

    def split_mode(self, brief: dict[str, Any]) -> None:
        
        brief.update({key: {} for key in ('buy_args', 'refuse_args') if not isinstance(brief.get(key), dict)})
        tool = self.roles.get('buy')
        if not tool or tool != self.roles.get('refuse'):
            return
        if brief['buy_args'] and brief['refuse_args'] and brief['buy_args'] != brief['refuse_args']:
            return
        ask = f'The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(tool)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
        fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
        on, off = (fixed.get('buy_args'), fixed.get('refuse_args'))
        if isinstance(on, dict) and isinstance(off, dict) and on != off:
            brief['buy_args'], brief['refuse_args'] = (on, off)

    def take_items(self, brief: dict[str, Any]) -> dict[int, int]:
        self.items += [_brief_item(e) for e in _items(brief.get('items')) if isinstance(e, dict)]
        return self.merge_twins()

    def merge_twins(self) -> dict[int, int]:
        
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if _same_line(m, item)), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
                continue
            became[old] = merged.index(twin) + 1
            twin.musts += [m for m in item.musts if m not in twin.musts]
            self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged or [Item('the requested product', '', 1, [])]
        return became

    def take_fits(self, brief: dict[str, Any], became: dict[int, int]) -> None:
        lines = range(1, len(self.items) + 1)
        links = [(became.get(_count(p.get('item')) or 0), became.get(_count(p.get('with')) or 0)) for p in _items(brief.get('fits')) if isinstance(p, dict)]
        for link in links:
            if link[0] in lines and link[1] in lines and link[0] != link[1] and link not in self.bound:
                self.bound.append(link)

    def take_lookups(self, brief: dict[str, Any]) -> None:
        for lookup in _items(brief.get('lookups')):
            text = self.must_named(str(lookup).strip())
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))

    def must_named(self, text: str) -> str:
        
        musts = [m for item in self.items for m in item.musts]
        if text in musts:
            return text
        key = _words(text.partition(':')[0])
        return next((m for m in musts if _words(m.partition(':')[0]) == key), text)

    def take_limits(self, brief: dict[str, Any]) -> None:
        budget = _amount(brief.get('budget'))
        if budget is not None:
            self.budget = budget
        limit = _count(brief.get('messages'))
        known = limit is not None and limit >= 0
        if known or self.roles.get('ask'):
            self.msg_limit = limit if known else 3
        steps = _count(brief.get('steps'))
        if steps in range(10, self.max_steps):
            self.max_steps = steps

    def take_process(self, brief: dict[str, Any]) -> None:
        self.process += [_process_step(e) for e in _items(brief.get('process')) if isinstance(e, dict) and str(e.get('step') or '').strip()]
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk, default=0), self.msg_limit)

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
            when = {'choose': 'before choosing', 'order': 'at the order (Yama does it when you check out)', 'end': 'at an ending without a purchase (Yama does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Yama] step {self.step} of {self.max_steps} used, {left} left.'
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
                        if getattr(self, 'src_item', 0):
                            self.found_for.setdefault(ident, self.src_item)
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
                if rows and ROWCUE:
                    self.note_rows(node, depth + 1)
        walk(observation, 0)

    def note_rows(self, node: list[Any], depth: int) -> None:
        

        places: dict[str, int] = {}
        for place, entry in enumerate((v for v in node if isinstance(v, dict))):
            ident = self.ident(entry, True, depth)
            if ident and ident not in places:
                places[ident] = place
        for ident, place in places.items():
            self.list_hits[ident] = self.list_hits.get(ident, 0) + 1
            if place < self.list_top.get(ident, place + 1):
                self.list_top[ident] = place

    def row_cue(self, ident: str) -> tuple[int, int]:
        


        if not ROWCUE:
            return (0, 0)
        return (-self.list_hits.get(ident, 0), self.list_top.get(ident, 1 << 20))

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
            if SRCLINE and len(self.items) > 1 and self.found_for.get(ident) and (not self.owner_of(label)):
                label = self.items[self.found_for[ident] - 1].name
                self.src_grouped = getattr(self, 'src_grouped', 0) + 1
            group = groups.setdefault(label, {'n': 0, 'f': 0, 'keys': {}})
            group['n'] += 1
            group['f'] += 1 if block else 0
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        self.group_facts = {label: group['f'] for label, group in groups.items()}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
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
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            if PARTIALFIRST and group['f'] >= 2:
                first = _partial_first([name for name, _ in kept], {name: slot['n'] / group['f'] for name, slot in kept})
                kept.sort(key=lambda kv: first.index(kv[0]))
            for name, slot in kept[:14]:
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
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
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
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
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
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
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

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
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
            return f"\n[Yama] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
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
        rest = f"\n[Yama] {count} listings matched and Yama has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Yama read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
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
        return f'\n[Yama] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Yama] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
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
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        

        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())):
                continue
            if self.left() <= keep or self.finished:
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
        

        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and re.search(CAP_ARG, _snake(k), re.I) and not re.search(CAP_NOT, k, re.I)]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(SHELF_ARG, _snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and not any(k != shelves[0] and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

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
            if self.left() <= self.tail() + 5 or self.finished:
                return
            self.capped.add((line, price))
            faults: list[str] = []
            args = _shape({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.paging(tool, reply.get('observation'))
            if page and page[3] is not None and page[1] < page[0] <= 2 * page[1] and self.left() > self.tail() + 5 and not self.price_sorted(reply.get('observation')):
                call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                more = None if faults else self.act(tool, call)
                if more is not None:
                    self.observe(tool, more.get('observation'))

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
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)

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
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
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
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
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
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Yama's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        






        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
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
        summary = f'Yama asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out. Put the line of that item into the cart again after this answer even when it already fits (remove it and add it back, or set its quantity again), so the cart is filled after the shopper\'s latest answer.'
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
        


        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool or not self.searched:
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
        last = self.searched[-1]
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
        if not plan or self.left() < VALUE_FLOOR:
            return ''
        many = VALUE_PER_ITEM and len(numbers) > 1
        seen: list[str] = []
        pool = VALUE_SHARED
        for place, number in enumerate(numbers):
            if self.finished:
                break
            if many and place and self.left() < VALUE_NEXT + self.tail():
                self.trace.append({'kind': 'value_search_stop', 'item': number, 'left': self.left()})
                break
            pool -= self.search_item_values(plan, number, VALUE_EACH if many else pool, many, seen)
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Yama searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def search_item_values(self, plan: tuple[str, dict[str, Any], str], number: int, steps: int, named: bool, seen: list[str]) -> int:
        

        pairs = self.value_words(number)
        fresh = [v for v, told in pairs if told]
        if not fresh or steps <= 2:
            return 0
        tool, base, field = plan
        lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
        values = lead + [v for v, _ in pairs if v not in lead]
        used = 0
        for words in _value_tries(values, fresh, bool(lead), self.items[number - 1].name, named):
            if used >= steps:
                break
            faults: list[str] = []
            args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished:
                break
            reply = self.act(tool, args)
            used += 1
            if reply is None:
                break
            observation = reply.get('observation')
            self.observe(tool, observation)
            rows = self.value_rows(number, observation)
            seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
            if not rows:
                continue
            opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, steps - used)], self.tail() + 16)
            used += len(opened)
            if opened:
                seen.append('opened: ' + ' | '.join(opened)[:2400])
            break
        return used

    def value_rows(self, number: int, observation: Any) -> list[str]:
        

        item = self.items[number - 1]
        cat_key = self.category_key()
        found = [i for i, _ in self.records_in(observation)]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        if cap is not None and found and all(self.price_of(i) is not None and self.price_of(i) > 10 * cap for i in found):
            cap = None
        label = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '')
        rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, label(i))) and (cap is None or (self.price_of(i) or 0) <= cap)]
        home = self.home_label(number)
        return sorted(rows, key=lambda i: _words(label(i)) != home) if cat_key and home else rows

    def home_label(self, number: int) -> str:
        


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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:LIST_SIZE if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        


        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def refill_dry(self) -> bool:
        




        if self.dry_done or self.msgs_free() <= 0 or not self.lanes:
            return False
        self.dry_done = True
        added = 0
        for number in sorted(self.lanes):
            have = {_words(x) for x in self.lanes[number]} | self.lane_used.get(number, set())
            fresh = [k for k in self.dry_row_keys(number) if _words(k) not in have and not self.told_about(number, k)]
            self.lanes[number] = self.lanes[number] + fresh
            added += len(fresh)
        source = 'rows'
        if not added and self.clock() < QUESTION_BY - 20 and self.judges():
            source = 'plan'
            asked = sorted({x for used in self.lane_used.values() for x in used})
            prompt = self.plan_prompt(LIST_SIZE, self.key_table()) + '\n\nALREADY ASKED (name other properties than these):\n' + ('\n'.join(asked) or '(none)')
            picks, _ = self.plan_picks(self.ask_json(prompt, [HEAVY, LIGHT], 2500) or {})
            for number, labels in picks.items():
                if number not in self.lanes:
                    continue
                have = {_words(x) for x in self.lanes[number]} | self.lane_used.get(number, set())
                fresh = [p for p in labels if _words(p) not in have and not self.told_about(number, p)]
                self.lanes[number] = self.lanes[number] + fresh
                added += len(fresh)
        self.trace.append({'kind': 'dry_lane', 'source': source, 'added': added})
        return added > 0

    def dry_row_keys(self, number: int) -> list[str]:
        



        cat_key = self.category_key()
        item = self.items[number - 1]
        rows = [r for r in self.rows.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        if not rows and len(self.items) == 1:
            rows = list(self.rows.values())
        if not rows and SRCLINE:
            rows = [r for i, r in self.rows.items() if self.found_for.get(i) == number]
        if not rows and ALLREC and len(self.items) > 1:
            rows = list(self.rows.values())
        skip = {cat_key, self.price_key, self.stock_key}
        seen: dict[str, list[str]] = {}
        for row in rows:
            for key, value in row.items():
                if key in skip or value is None or isinstance(value, (dict, list, bool)):
                    continue
                text = str(value).strip().lower()
                if text:
                    seen.setdefault(key, []).append(text)
        ranked = []
        for key, vals in seen.items():
            name = _words(key)
            if not name or re.search(NOT_PROPERTY, name):
                continue
            mean = sum(len(v) for v in vals) / len(vals)
            if mean > 40 or sum(1 for v in vals if re.fullmatch('[\\d.,$%/x -]+', v) or _id_like(v)) * 2 >= len(vals):
                continue
            if len(vals) >= 4 and len(set(vals)) == len(vals) and mean > 12:
                continue
            ranked.append((bool(re.search(CATCH_ALL, name)), len(set(vals)) < 2, -len(vals), name))
        return [name for *_, name in sorted(ranked)]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        


        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        


        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                if not labels and DRY_LANE and self.refill_dry():
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
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
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
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        










        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if new or not crowded:
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                self.follow_up.insert(0, (number, labels, True))
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
        self.lanes[number] = rest + ahead if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        


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
        if ALLREC and len(self.items) > 1 and (not rows) and table:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
            self.allrec_lines = getattr(self, 'allrec_lines', set()) | {number}
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        if PARTIALFIRST:
            facts = getattr(self, 'group_facts', {})
            rows.sort(key=lambda r: not facts.get(r[0], 0) >= 2 or not 0 < r[2] < facts.get(r[0], 0))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
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
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        count = sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))
        if SRCLINE and len(self.items) > 1:
            count += sum((1 for ident, record in self.full.items() if self.found_for.get(ident) == number and _fact_block(record) and (not (cat_key and self.owner_of(str(record.get(cat_key) or ''))))))
        return count

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        plan = self.ask_json(self.plan_prompt(count, table), [HEAVY, LIGHT], 3500) or {}
        lanes, hinted = self.plan_picks(plan)
        for number in lanes:
            lanes[number] = self.fill_lane(number, table, lanes[number], hinted[number])
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def long_lines(self) -> list[str]:
        


        if not LONGTEXT:
            return []
        cat_key = self.category_key()
        said = set(re.findall('[a-z]{3,}', self.request.lower()))
        groups: dict[str, dict[str, dict[str, list[str]]]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not isinstance(record, dict):
                continue
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            for key, value in _long_fields(record):
                groups.setdefault(label, {}).setdefault(key, {}).setdefault(ident, _segments(value))
        owned = {label for label in groups if self.owner_of(label)}
        if owned:
            groups = {label: keys for label, keys in groups.items() if label in owned}
        out: list[str] = []
        for label, keys in groups.items():
            ranked = []
            for key, per in keys.items():
                name = _words(key)
                if not name or re.search(NOT_PROPERTY, name):
                    continue
                parts = list(per.values())
                common = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set()
                kept: list[str] = []
                for segs in parts:
                    for seg in segs:
                        words = set(re.findall('[a-z]{3,}', seg))
                        if seg in common or seg in kept or not words or words <= said:
                            continue
                        kept.append(seg)
                if kept:
                    ranked.append((-len(kept), name, kept))
            ranked.sort(key=lambda r: r[0])
            for _, name, kept in ranked[:LONG_KEYS]:
                out.append(f'category {label!r} / {name}: ' + ' | '.join(kept)[:LONG_CLIP])
        out = out[:LONG_LINES]
        self.trace.append({'kind': 'long_text', 'lines': len(out), 'chars': sum(map(len, out))})
        return out

    def plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        settled = [f'item {n}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(self.items, 1)]
        listing = self.key_listing(table)
        parts = [PLAN.replace('COUNT', str(max(1, count))), '\n\nREQUEST:\n', self.request, '\n\nITEMS:\n', '\n'.join(settled)]
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            parts += ['\n\nASK FIRST (the request asks to be asked about these):\n', '\n'.join(first)]
        parts += ['\n\nATTRIBUTE KEYS SEEN:\n', '\n'.join(listing) or '(no full listing records yet)']
        long = self.long_lines()
        if long:
            parts += ['\n\nLONG TEXT ON THE OPENED PAGES (only the parts that differ between the listings and use words the '
                      'request does not): a firm unwritten preference often sits on a property that only this text names, '
                      'as the name or the value of a "name: value" or "name=value" pair, or as a feature the listings word '
                      'differently.\n', '\n'.join(long),
                      '\nWhen such a property is left open by the request, rank it with the listed keys of that item, before '
                      'keys whose values barely differ; its label is the short plain name the text gives the property (or '
                      'the feature itself when the text gives no name), never a whole phrase or a value list.']
        return ''.join(parts)

    def key_listing(self, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        
        out: list[str] = []
        for index, (label, name, carried, values) in enumerate(table, 1):
            before = table[index - 2][0] if index > 1 else None
            if label != before:
                out.append(f'category {label!r}:')
            mixed = ' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''
            out.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + mixed)
        return out

    def plan_picks(self, plan: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        
        size = len(self.items)
        lanes: dict[int, list[str]] = {n: [] for n in range(1, size + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for number, label, hint in filter(None, (_plan_pick(p, size) for p in _items(plan.get('ask')))):
            if _words(label) not in map(_words, lanes[number]):
                lanes[number].append(label)
            if not hint:
                continue
            hinted[number].add(_words(label))
            if number not in self.plan_order:
                self.plan_order.append(number)
        return lanes, hinted

    def fill_lane(self, number: int, table: list[tuple[str, str, int, list[str]]], picks: list[str], hints: set[str]) -> list[str]:
        

        keys = self.lane_keys(number, table)
        read = max(1, self.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if self.owner_of(label) == number:
                own[key] = max(own.get(key, 0.0), carried / read)
        hints = {_words(_record_spelling(own, h)) for h in hints}
        lane: list[str] = []
        for pick in picks:
            word = _record_spelling(own, pick)
            if _words(word) not in map(_words, lane):
                lane.append(word)
        many = len(self.items) > 1
        for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and not re.search(CATCH_ALL, k)][:GUARD_KEYS]:
            lane = _guard_place(lane, key, own, hints, many)
        for key in keys:
            if _words(key) not in map(_words, lane):
                lane.append(key)
        extra = 0
        if LONGREAD:
            cat_key = self.category_key()
            item = self.items[number - 1]
            mine = [r for r in self.full.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
            if not mine and len(self.items) == 1:
                mine = list(self.full.values())
            if not mine and SRCLINE:
                mine = [r for i, r in self.full.items() if self.found_for.get(i) == number]
            if not mine and ALLREC and len(self.items) > 1:
                mine = list(self.full.values())
            heard: dict[str, list[str]] = {}
            holders = 0
            for record in mine:
                found = _text_names(record) if isinstance(record, dict) else []
                holders += bool(found or _fact_block(record))
                for name in found:
                    heard.setdefault(_words(name), []).append(name)
            if holders >= 2:
                for word, seen in sorted(heard.items(), key=lambda kv: -len(kv[1])):
                    if word and word not in map(_words, lane):
                        lane.append(seen[0])
                        own[seen[0]] = len(seen) / holders
                        extra += 1
                read = max(read, holders)
        if SRCLINE and getattr(self, 'src_grouped', 0) and (not getattr(self, 'srcline_told', False)):
            self.srcline_told = True
            self.trace.append({'kind': 'lever_fire', 'why': 'srcline', 'records': self.src_grouped, 'found': len(self.found_for), 'step': self.step})
        if ALLREC and getattr(self, 'allrec_lines', None) and (not getattr(self, 'allrec_told', False)):
            self.allrec_told = True
            self.trace.append({'kind': 'lever_fire', 'why': 'allrec', 'lines': len(self.allrec_lines), 'step': self.step})
        if PARTIALFIRST and read >= 2:
            lane = _partial_first(lane, own)
            self.trace.append({'kind': 'partial_first', 'item': number, 'partial': sum((1 for x in lane[:8] if 0.0 < own.get(x, 0.0) < 1.0)), 'lane': len(lane), 'read': read, 'long': extra})
        return sorted(lane, key=lambda x: bool(re.search(CATCH_ALL, _words(x))))

    def name_parts(self, number: int) -> list[str]:
        


        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        




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
        known_before = sum((len(item.told) for item in self.items))
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
        if sum((len(item.told) for item in self.items)) > known_before:
            self.answer_rounds += 1

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return (True, '')
        run = GateRun(stage, steps, {self.roles.get('buy'), self.roles.get('refuse')} - {None})
        for _ in range(GATE_ROUNDS):
            got = self.ask_json(self.gate_prompt(run), [HEAVY, LIGHT] if run.nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            step = self.gate_round(run, got)
            if isinstance(step, tuple):
                return step
            if not step:
                break
        return (True, 'Before going on, Yama did the steps the request asks for:\n' + '\n'.join(run.said) if run.said else '')

    def gate_prompt(self, run: GateRun) -> str:
        tools = [_tool_line(t) for t in self.tools if t['function']['name'] not in run.ending]
        record = '\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)'
        notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else ''
        what = 'order' if run.stage == 'order' else 'ending'
        parts = [GATE.replace('STAGE', 'place the order' if run.stage == 'order' else 'end without a purchase'),
                 '\n\nREQUEST:\n', self.request, '\n\nSTEPS DUE NOW:\n', '\n'.join((f"- {p['step']}" for p in run.steps)),
                 '\n\nLEDGER:\n', self.ledger(), '\n\nCART NOW: ', self.cart_text(),
                 '\n\nRECORD (messages and step calls so far):\n', record, notes,
                 f'\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {what} itself takes one)',
                 '\n\nTOOLS:\n', '\n'.join(tools)]
        return ''.join(parts)

    def gate_round(self, run: GateRun, got: dict[str, Any]) -> tuple[bool, str] | bool:
        

        told: list[str] = []
        hold = ''
        if run.said:
            self.take_heard({'needs': got.get('needs')}, told)
            hold = str(got.get('hold') or '').strip()
        call = got.get('call') if isinstance(got.get('call'), dict) else None
        self.trace.append({'kind': 'gate', 'stage': run.stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
        if told or hold:
            self.version += 1
            done = 'Before going on, Yama did the steps the request asks for:\n' + '\n'.join(run.said) + '\n' if run.said else ''
            return (False, done + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.')
        if not call:
            if self.gate_nudge(run, got):
                return True
            self.gate_ok.add((run.stage, self.cart_mark() + f'#{self.version}'))
            return False
        name = str(call.get('tool') or '')
        if name not in self.schemas or name in run.ending or self.left() < 2:
            return False
        if name == self.roles.get('ask') and self.msg_used >= self.msg_limit:
            return False
        return self.gate_call(run, name, call)

    def gate_nudge(self, run: GateRun, got: dict[str, Any]) -> bool:
        

        if run.said or run.nudged or not self.roles.get('ask') or self.msg_used >= self.msg_limit or self.left() < 2:
            return False
        talk = run.stage == 'order' and any((p['messages'] > 0 for p in run.steps))
        proof = {i for i in range(1, len(self.done_log) + 1) if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
        unproven = [e for e in _items(got.get('steps')) if isinstance(e, dict) and (e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines')))))]
        if not unproven:
            return False
        run.nudged = True
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        self.done_log.append(f'(Yama: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)
        return True

    def gate_call(self, run: GateRun, name: str, call: dict[str, Any]) -> bool:
        problems: list[str] = []
        args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', problems)
        if problems:
            return False
        if name == self.roles.get('ask'):
            text = next((v for v in args.values() if isinstance(v, str)), '')
            batch = self.send([(1, '', text)], 1)
            if not batch:
                return False
            self.gate_msgs += 1
            run.said.append(f'Q: {text}\nA: {batch[0][2]}')
            return True
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb(seen)
        self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        run.said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return True

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
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
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
        return "lines by Yama's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def line_ids(self, number: int) -> list[str]:
        

        out = self.item_ids(number)
        if out or (not SRCLINE):
            return out
        if len(self.items) == 1:
            return list(self.seen_order)
        return [i for i in self.seen_order if self.found_for.get(i) == number]

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

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
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.fit_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
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
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def missing_items(self) -> list[int]:
        
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        

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
        

        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            pick = self.stating_pick(number, bound)
            if not pick or not self.cart_line(cart, number, pick):
                return False
        return True

    def stating_pick(self, number: int, bound: set[int]) -> str | None:
        


        if number in bound:
            return None
        item = self.items[number - 1]
        fits = sorted((i for i in self.line_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)), key=lambda i: self.price_of(i) or 0.0)
        if not fits:
            return None
        if STATERANK and self.brief.get('pick') != 'cheapest':
            fits = self.seen_first(fits, number)
        if isinstance(self.budget, (int, float)):
            each = self.brief.get('budget_scope') == 'each'
            held = 0.0 if each else sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
            if (self.price_of(fits[0]) or 0.0) * (1 if each else item.qty) + held > self.budget + 1e-09:
                return None
        return fits[0]

    def seen_first(self, ids: list[str], number: int) -> list[str]:
        

        qty = self.items[number - 1].qty
        each = self.brief.get('budget_scope') == 'each'
        held = 0.0 if each else sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
        cap = self.budget if isinstance(self.budget, (int, float)) else None

        def order(ident: str) -> tuple[bool, int, int, int, int, float]:
            price = self.price_of(ident) or 0.0
            over = cap is not None and price * (1 if each else qty) + held > cap + 1e-09
            cue = self.row_cue(ident)
            return (over, -self.open_hits.get(ident, 0), -self.open_last.get(ident, -1), cue[0], cue[1], price)
        return sorted(ids, key=order)

    def cart_line(self, cart: str, number: int, pick: str) -> bool:
        
        held = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}
        calls = self.cart_calls({**held, pick: self.items[number - 1].qty}, [])
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

    def fill_empty_cart(self) -> bool:
        



        if not EMPTY_FILL or not self.roles.get('cart') or not (any((item.told for item in self.items)) or (BUY_UNTIL_TOLD and self.answer_rounds < BUY_UNTIL_TOLD)):
            return False
        if self.left() < len(self.items) + 3:
            return False
        bound = {n for pair in self.bound for n in pair}
        wanted = self.missing_items()
        if len(wanted) != len(self.items) or not all((self.stating_pick(n, bound) for n in wanted)):
            self.trace.append({'kind': 'empty_fill', 'done': False, 'items': len(wanted)})
            return False
        done = self.fill_short_cart()
        self.trace.append({'kind': 'empty_fill', 'done': done, 'lines': self.cart_ids()})
        return done and bool(self.cart_ids())

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

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
                fuller = self.fuller_label(n, label[i])
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
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
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
                if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
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
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Yama opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
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
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if not faults:
                self.link_searches += 1
                reply = self.act(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                    known = self.shop_rows(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
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
        text = f"[Yama] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        for ident in reversed(self.seen_order):
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
        if self.left() < len(plan) + 5:
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
        if (calls is None if CARTREADY else not calls) or len(calls) + 1 > self.left():
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
        return 'Not run (Yama acted instead): the independent check confirmed ' + self.confirmed + ', so Yama set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        tied = self.id_cols(name) if IDMATCH else []
        if tied:
            out = self.match_ids(out, tied)
        if DECLNAME and self.brief.get('quiet_decline') is not True:
            return self.cite_args(name, out)
        for key, spec in props.items():
            if key in tied or not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
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
        if REPUT and (not removal):
            self.put_told = self.told_counts()
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
        text = _dump(_clip(observation, SHOW_CHARS))
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
        batch = self.join_opens(calls)
        answers: list[tuple[str, str]] = []
        for call in batch:
            turn = Turn(call)
            said = self.admit(turn) or self.screen(turn, len(batch)) or self.end_screen(turn)
            if said:
                answers.append((turn.cid, said))
                continue
            refusing = REFUSE_REJECTED and self.is_refuse(turn.name, turn.args)
            reply = self.carry_out(turn)
            if refusing and (not self.finished) and (not reply.startswith('The call failed')):
                self.refuse_rejects += 1
                self.trace.append({'kind': 'refuse_rejected', 'n': self.refuse_rejects, 'left': self.left()})
            answers.append((turn.cid, reply))
        return answers

    def admit(self, turn: Turn) -> str | None:
        

        allowed = {t['function']['name'] for t in self.offered()}
        if turn.call.get('joined'):
            return 'Shown in the side-by-side result above: Yama opened these listings together in one step.'
        fn = turn.call.get('function') or {}
        name = turn.name = fn.get('name')
        raw = fn.get('arguments')
        if self.finished:
            return 'Not run: the episode is over.'
        if name == self.roles.get('ask'):
            return "Not run (no step used): Yama's interviewer handles every message to the shopper."
        if name not in self.schemas:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."
        if name not in allowed:
            return 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'
        args = raw if isinstance(raw, dict) else _json_obj(raw) if (raw or '').strip() else {}
        if args is None:
            return 'Not run (no step used): the arguments were not one JSON object.'
        problems: list[str] = []
        args = _shape(args, self.schemas[name], 'args', problems)
        if problems:
            return 'Not run (no step used): ' + '; '.join(problems[:6])
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return 'Not run (no step used): ' + wrong + '.'
        turn.args = self.whole_removal(name, args) if name == self.roles.get('cart') else args
        return None

    def screen(self, turn: Turn, count: int) -> str | None:
        


        name, args = turn.name, turn.args
        many = turn.many = self.compare_target() if name == self.roles.get('compare') else None
        listed = args.get(many) if many else None
        if isinstance(listed, list) and len(listed) > self.compare_room() and self.compare_bad:
            room = self.compare_room()
            turn.note = f"\n[Yama] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, listed[room:]))}."
            args = turn.args = {**args, many: listed[:room]}
        target = self.single_target(name)
        picked = str(args.get(target)) if target else ''
        if target and name in (self.roles.get('open'), self.roles.get('offer')) and picked in self.gone:
            return f'Not run (no step used): {args[target]} sold out at step {self.gone[picked]} and will not come back. Replace its cart line with another fitting listing.'
        if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
            absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
            if absent:
                return f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()
        ending = turn.ending = self.is_buy(name, args) or self.is_refuse(name, args)
        if ending and count > 1:
            return 'Not run (no step used): send the finishing call on its own, after reading your other results.'
        if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
            return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
        mark = turn.mark = name + _dump(args)
        if not ending and name != self.roles.get('cart') and mark in self.recent:
            return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'
        readers = turn.readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        if name not in readers:
            return None
        cached = self.read_cache.get(mark)
        if cached and cached[0] == self.version:
            return f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]
        if target and name != self.roles.get('compare') and self.shown_already(name, picked):
            return f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(picked)
        return self.trim_shown(turn)

    def trim_shown(self, turn: Turn) -> str | None:
        

        listed = turn.args.get(turn.many) if turn.many else None
        if not isinstance(listed, list) or not listed:
            return None
        keep = [i for i in listed if not self.shown_already(turn.name, str(i))]
        if not keep:
            return 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in listed))[:4000]
        if len(keep) < len(listed):
            turn.note += f"\n[Yama] Left out {', '.join((str(i) for i in listed if i not in keep))}: shown in full already, nothing changed since."
            turn.args = {**turn.args, turn.many: keep}
            turn.mark = turn.name + _dump(turn.args)
        return None

    def told_counts(self) -> tuple[int, ...]:
        
        return tuple(len(item.told) for item in self.items)

    def reput_hold(self) -> str | None:
        



        if not REPUT or self.reput_said or self.put_told is None or self.finished or self.left() < 4:
            return None
        now = self.told_counts()
        grew = [n for n, (a, b) in enumerate(zip(now, self.put_told), 1) if a > b]
        if not grew:
            return None
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        if not (self.cart_ids() if fresh else self.cart_lines):
            return None
        self.reput_said = True
        self.trace.append({'kind': 'reput_hold', 'items': len(grew), 'left': self.left()})
        names = '; '.join(self.items[n - 1].name for n in grew)
        return ('Not run (no step used). The shopper disclosed a requirement after the cart was last filled (' + names +
                '). Check that line against it, then put the line that meets it into the cart again now, even when the '
                'one in the cart already fits (remove it and add it back, or set its quantity again; keep the quantity '
                'the request asks for), so the cart is filled after the shopper\'s latest answer. Then check out.')

    def end_screen(self, turn: Turn) -> str | None:
        

        if SECONDEND and self.is_refuse(turn.name, turn.args):
            self.end_tries += 1
        if turn.ending and self.left() > 2:
            if self.is_buy(turn.name, turn.args):
                held = self.buy_hold()
            else:
                held = self.refuse_hold() if self.left() > 5 else None
            if held:
                return held
        if self.is_refuse(turn.name, turn.args):
            held = self.decline_hold(turn)
            if held:
                return held
        buying = self.is_buy(turn.name, turn.args)
        if buying:
            late = self.late_question()
            if late:
                return 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()
            again = self.reput_hold()
            if again:
                return again + '\nCart now: ' + self.cart_text()
        if turn.ending and self.left() >= 2:
            go, said = self.gate('order' if self.is_buy(turn.name, turn.args) else 'end')
            if not go:
                return 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
            if said:
                turn.note = '\n[Yama] ' + said + turn.note
        return None

    def buy_hold(self) -> str | None:
        


        cart = lambda: '\nCart now: ' + self.cart_text()
        wider = self.wider_hold()
        if wider:
            return wider + cart()
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        viewer = self.roles.get('cart_view')
        if self.cart_touched and self.cart_seen_at < self.cart_at and viewer and self.left() > 3:
            problems: list[str] = []
            view_args = _shape({}, self.schemas.get(viewer), 'args', problems)
            view = None if problems else self.act(viewer, view_args)
            if view is not None:
                self.cart_view = view.get('observation')
                self.cart_seen_at = self.step
                change = self.observe(viewer, self.cart_view)
                if change:
                    self.queued = []
                    return 'Not run yet (no step used). Looking at the cart showed a change: ' + change + cart() + '\nApply the change, fix the cart if needed, then check out.'
        swap = self.swap_hint()
        gone = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
        if swap or gone:
            if not swap:
                self.held += 1
            return 'Not run yet (no step used). ' + (swap or f"{', '.join(gone)} in the cart is sold out: replace that line first.") + cart()
        chain, hold = self.trail()
        if hold:
            return 'Not run yet (no step used). ' + chain + cart()
        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
        if misfit:
            self.fit_holds += 1
            self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
            return 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.' + cart()
        problems = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
        if problems and all((' is held at ' in f for f in problems)):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                problems = self.line_faults()
                if not problems:
                    self.line_holds += 1
                    return "Not run yet (no step used). Yama set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.' + cart()
        if problems:
            self.line_holds += 1
            self.trace.append({'kind': 'line_hold', 'faults': problems[:6]})
            return 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(problems[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out." + cart()
        self.gap_fill()
        lines = self.cart_ids()
        read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
        self.show_links(lines, self.tail() + 1)
        read += self.cheaper_rows()
        verdict = self.check_cart()
        if verdict:
            seen = "Yama read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
            return 'Not run yet. ' + chain + seen + verdict + cart()
        if reread and self.cost > SPEND_SOFT:
            return "Not run yet (no step used). Yama re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return None

    def refuse_hold(self) -> str | None:
        

        review = self.audit()
        if review:
            return 'Not run yet (no step used). ' + review
        if not self.leads or self.left() <= 4 or self.refuse_blocks > 1:
            return None
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not (self.verdicts.get(self.check_key([first])) and first in self.leads):
            return None
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."

    def decline_hold(self, turn: Turn) -> str | None:
        


        rejected = self.refused_retry()
        if rejected:
            return rejected
        if self.left() <= 5 and not self.late_held:
            lines = [i for i in self.built if not self.sold_out(i)]
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
            if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                self.late_held = True
                self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                return 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'
        if not self.leads and not self.pushed and self.left() > self.max_steps // 3:
            self.pushed = True
            return f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        urge = self.urge_buy()
        if urge:
            return urge
        done = self.complete_cart()
        if done:
            return done + (' [episode finished]' if self.finished else '')
        named = self.name_hold(turn)
        if named:
            return named
        if SECONDEND and self.cart_instead(turn):
            return None
        turn.args = self.quiet(turn.name, turn.args)
        return None

    def free_fields(self, name: str) -> list[str]:
        
        schema = self.schemas.get(name) or {}
        need = schema.get('required') or []
        keys = [k for k, spec in (schema.get('properties') or {}).items() if isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec]
        if IDMATCH:
            tied = set(self.id_cols(name))
            keys = [k for k in keys if k not in tied]
        return sorted(keys, key=lambda k: k not in need)

    def decline_ids(self) -> list[str]:
        

        out: list[str] = []
        for ident in list(self.leads) + list(self.cart_ids()) + list(reversed(self.open_order)):
            if ident in self.rows and ident not in out:
                out.append(ident)
        return out[:3]

    def decline_text(self) -> str:
        
        ids = self.decline_ids()
        musts = '; '.join((str(m) for item in self.items for m in item.musts if str(m).strip()))
        need = ' (' + musts[:200] + ')' if musts else ''
        if not ids:
            return 'Nothing in the store meets every requirement' + need + ', so I am not buying anything.'
        return 'The closest listings I checked are ' + ', '.join(ids) + '; none of them meets every requirement' + need + ', so I am not buying anything.'

    def cite_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        

        free = self.free_fields(name)
        if not free or any((str(args.get(k) or '').strip() for k in free)):
            return args
        out = dict(args)
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(free[0]) or {}
        out[free[0]] = self.decline_text()[:int(spec.get('maxLength') or 400)]
        return out

    def name_hold(self, turn: Turn) -> str | None:
        



        if not DECLNAME or self.name_said or self.brief.get('quiet_decline') is True or self.left() <= 2:
            return None
        free = self.free_fields(turn.name)
        ids = self.decline_ids()
        if not free or not ids:
            return None
        said = ' '.join((str(turn.args.get(k) or '') for k in free))
        if any((ident in said for ident in self.rows)):
            return None
        self.name_said = True
        self.trace.append({'kind': 'name_hold', 'ids': len(ids), 'left': self.left()})
        return ('Not run (no step used). An ending without a purchase is accepted only when its message names the '
                'listings you checked that come closest and says which requirement each one misses. Send the ending '
                'again with that message in ' + free[0] + ': the ids of up to three of the closest listings you opened '
                '(such as ' + ', '.join(ids) + ') and, for each, the requirement it does not meet.')

    def id_cols(self, name: str) -> list[str]:
        

        props = (self.schemas.get(name) or {}).get('properties') or {}
        found: list[str] = []
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            values = [row[key] for row in self.rows.values() if key in row]
            if not values or len({str(v) for v in values}) < len(values):
                continue
            if not all((_id_like(v, True) for v in values)):
                continue
            if len(values) == 1 and not _id_like(values[0]):
                continue
            found.append(key)
        return found

    def match_ids(self, args: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        

        pick: dict[str, Any] | None = None
        for key in keys:
            given = args.get(key)
            if not (isinstance(given, str) and given.strip()):
                continue
            want = given.strip()
            pick = self.rows.get(want) or next((row for row in self.rows.values() if row.get(key) == want), None)
            if pick:
                break
        if pick is None:
            pool = self.decline_ids() or [i for i in reversed(self.seen_order) if i in self.rows][:1]
            ready = [i for i in pool if not self.sold_out(i)] or pool
            pick = self.rows.get(ready[0]) if ready else None
        if not pick:
            return args
        out = dict(args)
        moved = 0
        for key in keys:
            if key in pick and out.get(key) != pick[key]:
                out[key] = pick[key]
                moved += 1
        if moved:
            self.trace.append({'kind': 'id_match', 'moved': moved, 'keys': len(keys), 'step': self.step})
        return out

    def cart_instead(self, turn: Turn) -> bool:
        


        buy = self.roles.get('buy')
        if self.second_end or self.end_tries < 2 or self.buys or self.finished or not buy:
            return False
        if self.answer_rounds >= SECONDEND_TOLD or self.left() < 1:
            return False
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        whole = bool(held) and (not FULLONLY or self.whole_cart(held))
        if not whole and not (NEARFILL and self.roles.get('cart')):
            return False
        self.second_end = True
        added = 0 if whole else self.near_fill(held)
        faults: list[str] = []
        order = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        self.trace.append({'kind': 'lever_fire', 'why': 'secondend', 'held': len(held), 'added': added, 'items': len(self.items),
                           'answers': self.answer_rounds, 'tries': self.end_tries, 'faults': len(faults), 'step': self.step,
                           'rows': getattr(self, 'near_rows', 0)})
        if faults or not (whole or added > 0):
            return False
        turn.name, turn.args = buy, order
        turn.note = ('\n[Yama] Ending without a purchase came a second time while the shopper has added little, so yama '
                     + ('put the nearest opened listing of each item without a line in the cart and ' if added else '')
                     + 'checked out the cart.') + turn.note
        return True

    def gap_fill(self) -> int:
        


        if not GAPFILL or self.gap_tried or self.buys or self.finished or len(self.items) < 2 or not self.roles.get('cart'):
            return 0
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not held or self.whole_cart(held):
            return 0
        self.gap_tried = True
        added = self.near_fill(held)
        self.trace.append({'kind': 'lever_fire', 'why': 'gapfill', 'held': len(held), 'added': added, 'items': len(self.items),
                           'rows': getattr(self, 'near_rows', 0), 'step': self.step})
        return added

    def whole_cart(self, held: list[str]) -> bool:
        

        key = self.category_key()
        shelves = {(str((self.rows.get(i) or {}).get(key) or '').strip().lower() or i) if key else i for i in held}
        return len(shelves) >= len(self.items)

    def near_fill(self, held: list[str]) -> int:
        

        cart = self.roles.get('cart') or ''
        key = self.category_key()
        if held and not key:
            return 0
        shelf = lambda i: str((self.rows.get(i) or {}).get(key) or '') if key else ''
        bare = [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, shelf(i)) for i in held))]
        if not bare or len(bare) + len({shelf(i) for i in held}) < len(self.items) or self.left() < len(bare) + 2:
            return 0
        if RECSEARCH:
            self.rec_search(bare)
        picks: dict[int, str] = {}
        for number in bare:
            pick = self.near_line(number, set(held) | set(picks.values()))
            if not pick:
                return 0
            picks[number] = pick
        if NEARASK:
            picks = self.near_ask(picks, held)
        for number, pick in picks.items():
            if not self.cart_line(cart, number, pick):
                return 0
        return len(picks)

    def rec_search(self, bare: list[int]) -> int:
        





        if getattr(self, 'rec_done', False) or self.finished or self.cost > SPEND_HARD or self.no_credit or self.broke:
            return 0
        plan = self.value_template()
        room = min(RECSEARCH_RUNS, self.left() - self.tail() - len(bare) - 2)
        if not plan or room < 1:
            return 0
        self.rec_done = True
        tool, base, field = plan
        tried = {' '.join(str(a.get(field) or '').lower().split()) for a in self.searched} - {''}
        order = {i: n for n, i in enumerate(self.seen_order)}
        blocks = []
        for number in bare:
            ids = sorted(self.line_ids(number), key=lambda i: (i not in self.full, -order.get(i, -1)))[:RECSEARCH_ROWS]
            width = max(300, RECSEARCH_CHARS // max(1, len(bare) * max(1, len(ids))))
            shown = [str(i) + ': ' + _dump(_clip(self.full.get(i) or self.rows.get(i) or {}, 200))[:width] for i in ids]
            blocks.append(f'ITEM {number}: {self.items[number - 1].name}\nLISTINGS SEEN FOR IT:\n' + ('\n'.join(shown) or 'none'))
        schema = self.schemas.get(tool)
        whole = bool(RECARGS and isinstance(schema, dict) and self.searched)
        if whole:
            prompt = (RESEEK_ARGS + '\n\nREQUEST:\n' + str(self.request or '') + '\n\nSEARCH TOOL ' + str(tool)
                      + ' ARGUMENT SCHEMA:\n' + _dump(schema)[:RECARGS_SCHEMA]
                      + '\n\nRECENT SEARCHES (whole arguments, oldest first):\n' + '\n'.join((_dump(a) for a in self.searched))
                      + '\n\n' + '\n\n'.join(blocks))
        else:
            prompt = (RESEEK + '\n\nREQUEST:\n' + str(self.request or '') + '\n\nSEARCHES ALREADY RUN:\n'
                      + ('\n'.join(sorted(tried)) or 'none') + '\n\n' + '\n\n'.join(blocks))
        got = self.ask_json(prompt, [LIGHT], 600 if whole else 400) or {}
        props = schema.get('properties') if whole and isinstance(schema.get('properties'), dict) else {}
        done = {json.dumps(a, sort_keys=True, default=str) for a in self.searched}
        wanted: dict[int, list[Any]] = {}
        for entry in _items(got.get('searches')):
            if not isinstance(entry, dict):
                continue
            try:
                number = int(entry.get('item'))
            except (TypeError, ValueError):
                continue
            given = entry.get('args')
            if whole and isinstance(given, dict) and number in bare:
                given = {k: v for k, v in given.items() if not props or k in props}
                faults: list[str] = []
                call = _shape(given, schema, 'args', faults)
                have = wanted.setdefault(number, [])
                if given and not faults and json.dumps(call, sort_keys=True, default=str) not in done and len(have) < RECSEARCH_EACH:
                    done.add(json.dumps(call, sort_keys=True, default=str))
                    have.append(call)
                continue
            query = ' '.join(str(entry.get('query') or '').lower().split())
            if number not in bare or not query or len(query) > 60 or len(query.split()) > 6 or query in tried:
                continue
            have = wanted.setdefault(number, [])
            if query not in have and len(have) < RECSEARCH_EACH:
                have.append(query)
        runs = [(n, q[k]) for k in range(RECSEARCH_EACH) for n, q in wanted.items() if k < len(q)][:room]
        found: dict[int, list[str]] = {}
        ran = 0
        for number, query in runs:
            faults: list[str] = []
            args = query if isinstance(query, dict) else _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            ran += 1
            seen = reply.get('observation')
            self.observe(tool, seen)
            mine = found.setdefault(number, [])
            for ident, _ in self.records_in(seen):
                self.found_for.setdefault(ident, number)
                if ident not in mine:
                    mine.append(ident)
        self.rec_found = found
        self.trace.append({'kind': 'lever_fire', 'why': 'recsearch', 'lines': len(blocks), 'whole': int(whole), 'got': sum((len(q) for q in wanted.values())),
                           'args': sum((1 for q in wanted.values() for v in q if isinstance(v, dict))),
                           'ran': ran, 'rows': sum((len(v) for v in found.values())), 'step': self.step})
        return ran

    def near_ask(self, picks: dict[int, str], held: list[str]) -> dict[int, str]:
        




        if not picks or self.cost > SPEND_HARD or self.no_credit or self.broke:
            return picks
        lists = getattr(self, 'near_lists', {})
        room = max(2000, NEARASK_CHARS // len(picks))
        blocks = []
        for number, first in picks.items():
            item = self.items[number - 1]
            shown: list[str] = []
            used = 0
            for ident in lists.get(number) or [first]:
                record = self.full.get(ident) or self.rows.get(ident) or {}
                price = self.price_of(ident)
                head = ('OPENED ' if ident in self.full else 'ROW ') + str(ident) + (f' (price {price})' if price is not None else '')
                line = head + ': ' + _dump(_clip(record, 300))[:max(600, room // 4)]
                if shown and used + len(line) > room:
                    break
                shown.append(line)
                used += len(line)
            blocks.append(f'ITEM {number}: {item.name} (quantity {item.qty})\nCANDIDATES:\n' + '\n'.join(shown))
        prompt = (NEARPICK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
                  + '\n\n' + '\n\n'.join(blocks))
        got = self.ask_json(prompt, [LIGHT], 900) or {}
        chosen: dict[int, str] = {}
        for entry in _items(got.get('picks')):
            if isinstance(entry, dict):
                try:
                    chosen[int(entry.get('item'))] = str(entry.get('id') or '').strip()
                except (TypeError, ValueError):
                    continue
        out: dict[int, str] = {}
        taken = set(held)
        moved = 0
        for number, first in picks.items():
            options = lists.get(number) or [first]
            want = chosen.get(number)
            if want in options and isinstance(self.budget, (int, float)):
                price = self.price_of(want)
                if price is not None and price * self.items[number - 1].qty > self.budget + 1e-09:
                    want = None
            order = ([want] if want in options else []) + [first] + list(options)
            pick = next((i for i in order if i not in taken), None)
            if pick is None:
                return picks
            out[number] = pick
            taken.add(pick)
            moved += pick != first
        self.trace.append({'kind': 'lever_fire', 'why': 'nearask', 'lines': len(picks), 'answered': len(chosen),
                           'moved': moved, 'step': self.step})
        return out

    def near_line(self, number: int, taken: set[str]) -> str | None:
        


        wide = bool(NEARRANK)
        seen = wide and self.brief.get('pick') != 'cheapest'
        pool = [i for i in self.line_ids(number) if (i in self.full or (wide and i in self.rows)) and i not in taken and self.buyable(i)]
        if not pool:
            return None
        bag = lambda t: set(''.join((c if c.isalnum() else ' ' for c in _words(str(t)))).split())
        wants = [bag(need.partition(':')[2] or need) for owner, need in self.needs_now() if owner == number]
        qty = self.items[number - 1].qty

        def rank(ident: str) -> tuple[bool, int, int, int, int, int, float]:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            block = _fact_block(record) or record
            said = bag(' '.join((str(v) for v in block.values())))
            hits = sum((1 for w in wants if w and w <= said))
            price = self.price_of(ident)
            over = isinstance(self.budget, (int, float)) and price is not None and price * qty > self.budget + 1e-09
            opens = (-self.open_hits.get(ident, 0), -self.open_last.get(ident, -1)) if seen else (0, 0)
            cue = self.row_cue(ident) if seen else (0, 0)
            return (bool(over), -hits, opens[0], opens[1], cue[0], cue[1], price if price is not None else float('inf'))
        ranked = sorted(pool, key=rank)
        pick = ranked[0]
        top = ranked[:NEARASK_EACH]
        if RECSEARCH:
            late = [i for i in getattr(self, 'rec_found', {}).get(number, []) if i in pool and i not in top][:RECSEARCH_KEEP]
            top = top[:NEARASK_EACH - min(len(late), RETRY_SWAP)] + late
        self.near_lists = {**getattr(self, 'near_lists', {}), number: top}
        self.near_rows = getattr(self, 'near_rows', 0) + (pick not in self.full)
        return pick

    def urge_buy(self) -> str | None:
        


        if not BUY_UNTIL_TOLD or self.buy_urged or self.answer_rounds >= BUY_UNTIL_TOLD:
            return None
        if not self.rows or not self.roles.get('buy') or self.left() < len(self.items) + 2:
            return None
        self.buy_urged = True
        self.trace.append({'kind': 'buy_urged', 'answer_rounds': self.answer_rounds, 'left': self.left()})
        return ('Not run yet (no step used). The shopper has said too little so far to rule out what the store showed, '
                'and an ending without a purchase in that state almost never counts. For each requested item take the '
                'listing you opened that comes closest to the request, put it in the cart at the requested quantity '
                '(keep the lines already there) and check out.')

    def carry_out(self, turn: Turn) -> str:
        
        reply = self.act(turn.name, turn.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        seen = reply.get('observation')
        self.check_side_view(turn, seen)
        self.book_call(turn, seen)
        plain = {tool for job, tool in self.roles.items() if job != 'search'}
        browse = turn.name not in plain and not turn.ending
        paged = self.more_pages(turn.name, turn.args, seen) if browse else ''
        self.learn_viewer(turn.name, seen)
        self.book_effects(turn, seen, reply)
        return self.reply_text(turn, seen, reply, paged, browse)

    def check_side_view(self, turn: Turn, seen: Any) -> None:
        listed = turn.args.get(turn.many) if turn.many else None
        if not isinstance(listed, list) or len(listed) <= 1:
            return
        shown = _dump(seen)
        if any((str(i) in shown for i in listed)):
            self.compare_ok = max(self.compare_ok, len(listed))
            return
        self.compare_bad = min(self.compare_bad or 999, len(listed))
        turn.note += f'\n[Yama] The side-by-side view showed nothing for {len(listed)} listings: send at most {self.compare_room()} per call.'

    def book_call(self, turn: Turn, seen: Any) -> None:
        name = turn.name
        self.recent = (self.recent + [turn.mark])[-3:]
        for key, value in turn.args.items():
            group = value if isinstance(value, list) else [value]
            if group and all((isinstance(x, str) and x in self.rows for x in group)):
                self.id_args.add((name, key))
        self.absorb(seen)
        if name == self.roles.get('compare'):
            rows = self.records_in(seen)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in turn.readers - {self.roles.get('search')}:
            self.note_view(name, seen)

    def learn_viewer(self, name: str, seen: Any) -> None:
        
        fixed = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'))
        if not self.single_target(name) or name in fixed:
            return
        listing = self.ident(seen) if isinstance(seen, dict) else None
        if not listing:
            return
        self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(seen))
        if name == self.roles.get('open') and not self.facts_from[name]:
            skip = fixed + (self.roles.get('search'),)
            for tool in self.schemas:
                if tool != name and self.single_target(tool) and self.facts_from.get(tool) is not False and tool not in skip:
                    self.roles['open'] = tool
                    break
        if not self.facts_from[name] and name != self.roles.get('open') and not self.roles.get('offer'):
            self.roles['offer'] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def book_effects(self, turn: Turn, seen: Any, reply: dict[str, Any]) -> None:
        name, args = turn.name, turn.args
        if name == self.roles.get('cart'):
            self.note_cart(name, args)
            self.cart_said = _dump(seen)
            self.recent = []
            counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
            if counts:
                for value in args.values():
                    if not isinstance(value, str) or value not in self.rows:
                        continue
                    if counts[0] <= 0 and not _in_rows(seen, value):
                        self.dropped.add(value)
                    elif counts[0] > 0:
                        self.dropped.discard(value)
            lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
            if len(lines) >= len(self.built):
                self.built = lines
        if name == self.roles.get('search') and self.records_in(seen):
            self.searched = (self.searched + [dict(args)])[-8:]
        if name == self.roles.get('cart_view'):
            self.cart_view = seen
            self.cart_seen_at = self.step
        if self.is_buy(name, args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step

    def reply_text(self, turn: Turn, seen: Any, reply: dict[str, Any], paged: str, browse: bool) -> str:
        name, args = turn.name, turn.args
        parts = [self.show(seen), paged, turn.note]
        if browse:
            parts += [self.cheapest_here([seen] + self.paged_obs), self.cap_ignored(name, args, seen)]
        base = ''.join(parts)
        proof = self.evidence(seen)
        if proof:
            parts.append('\n' + proof)
        if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
            if any((isinstance(v, str) and v in self.rows and _in_rows(seen, v) for v in args.values())):
                parts.append('\n[Yama] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.')
        self.note_sold = False
        loose_lines = self.loose_lines(seen)
        if loose_lines:
            meaning = self.take_note(loose_lines)
            if meaning:
                parts.append('\n[note read] ' + meaning + '. Apply this change to your picks before checking out.')
        refused = self.is_buy(name, args) and not reply.get('finished')
        again = refused and any((line in self.spoken(seen) for line in self.sold_lines))
        loose = self.note_sold and (refused or (not self.note_named and not self.is_buy(name, args)))
        if (loose or again) and self.cart_ids() and not self.finished:
            found = self.find_sold()
            if found:
                parts.append("\n[Yama re-read the cart's lines after the refusal]\n" + found)
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            parts.append('\n' + swap)
        if self.queued:
            parts.append("\n[notes seen on Yama's own calls] " + ' | '.join(self.queued))
            self.queued = []
        if name in turn.readers:
            self.read_cache[turn.mark] = (self.version, self.step, base)
        if reply.get('finished'):
            parts.append(' [episode finished]')
        return ''.join(parts)

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
        if not holding and buy and (fresh or not self.cart_touched) and self.fill_empty_cart():
            holding = True
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
        opening = _as_records(self.view.get('opening')) if TEXTROW_ON else self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Yama] Yama tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
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
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
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
        episode = Yama(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
