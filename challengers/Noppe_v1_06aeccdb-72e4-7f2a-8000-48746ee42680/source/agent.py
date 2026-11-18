from __future__ import annotations
import json
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
# the price-capped category listing at the first checkout of a cheapest pick: only with this many steps left and
# this many seconds before the clock stop, and at most this many pages of it
CAPPED_ROOM = 18
CAPPED_CLOCK = 90
CAPPED_PAGES = 5
# an episode has 300 s: past CLOCK_STOP no new model turn starts and the held cart is checked out (or the episode ended)
CLOCK_STOP = 268
EPISODE_SECONDS = 300
# a search paged this many times with no new row of a requested kind holding its own words is not paged on
EMPTY_PAGES = 3
# after the store refused to let the episode close unbought, a second such close is held back (twice at most)
NO_SECOND_END = True
# the engine reads a profile chosen by the number of requested products (see Profile)
SIZE_PROFILES = True
# question plan: a second PLAN sample from the other model; properties both samples chose go first
P1_ON = True
# question plan: the key table keeps every key seen on 2+ read listings past the first 14 (no purchase-option fields)
P2_ON = True
# question plan: with 2 messages or fewer in all, a hinted key whose values are measures does not take the first one
P3_ON = True
# P10: one question names up to LIST_KEYS of an item's planned properties, the best ranked first. Real race 14: a
# property named alone drew the shopper's need .86 of the time, named 1st-3rd in a list .68, 4th-8th .42, not
# named .13; lists cover more properties per message, and none of 455 list questions drew a one-at-a-time reply
# one-item lists are off (user plan 11-05): they won only under race 14's reader (r13's cap-2 reader refused 10 of
# 13 on the Arena; r5-r12 lost .15-.20 on 2+ products); single questions, and multi-item messages when short of messages
P10_ON = False
LIST_KEYS = 6
# lists go out only when the request allows at most LIST_CAP messages: real races 8-14, first question a list vs a single
# one, all hidden needs drawn: the 3-message race (r14) .91 vs .74 on 1 product, .28 vs .10 on 2+; the 4-5 message races
# (r8-r12) tie on 1 product and lose .15-.20 on 2+
LIST_CAP = 3
LIST_FRONT = 3
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). A value the answer states itself is only what the shopper wants: words around it that stress how strongly they want it or that rule out everything else ("it must be oval, oval only" gives "oval") are not part of it; limits and named exclusions stay ("at least 30 cm", "not wicker").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; listings shown as sold out, and listings over the budget, do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
CLOSEST = 'A shopping agent is about to end without buying because no listing meets every requirement, and the request wants its nearest miss named instead. For every listing record shown, say which requested item it is a kind of and list the requirements in force for that item that its record does not show met, with these reading rules: ' + READING + '\nLeave stock and price out of the lists: the agent counts those itself. Return ONE JSON object and nothing else: {"listings": [{"id": listing id, "item": the number of the requested item it is a kind of (0 when none), "misses": [the property name of each requirement it does not show met]}]}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example a description naming a two-pack rules out the same shop\'s single unit of that kind); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
# a key whose shown values are mostly lists of several phrases (a features or contents field), judged by value shape only
LIST_PARTS, LIST_PART_CHARS, LIST_VALUE_CHARS = 3, 10, 60
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
# the part of NOT_PROPERTY that names a listing's purchase options or identifiers (model, condition and video stay listed)
TABLE_SKIP = '\\b(?:options?|variants?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|url|links?|images?|photos?)\\b'
TABLE_FIRST = 14
# a value made of a number and at most a short unit word (judged by shape only)
MEASURE_VALUE = '^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:(?:no\\.?|number|#)\\s*[:.]?|:)\\s*(' + FIT_CODE + ')\\b', text, re.I):
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

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

BOUND_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|than|like|either|unless|if|only)\\b'
# word endings stripped when comparing two spellings of the same word (grammar, not product words)
INFLECTIONS = ('s', 'es', 'ed', 'd', 'ing', 'er', 'ers', 'est', 'ies', 'ied', 'ly')

def _plain_words(text: Any) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', str(text).lower())]

def _spread(value: str, stated: Any) -> str:
    """How a listing's attribute holds a plain required value of two or more words: 'phrase' when one part of it
    holds those words together in order, 'apart' when a part holds them all only with other words between or in
    another order (then it names another value), else ''."""
    want = _plain_words(value)
    if len(want) < 2 or re.search(BOUND_WORDS, value, re.I):
        return ''
    parts = [_plain_words(p) for p in re.split('[,/;|]', str(stated))]
    size = len(want)
    if any((p[i:i + size] == want for p in parts + [_plain_words(stated)] for i in range(len(p) - size + 1))):
        return 'phrase'
    return 'apart' if any((set(want) <= set(p) for p in parts)) else ''

FORM_ENDS = ('ing', 'ies', 'ied', 'ed', 'es', 's', 'e', 'd')

def _form_roots(word: str) -> dict[str, str]:
    out = {word: ''}
    for end in FORM_ENDS:
        if word.endswith(end) and len(word) - len(end) >= 3 and (end != 's' or not word.endswith('ss')):
            out.setdefault(word[:-len(end)] + ('y' if end in ('ies', 'ied') else ''), end)
    return out

def _skeleton(word: str) -> str:
    return re.sub('(.)\\1', '\\1', re.sub('[aeiou]', '', word))

def _same_form(a: str, b: str) -> str:
    """'same', 'form' (one inflects the other), 'vowel' (a past participle in -en against its base: woven/weave),
    or '' when the two words are not forms of one word. Numbers match only as written."""
    if a == b:
        return 'same'
    if re.search('\\d', a + b):
        return ''
    ra, rb = (_form_roots(a), _form_roots(b))
    # a bare final e is no inflection by itself (plan / plane): one side must carry a real ending
    if any((len(r) >= 4 and {ra[r], rb[r]} - {'', 'e'} for r in ra.keys() & rb.keys())):
        return 'form'
    for x, y in ((a, b), (b, a)):
        if x.endswith('en') and (not y.endswith('en')) and len(x) >= 5 and len(y) >= 5 and x[0] == y[0]:
            base = y[:-1] if y.endswith('e') else y
            if len(_skeleton(x[:-2])) >= 2 and _skeleton(x[:-2]) == _skeleton(base):
                return 'vowel'
    return ''

def _form_join(want: list[str], got: list[str], kinds: tuple[str, ...]=()) -> tuple[str, ...] | None:
    """How got writes want word for word, with words joined or split at their boundaries (flat weave /
    flatweave) or in another form of the same word; None when some word of either side is left over."""
    if not want or not got:
        return kinds if not want and (not got) else None
    a, b = (want[0], got[0])
    kind = _same_form(a, b)
    if kind:
        found = _form_join(want[1:], got[1:], kinds + (kind,))
        if found is not None:
            return found
    if re.search('\\d', a + b):
        return None
    if len(a) > len(b) >= 3 and a.startswith(b) and (len(a) - len(b) >= 3):
        return _form_join([a[len(b):]] + want[1:], got[1:], kinds + ('join',))
    if len(b) > len(a) >= 3 and b.startswith(a) and (len(b) - len(a) >= 3):
        return _form_join(want[1:], [b[len(a):]] + got[1:], kinds + ('join',))
    return None

def _form_match(value: str, stated: Any) -> bool:
    """A listing's attribute that states a plain required value in another writing: the same
    words joined, hyphenated or split, or another form of the same word ("flat-woven" for "flatweave"). Every
    word on both sides is used in order, so a word left over (a negation, an extra qualifier, words apart or in
    another order) is never a match."""
    if re.search(BOUND_WORDS, value, re.I):
        return False
    want = re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', value.lower())
    if not want or len(''.join(want)) < 4:
        return False
    if any((re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', part.lower()) == want for part in re.split('[,/;|]', str(stated)))):
        return False
    # the whole attribute only: a list of several values is not read as stating one of them
    kinds = _form_join(want, re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', str(stated).lower()))
    return bool(kinds and set(kinds) - {'same'} and ('vowel' not in kinds or 'same' in kinds or 'join' in kinds))

def _request_value(request: str, must: str) -> str:
    """A plain must whose value word the reading cut from a longer word the request writes right after the
    property's name gets the request's own word back ("x is ab" read as "x: a")."""
    prop, sep, value = must.partition(':')
    words = re.findall('[a-z0-9]+', value.lower())
    named = re.findall('[a-z0-9]+', prop.lower())
    if not sep or not named or (not words) or len(words) > 3 or re.search('\\d', value) or re.search(BOUND_WORDS, value, re.I):
        return must
    tokens = re.findall('[a-z0-9]+', request.lower())
    size = len(named)
    spots = [i + size for i in range(len(tokens) - size + 1) if tokens[i:i + size] == named]
    out = value
    for word in words:
        if len(word) < 3:
            continue
        for at in spots:
            near = tokens[at:at + 3]
            if word in near:
                break
            longer = {t for t in near if t.startswith(word) and t[len(word):] not in INFLECTIONS}
            if len(longer) == 1:
                out = re.sub('\\b' + re.escape(word) + '\\b', longer.pop(), out, count=1, flags=re.I)
                break
    return prop + sep + out if out != value else must

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

_NO = object()
NUMERAL = '\\s*-?\\d+(\\.\\d+)?\\s*'

def _as_integer(v: Any) -> Any:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return int(float(v)) if isinstance(v, str) and re.fullmatch(NUMERAL, v) else _NO

def _as_number(v: Any) -> Any:
    if not (isinstance(v, str) and re.fullmatch(NUMERAL, v)):
        return _NO
    x = float(v)
    return int(x) if x.is_integer() else x

def _as_string(v: Any) -> Any:
    return str(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) else _NO

def _as_boolean(v: Any) -> Any:
    word = v.strip().lower() if isinstance(v, str) else ''
    return word == 'true' if word in ('true', 'false') else _NO

def _as_array(v: Any) -> Any:
    return _NO if isinstance(v, list) or v is None else [v]

# JSON-schema type name -> how a near miss of that type is turned into it
_COERCE = {'integer': _as_integer, 'number': _as_number, 'string': _as_string, 'boolean': _as_boolean, 'array': _as_array}

def _bound(schema: dict[str, Any], key: str, kinds: Any) -> Any:
    got = schema.get(key)
    return got if isinstance(got, kinds) else None

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Fit a model-written value to a tool's JSON schema: near misses of a type are converted, limits are checked
    in schema order, and every fault is reported with its path."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)

    def typed(v: Any) -> bool:
        return any((_fits_kind(v, k) for k in kinds))
    if kinds and (not typed(value)):
        turned = (_COERCE[k](value) for k in kinds if k in _COERCE)
        value = next((t for t in turned if t is not _NO), value)
        if not typed(value):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    allowed = schema.get('enum')
    if isinstance(allowed, list) and value not in allowed:
        if isinstance(value, str):
            value = next((o for o in allowed if isinstance(o, str) and o.lower() == value.strip().lower()), value)
        if value not in allowed:
            faults.append(f'{where}: allowed values are {allowed}')
    if isinstance(value, str):
        longest, shortest = (_bound(schema, 'maxLength', int), _bound(schema, 'minLength', int))
        if longest is not None and len(value) > longest:
            value = value[:longest]
        if shortest is not None and len(value) < shortest:
            faults.append(f'{where} is too short')
    elif isinstance(value, (int, float)) and (not isinstance(value, bool)):
        low, high = (schema.get('minimum'), schema.get('maximum'))
        if isinstance(low, (int, float)) and value < low:
            faults.append(f'{where} is below {low}')
        if isinstance(high, (int, float)) and value > high:
            faults.append(f'{where} is above {high}')
    elif isinstance(value, list):
        most, least = (_bound(schema, 'maxItems', int), _bound(schema, 'minItems', int))
        if most is not None and len(value) > most:
            faults.append(f'{where} allows at most {most} entries')
        if least is not None and len(value) < least:
            faults.append(f'{where} needs at least {least} entries')
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    elif isinstance(value, dict):
        props = schema['properties'] if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            faults += [f'{where}.{name} is not an allowed argument' for name in value if name not in props]
        faults += [f'{where}.{name} is required' for name in schema.get('required') or [] if name not in value]
        value = {k: _shape(v, props[k], f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _next_units(have: int, count: int | None, removal: bool, sets: bool) -> int:
    """Units of a cart line after a call naming it: taken off, set, or added."""
    if removal:
        return have - count if count else 0
    if sets:
        return 1 if count is None else count
    return have + (1 if count is None else count)

def _free_text(values: list[str]) -> bool:
    """Values no shopper could pick from: none at all, long on average, mostly slash lists, or near-unique
    longer texts that rarely share a word with the others."""
    if not values:
        return True
    mean = sum((len(v) for v in values)) / len(values)
    if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
        return True
    words = lambda text: set(re.findall('[a-z]{3,}', text))
    recur = sum((len(words(v) & words(' '.join((o for o in values if o is not v)))) > 0 for v in values))
    return len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and (recur * 2 < len(values))

def _cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the chat with the first and the last text message marked for the provider's prompt cache."""
    if len(messages) <= 1:
        return messages
    out = [dict(m) for m in messages]
    for index in (0, len(out) - 1):
        content = out[index].get('content')
        if isinstance(content, str) and content:
            out[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
    return out

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

def _listy(values: list[str]) -> bool:
    """Most shown values are long texts or lists of several phrases: the shape of a features or contents field."""
    listy = 0
    for value in values:
        parts = [x.strip() for x in re.split('[,;|]', str(value)) if x.strip()]
        if len(str(value)) >= LIST_VALUE_CHARS or (len(parts) >= LIST_PARTS and sum((len(x) for x in parts)) >= LIST_PART_CHARS * len(parts)):
            listy += 1
    return bool(values) and listy * 2 >= len(values)

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

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Profile:
    """How the shared engine spends messages, steps and time on one task size (1, 2, 3 or 4+ products).

    Real races 10-13 show the same question plan is best at every size: one property per message, every
    product asked once before any is asked twice, the planner's order first, questions after the first
    reads. A need sits on one product in a few, and nothing in the request tells which. What the size
    changes is how far the message cap goes: a message waits for a step due at an ending without a
    purchase only while the cap still leaves every product a question and one more. When the item count is
    uncertain (items merged, two items on one shelf), the profile keeps the earlier behaviour, which every
    size has run with.
    """

    def __init__(self, count: int, sure: bool):
        self.count = max(1, count)
        self.sure = sure
        self.size = str(self.count) if self.count <= 3 else '4+'
        self.active = SIZE_PROFILES and sure

    def end_room(self, limit: int, held: int) -> bool:
        """Whether `held` of `limit` messages may wait for a step due at an ending without a purchase."""
        return not self.active or limit - held >= self.count + 1

class Job:
    """One tool call from the model on its way through Kyo's desk: screened, held at the close, or run."""

    def __init__(self, name: str, args: dict[str, Any], ending: bool, many: str | None):
        self.name = name
        self.args = args
        self.ending = ending
        self.many = many
        self.searching = False
        self.extra = ''
        self.chain = ''
        self.reread = ''

    def mark(self) -> str:
        return self.name + _dump(self.args)

    def listed(self) -> list[Any] | None:
        """The listings a side-by-side call names, when it is one."""
        ids = self.args.get(self.many) if self.many else None
        return ids if isinstance(ids, list) else None

# the episode's starting state: (attribute names, start value or a factory for a fresh one)
EPISODE_STATE = (
    ('step waited misses calls loop_calls msg_limit msg_used reserve gate_msgs version held refuse_blocks buys end_refused end_blocks refusals trails trail_blocks compare_ok compare_bad pages_read line_holds fit_holds link_searches', 0),
    ('cost', 0.0),
    ('cart_at cart_seen_at refused_at', -1),
    ('cart_said', ''),
    ('broke no_credit asked cart_touched note_named audited note_sold pushed late_held completed wider_held capped_more capped_done hunted rule_told', False),
    ('budget cart_view confirmed id_key price_key stock_key routed last_seen closest word_rule', None),
    ('trace process done_log done_steps qa seen_order notes leads built queued recent chat refs paged_obs cap_calls kind_calls plan_order bound searched widened page_log search_log rule_log', list),
    ('brief roles rows full cart_lines offer_seen gone lead_items verdicts facts_from ref_values ref_pages tool_keys view_keys read_cache shapes link_seen', dict),
    ('gate_ok heard checked unavailable sold_lines id_args dropped opened cheaper_marks form_seen repick closest_ids built_needs', set),
)

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        cap = self.view.get('max_steps')
        steps = int(cap) if _count(cap) is None and isinstance(cap, float) else _count(cap)
        self.max_steps = steps if steps and steps > 0 else 30
        self.finished = bool(self.view.get('finished'))
        # every counter, flag, record and memory of the episode starts empty
        for names, start in EPISODE_STATE:
            for name in names.split():
                setattr(self, name, start() if callable(start) else start)
        self.items: list[Item] = []
        self.began = time.monotonic()
        self.profile = Profile(1, False)

    def route(self) -> Profile:
        """Pick the profile for the current item count; again whenever that count has changed."""
        sure = not self.shared_shelf(self.brief) and not any((t.get('kind') == 'items_merged' for t in self.trace))
        if self.routed != (len(self.items), sure):
            self.routed = (len(self.items), sure)
            self.profile = Profile(len(self.items), sure)
            self.trace.append({'kind': 'profile', 'size': self.profile.size, 'sure': sure})
        return self.profile

    def pointers_open(self) -> list[str]:
        """Requirements in force whose value sits on another listing and has not been read yet."""
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def spent(self) -> float:
        return time.monotonic() - self.began

    def late(self) -> bool:
        """Past the clock stop: the remaining seconds go to finishing, not to new turns."""
        return self.spent() > CLOCK_STOP

    def left(self) -> int:
        return self.max_steps - self.step

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
            # past the clock stop a second model would start a call the 300 s limit cuts off
            if model != models[0] and self.late():
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL or self.late():
                    break
                pause = WAITS[tries] * (0.75 + 0.05 * (self.calls % 11))
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One chat call through the proxy: (the reply message, 200), or (None, the failure status)."""
        body: dict[str, Any] = {'model': model, 'messages': _cache_marked(messages) if model.startswith('anthropic/') else messages, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        # every call ends inside the episode's 300 s: its wait is cut to the seconds left (Arena run 13021: a call
        # started near the clock stop ran past the limit and the episode ended with no finish)
        room = EPISODE_SECONDS - self.spent() - 6
        if room < 4:
            self.trace.append({'kind': 'llm_skip', 'model': model, 'room': round(room, 1)})
            return (None, 0)
        base = getattr(_llm, 'timeout', None)
        if isinstance(base, (int, float)):
            tries = 1 + max(0, int(getattr(_llm, 'max_retries', 0) or 0))
            _llm.timeout = max(4, int(min(base, room / tries)))
        try:
            result = _llm.post_verbose('/inference/chat/completions', body)
        finally:
            if isinstance(base, (int, float)):
                _llm.timeout = base
        data = result.data if isinstance(result.data, dict) else None
        if data is not None and data.get('choices'):
            spent = (data.get('usage') or {}).get('cost')
            if isinstance(spent, (int, float)):
                self.cost += float(spent)
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        failed = data.get('error') if data is not None else None
        if isinstance(failed, dict):
            status, body_text = (_count(failed.get('code')), failed.get('message'))
        else:
            error = result.error or {}
            status, body_text = (error.get('status'), error.get('body'))
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(body_text or '')[:200]})
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
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    # a view that came back without a listing (a refused side-by-side size) has not opened it
                    shown = _dump(seen)
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip() in shown))
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

    def read_request(self) -> None:
        """Turn the request into items, tool roles, limits and the steps the request asks for. The reading
        comes from one brief call; a second opinion is asked when it is incomplete or puts two items on one
        shelf. The message reserve follows the profile chosen for the item count."""
        self.brief = self.brief_reading()
        roles = self.brief.get('roles') if isinstance(self.brief.get('roles'), dict) else {}
        self.roles = {job: tool for job, tool in roles.items() if isinstance(tool, str) and tool in self.schemas}
        self.split_end_modes()
        self.take_links(self.take_items(_items(self.brief.get('items'))))
        self.take_limits()
        self.reserve = self.end_reserve(self.route())

    def tool_lines(self, skip: set[str] | None=None) -> str:
        specs = [t['function'] for t in self.tools if t['function']['name'] not in (skip or set())]
        return '\n'.join((f"- {spec['name']}: {spec.get('description', '')} | parameters: {_dump(spec.get('parameters') or {})}" for spec in specs))

    def brief_reading(self) -> dict[str, Any]:
        prompt = BRIEF + '\nTOOLS:\n' + self.tool_lines() + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))

        def usable(got: dict[str, Any]) -> bool:
            return bool(_items(got.get('items'))) and isinstance(got.get('roles'), dict) and bool(got['roles'].get('buy'))
        first = self.ask_json(prompt, [LIGHT], 4000) or {}
        if self.no_credit:
            return first
        if not usable(first):
            return self.ask_json(prompt, [HEAVY, LIGHT], 2500) or first
        if self.shared_shelf(first):
            second = self.ask_json(prompt, [HEAVY], 2500) or {}
            if usable(second):
                return second
        return first

    def split_end_modes(self) -> None:
        """One tool may both check out and end without a purchase: then the arguments that pick the mode must differ."""
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(self.brief.get(key), dict):
                self.brief[key] = {}
        tool = self.roles.get('buy')
        buying, ending = (self.brief['buy_args'], self.brief['refuse_args'])
        if not tool or tool != self.roles.get('refuse') or (buying and ending and (buying != ending)):
            return
        ask = f'The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(tool)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
        modes = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
        buying, ending = (modes.get('buy_args'), modes.get('refuse_args'))
        if isinstance(buying, dict) and isinstance(ending, dict) and (buying != ending):
            self.brief['buy_args'], self.brief['refuse_args'] = (buying, ending)

    def take_items(self, entries: list[Any]) -> dict[int, int]:
        """Items from the brief. Two entries on one shelf whose requirements never give one property two values
        are one item. Returns each brief entry's number in the final item list."""

        def clash(one: Item, other: Item) -> bool:
            heads = [(_words(m.split(':')[0]), _words(m.partition(':')[2])) for m in one.musts]
            return any((key == _words(m.split(':')[0]) and value != _words(m.partition(':')[2]) for key, value in heads for m in other.musts))
        became: dict[int, int] = {}
        for number, entry in enumerate((e for e in entries if isinstance(e, dict)), 1):
            quantity = _count(entry.get('quantity'))
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            kept_words = [_request_value(self.request, m) for m in musts]
            if kept_words != musts:
                self.trace.append({'kind': 'request_value', 'read': [m for m in musts if m not in kept_words], 'now': [m for m in kept_words if m not in musts]})
            item = Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, kept_words)
            home = next((n for n, kept in enumerate(self.items, 1) if item.shelf.strip() and _words(kept.shelf) == _words(item.shelf) and (not clash(kept, item))), 0)
            if not home:
                self.items.append(item)
                became[number] = len(self.items)
                continue
            kept = self.items[home - 1]
            kept.musts.extend((m for m in item.musts if m not in kept.musts))
            became[number] = home
            self.trace.append({'kind': 'items_merged', 'into': kept.name, 'dropped': item.name})
        if not self.items:
            self.items.append(Item('the requested product', '', 1, []))
        return became

    def take_links(self, became: dict[int, int]) -> None:
        """Part-device pairs, and requirements whose value is read from another listing."""
        size = len(self.items)
        for pair in _items(self.brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            link = (became.get(_count(pair.get('item')) or 0, 0), became.get(_count(pair.get('with')) or 0, 0))
            if 0 < link[0] <= size and 0 < link[1] <= size and (link[0] != link[1]) and (link not in self.bound):
                self.bound.append(link)
        musts = [(n, m) for n, item in enumerate(self.items, 1) for m in item.musts]
        for lookup in _items(self.brief.get('lookups')):
            text = str(lookup).strip()
            if not text:
                continue
            head = _words(text.split(':')[0])
            owner = next(((n, m) for n, m in musts if m == text), None) or next(((n, m) for n, m in musts if _words(m.split(':')[0]) == head), None) or (1, text)
            if owner not in self.refs:
                self.refs.append(owner)
        self.unmix()

    def take_limits(self) -> None:
        """Budget, message cap, step cap and the steps the request asks for at a stage."""
        budget = self.brief.get('budget')
        if isinstance(budget, str) and budget.strip().replace('.', '', 1).isdecimal():
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        cap = _count(self.brief.get('messages'))
        if cap is not None and cap >= 0:
            self.msg_limit = cap
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(self.brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(self.brief.get('process')):
            step = str(entry.get('step') or '').strip() if isinstance(entry, dict) else ''
            if step:
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': step, 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})

    def end_reserve(self, profile: Profile) -> int:
        """Messages kept back for the steps due at the close: the order's, or the ending's when the profile leaves room."""
        if not self.roles.get('ask'):
            return 0

        def owed(stage: str) -> int:
            return max((p['messages'] for p in self.process if p['when'] == stage), default=0)
        held = owed('order')
        if owed('end') > held and profile.end_room(self.msg_limit, owed('end')):
            held = owed('end')
        return min(held, self.msg_limit)

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
        item = self.items[number - 1]
        if re.search('\\b(?:other than|unlike)\\b', value, re.I):
            return False
        # A likeness to something the request reading found no listing for (a request requirement, not one in
        # its lookups) is the shopper's own description when no listing states that property.
        likened = need in item.musts and need not in item.told and re.search('\\bsame as\\b', value, re.I)
        if not likened and re.search('\\b(?:listing|page|shown|same as)\\b', value, re.I):
            return False
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
        """The requirements in force as the model reads them: one line per item, then the order's terms."""
        many = len(self.items) > 1
        lines = [self.item_ledger(number, item, many) for number, item in enumerate(self.items, 1)]
        return '\n'.join(('- ' + line for line in lines + self.terms_ledger()))

    def item_ledger(self, number: int, item: Item, many: bool) -> str:
        live = [m for m in item.musts + item.told if m not in item.dropped]
        told = [m for m in live if self.described(number, m)]
        pointed = lambda m: f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else ''
        needs = [m + pointed(m) for m in live if m not in told]
        title = (f'item {number}: ' if many else 'item: ') + item.name + (f' (store label: {item.shelf})' if item.shelf else '') + f', quantity {item.qty}'
        bits = [title, 'requirements: ' + ('; '.join(needs) or 'none beyond the kind of product')]
        if told:
            bits.append("described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told))
        if item.free:
            bits.append('shopper does not mind: ' + ', '.join(item.free))
        return '; '.join(bits)

    def terms_ledger(self) -> list[str]:
        brief = self.brief
        out: list[str] = []
        if self.budget is not None:
            each = brief.get('budget_scope') == 'each'
            out.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {('for each item' if each else 'in total')}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            out.append('pick: the cheapest listing that fits (lowest price)')
        else:
            out.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' ') if pick in ('most', 'least') else 'pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            out.append('in stock only')
        out += [f'wish (tie-breaker only): {w}' for w in _items(brief.get('wishes'))]
        out += [f'rule from the request: {r}' for r in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            out.append('if nothing fits: end without naming or suggesting any listing')
        stage = lambda when: 'before choosing' if when == 'choose' else 'at the order (Kyo does it when you check out)' if when == 'order' else 'at an ending without a purchase (Kyo does it then)' if when == 'end' else None
        for entry in self.process:
            if stage(entry['when']) is None:
                raise KeyError(entry['when'])
            out.append(f"step the request asks for, {stage(entry['when'])}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            out.append(self.attribute_source(facts))
        return out

    def attribute_source(self, facts: str) -> str:
        bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
        pages = sorted((t for t in self.page_tools() if t != facts))
        line = f'listing attributes come from {facts}'
        if bare:
            line += f"; {', '.join(bare)} show none"
        if pages:
            line += f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}"
        return line

    def status(self) -> str:
        left = self.left()
        text = f'[Kyo] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
            if self.read_room():
                text += f' {"No line is in the cart yet" if not self.cart_ids() else "An item has no cart line yet"}: read the most promising listing not read yet ({"the cheapest such row" if self.brief.get("pick") == "cheapest" else "one whose row shows the wanted values"}) with {self.facts_tool()}, cart it if it meets every requirement, then check out.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
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

    def catch_all(self, key: str) -> bool:
        """The key's shown values are mostly lists of several phrases or long texts: a bundle of features, asked last."""
        stamp = (len(self.full), self.version)
        if vars(self).get('_catch_stamp') != stamp:
            self._catch_stamp = stamp
            self._catch_keys = {_words(name) for _, name, _, values in self.key_table() if _listy(values)}
        return _words(key) in self._catch_keys

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Rows of (category, attribute key, listings carrying it, sample values) for the question planner: the
        requested kinds' categories only, their free-text and near-unique keys left out, twin keys merged."""
        groups = self.key_groups()
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted(self.asked_groups(groups).items(), key=lambda kv: -kv[1]['n']):
            kept = self.kept_keys(group)
            shown = kept[:14]
            if P2_ON:
                # a need can sit on a key that few listings carry: past the first rows, every key seen on two or more
                shown = kept[:TABLE_FIRST] + [kv for kv in kept[TABLE_FIRST:] if kv[1]['n'] >= 2 and (not re.search(TABLE_SKIP, kv[0]))]
            table += [(label, name, slot['n'], [v[:40] for v in slot['values'][:4]]) for name, slot in shown]
        return table

    def key_groups(self) -> dict[str, dict[str, Any]]:
        """Per category label: how many read pages, and per attribute name the pages carrying it with their values."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            group = groups.setdefault(str(record.get(cat_key) or 'listings') if cat_key else 'listings', {'n': 0, 'keys': {}})
            group['n'] += 1
            first: dict[str, Any] = {}
            for key, value in (_fact_block(record) or {}).items():
                first.setdefault(_words(key), value)
            for name, value in first.items():
                if not name:
                    continue
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        return groups

    def asked_groups(self, groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """The groups whose label is a requested item's kind (by name, store label or long shared word), or all
        of them when none is."""
        roots = lambda text: {w for w in _stem(text) if len(w) >= 4}
        wanted = [((_words(item.shelf or ''), _words(item.name)), roots(item.shelf or '') | roots(item.name)) for item in self.items]
        mine = lambda label: any((_words(label) in names or roots(label) & long for names, long in wanted)) or self.owner_of(label)
        matched = {label for label in groups if mine(label)}
        return {label: group for label, group in groups.items() if label in matched} if matched else groups

    def kept_keys(self, group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
            if _free_text(list(slot['per'].values())):
                continue
            twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
            if twin is None:
                kept.append((name, slot))
                continue
            for ident, text in slot['per'].items():
                twin['per'].setdefault(ident, text)
                if text not in twin['values']:
                    twin['values'].append(text)
            twin['n'] = len(twin['per'])
        kept.sort(key=lambda kv: -kv[1]['n'])
        return kept

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
        """Once per tool: a light model names the parts of a list result (its rows, count, next-page value and
        argument, price and stock fields); names the result or the schema does not carry are dropped."""
        if name in self.shapes:
            return self.shapes[name]
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))] if isinstance(observation, dict) else []
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows')
        if not rows or rows not in lists:
            return None
        sources = {'count': observation, 'next': observation, 'next_arg': (self.schemas.get(name) or {}).get('properties') or {}, 'price': observation[rows][0], 'stock': observation[rows][0]}
        shape: dict[str, Any] = {'rows': rows}
        for part, where in sources.items():
            said = got.get(part)
            shape[part] = said if isinstance(said, str) and said in where else None
        total = observation.get(shape['count']) if shape['count'] else None
        if shape['count'] and (isinstance(total, bool) or not isinstance(total, int)):
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
        """For a cheapest, most or least pick, Kyo reads the next pages of a short first-page result itself when
        steps allow (at most AUTO_PAGES in all), or says how to see the rest."""
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        self.learn_kind_call(name, args, observation)
        words = self.search_words(name, args)
        due = self.pages_due(plan, args, observation, pick, bool(re.search('[a-z]', words, re.I)) and (not re.search('\\d', words)))
        if isinstance(due, str):
            return due
        count, shown, param, value = plan
        texts, value = self.read_more(name, args, param, value, due)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        if not texts:
            return ''
        rest = '' if value is None else f"\n[Kyo] {count} listings matched and Kyo has read {shown * (len(texts) + 1)} rows of them; search again with other words, or page on to see the rest."
        return '\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest

    def search_words(self, name: str, args: dict[str, Any]) -> str:
        """A search's own words: its text arguments that are neither one of the tool's fixed choices nor a category
        label of the rows seen."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        cat_key = self.category_key()
        labels = {str(r.get(cat_key) or '').strip().lower() for r in self.rows.values()} - {''} if cat_key else set()
        return ' '.join((v for k, v in args.items() if isinstance(v, str) and v.strip() and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and v.strip().lower() not in labels))

    def pages_due(self, plan: Any, args: dict[str, Any], observation: Any, pick: Any, worded: bool=True) -> int | str:
        """How many next pages Kyo reads itself, or the note to show instead ('' for none)."""
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        if isinstance(given, int) and (not isinstance(given, bool)):
            later = given not in (None, 0, 1)
        else:
            later = bool(str(given or '').strip())
        if value is None or later or count <= shown or (pick == 'cheapest' and self.price_sorted(observation)):
            return ''
        # real races 13-14, cheapest picks: a later page of a category listing or of a search led by a number first
        # showed a paying listing on 0.4-4.7% of calls, a later page of a worded search 4-9%, the first page of a new
        # worded search 7-18%; Kyo reads one next page of a worded search itself, and none of the others
        pages = min(-(-(count - shown) // shown), 1 if worded else 0)
        if pages and self.pages_read + pages <= AUTO_PAGES and self.left() - self.tail() - pages >= 12:
            return pages
        return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing. A new search with other words (the product's plain name, a synonym, words from fitting titles seen) shows fitting listings far more often than later pages do; page on with {param}={_dump(value)} once new words run out."

    def read_more(self, name: str, args: dict[str, Any], param: str, value: Any, pages: int) -> tuple[list[str], Any]:
        texts: list[str] = []
        while len(texts) < pages and value is not None:
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
        return (texts, value)

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
        return f'\n[Kyo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        listed = self.capped_listing()
        if listed:
            rest = ' and page on through the rows not read yet while steps allow' if self.capped_more else ''
            return "Not run yet. The pick is the cheapest, so Kyo listed the cart's category at or under the price of its line itself:\n" + listed + f'\nRead the attributes of the cheapest of these rows that may fit{rest}; swap in the cheapest that fits every requirement, then check out.'
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def learn_kind_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A first-page search that returned rows, kept with the category labels of its rows: a base for a
        price-capped listing of that category."""
        paging = (self.shapes.get(name) or {}).get('next_arg')
        cat_key = self.category_key()
        rows = self.records_in(observation)
        if not rows or not cat_key or (paging and args.get(paging) not in (None, 0, '')):
            return
        kinds = {str(r.get(cat_key) or '').strip().lower() for _, r in rows} - {''}
        if kinds and (not any((c[0] == name and c[1] == args for c in self.kind_calls))):
            self.kind_calls = (self.kind_calls + [(name, dict(args), kinds)])[-12:]

    def cap_guesses(self) -> list[tuple[str, str, dict[str, Any], set[str]]]:
        """Calls that could list a category under a price: the capped searches the store has answered, then the
        category searches whose tool has an unused numeric argument named like an upper limit."""
        out = list(self.cap_calls)
        for name, args, kinds in self.kind_calls:
            props = (self.schemas.get(name) or {}).get('properties') or {}
            paging = (self.shapes.get(name) or {}).get('next_arg')
            for key, spec in props.items():
                if key == paging or key in args or (not isinstance(spec, dict)) or (not {'number', 'integer'} & set(_kinds(spec))):
                    continue
                if re.search('max|most|under|below|limit|cap|upto|up_to', key, re.I) and (not re.search('min|least|page|size|offset|start|row|count', key, re.I)):
                    out.append((name, key, dict(args), kinds))
        return out

    def capped_listing(self) -> str:
        """Once, at the first checkout of a cheapest pick with steps and seconds to spare: list each cart line's
        category at or under that line's price (up to CAPPED_PAGES pages), and name the rows not opened yet,
        cheapest first: a fitting listing that no search words found may still be cheaper."""
        if self.capped_done or self.brief.get('pick') != 'cheapest' or self.finished:
            return ''
        self.capped_done = True
        if self.left() < CAPPED_ROOM or self.spent() > CLOCK_STOP - CAPPED_CLOCK:
            self.trace.append({'kind': 'capped_skip', 'left': self.left(), 'spent': round(self.spent())})
            return ''
        cat_key = self.category_key()
        texts = []
        self.capped_more = False
        for line in self.cart_ids():
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower() if cat_key else ''
            if price is None or not kind or self.left() < self.tail() + 8 or self.spent() > CLOCK_STOP - CAPPED_CLOCK:
                continue
            calls = [c for c in self.cap_guesses() if kind in c[3]]
            if not calls:
                continue
            tool, key, args, _ = min(calls, key=lambda c: (c[3] != {kind}, len(c[2])))
            paging = (self.shapes.get(tool) or {}).get('next_arg')
            call = {**{k: v for k, v in args.items() if k not in (key, paging)}, key: int(price) if float(price).is_integer() else price}
            first = dict(call)
            seen_pages: list[Any] = []
            shown: list[str] = []
            plan = None
            for _ in range(CAPPED_PAGES):
                faults: list[str] = []
                shaped = _shape(call, self.schemas.get(tool), 'args', faults)
                if faults or self.left() < self.tail() + 6 or self.spent() > CLOCK_STOP - CAPPED_CLOCK / 2:
                    break
                reply = self.act(tool, shaped)
                if reply is None:
                    break
                seen = reply.get('observation')
                self.observe(tool, seen)
                seen_pages.append(seen)
                text = self.show(seen)
                self.read_cache[tool + _dump(shaped)] = (self.version, self.step, text)
                shown.append(f'[{tool} {_dump(shaped)}, one step]\n' + text)
                plan = self.paging(tool, seen) if paging else None
                if self.finished or not plan or plan[3] is None or (not self.records_in(seen)):
                    break
                call = {**call, plan[2]: plan[3]}
            if not seen_pages:
                continue
            self.learn_cap(tool, first, seen_pages[0])
            found = {i for page in seen_pages for i, _ in self.records_in(page)}
            cart = self.cart_ids()
            rows = sorted(((p, i) for i in found for p in [self.price_of(i)] if p is not None and p < price and i not in cart and i not in self.full and self.buyable(i)))
            # the cheapest rows under the line are opened by Kyo itself, so the pick and the review read their attributes
            if rows and self.spent() <= CLOCK_STOP - CAPPED_CLOCK / 2 and self.left() >= self.tail() + 8:
                tried = [i for _, i in rows[:CHEAPER_OPENS]]
                opened = self.fill_facts(tried, self.tail() + 6)
                pages = [self.full[i] for i in tried if i in self.full]
                if opened:
                    proof = self.evidence(pages)
                    shown.append('[Kyo opened the cheapest of these rows itself, one step each]\n' + '\n'.join(opened) + ('\n' + proof if proof else ''))
                    self.trace.append({'kind': 'capped_open', 'line': line, 'opened': [i for i in tried if i in self.full]})
                rows = [(p, i) for p, i in rows if i not in self.full]
            more = plan[0] - sum((len(self.records_in(page)) for page in seen_pages)) if plan and plan[3] is not None else 0
            self.trace.append({'kind': 'capped_listing', 'line': line, 'tool': tool, 'pages': len(seen_pages), 'rows': len(found), 'unread': len(rows), 'more': max(0, more)})
            note = f'\n[Kyo] Rows under the cart line ({price:g}) not opened yet, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in rows[:12])) + ('' if len(rows) <= 12 else f' and {len(rows) - 12} more') + '. The pick is the cheapest of them whose attributes meet every requirement, if any.' if rows else ''
            if more > 0:
                self.capped_more = True
                note += f'\n[Kyo] {more} more rows of this listing are not read yet: page on with {plan[2]}={_dump(plan[3])}.'
            texts.append('\n'.join(shown) + note)
        return '\n'.join(texts)

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

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def learn_cap(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A numeric search argument (not the paging one) that every returned row's price stays within acts as the
        search's price cap: the call is kept, with the category labels its rows carry."""
        rows = self.records_in(observation)
        prices = [self.price_of(i) for i, _ in rows]
        if len(prices) < 3 or None in prices:
            return
        paging = (self.shapes.get(name) or {}).get('next_arg')
        cat_key = self.category_key()
        kinds = {str(r.get(cat_key) or '').strip().lower() for _, r in rows} - {''} if cat_key else set()
        for key, value in args.items():
            if key == paging or isinstance(value, bool) or (not isinstance(value, (int, float))) or value < max(prices):
                continue
            base = {k: v for k, v in args.items() if k not in (key, paging)}
            if not any((c[0] == name and c[1] == key and {k: v for k, v in c[2].items() if k not in (key, paging)} == base for c in self.cap_calls)):
                self.cap_calls = (self.cap_calls + [(name, key, dict(args), kinds)])[-12:]

    def cheaper_search(self) -> str:
        """After a note dropped a requirement on a cheapest pick, list that item's category at or under the lowest
        price seen for it, with a price-capped search the store has already answered (one step). A listing that
        failed only the dropped requirement may never have been looked at, and rows come in no price order."""
        due, self.repick = (sorted(self.repick), set())
        if not due or self.finished or self.brief.get('pick') != 'cheapest' or (not self.cap_calls):
            return ''
        cat_key = self.category_key()
        texts = []
        for number in due:
            if self.finished or self.left() < self.tail() + 3 or number > len(self.items):
                break
            qty = max(1, self.items[number - 1].qty)
            ids = self.item_ids(number)
            kinds = {str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() for i in ids} - {''} if cat_key else set()
            prices = [p for i in ids if self.buyable(i) for p in [self.price_of(i)] if p is not None and (self.budget is None or p * qty <= self.budget)]
            calls = [c for c in self.cap_calls if kinds and c[3] & kinds]
            if not prices or not calls:
                continue
            tool, key, args, _ = min(calls, key=lambda c: len(c[2]))
            paging = (self.shapes.get(tool) or {}).get('next_arg')
            floor = min(prices)
            faults: list[str] = []
            call = _shape({**{k: v for k, v in args.items() if k != paging}, key: int(floor) if float(floor).is_integer() else floor}, self.schemas.get(tool), 'args', faults)
            hit = self.read_cache.get(tool + _dump(call))
            if faults or (hit and hit[0] == self.version):
                continue
            reply = self.act(tool, call)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            shown = self.show(seen)
            self.read_cache[tool + _dump(call)] = (self.version, self.step, shown)
            if any(((self.price_of(i) or 0) > floor for i, _ in self.records_in(seen))):
                self.cap_calls = [c for c in self.cap_calls if (c[0], c[1]) != (tool, key)]
            self.trace.append({'kind': 'cheaper_search', 'item': number, 'tool': tool, 'args': call, 'rows': len(self.records_in(seen))})
            texts.append(f'[Kyo listed this category at or under {floor:g}, the lowest price seen for item {number}, because a dropped requirement can make a listing never looked at the cheapest fit ({tool} {_dump(call)}, one step)]\n' + shown + self.cheapest_here([seen]))
        return '\n'.join(texts)

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
            self.widened += [i for i in picks if i in self.full]
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
                        self.widened += [i for i in picks[:2] if i in self.full]

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
        before = set(self.full)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)
        self.widened += [i for i in self.full if i not in before]

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
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
        self.route()
        self.widen()
        lanes = self.plan_lanes(self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        many = len(self.items) > 1
        rested = 0
        listing = P10_ON and 0 < self.msg_limit <= LIST_CAP
        rounds = self.item_rounds(lanes) if many and (not listing) else []
        self.trace.append({'kind': 'ask_form', 'form': 'list' if listing else 'multi-item' if rounds else 'single', 'messages': self.msgs_free(), 'items': len(lanes)})
        answered: dict[int, set[str]] = {n: set() for n in lanes}
        fronted: dict[int, set[str]] = {n: set() for n in lanes}
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
            pairs = [(n, next((x for x in lanes[n] if _words(x) not in used[n]), '')) for n in rounds.pop(0)] if rounds and (not follow) else []
            pairs = [(n, label) for n, label in pairs if label]
            if len(pairs) >= 2:
                for n, label in pairs:
                    used[n].add(_words(label))
                    spent[n] += 1
                batch = self.send([(0, pairs[0][1], self.items_text(pairs))])
                if not batch:
                    break
                sent += batch
                had = [len(item.told) for item in self.items]
                # a reply answers one of the items at most: a "don't mind" cannot be pinned to an item
                heard = {**self.hear(batch, (LIGHT, HEAVY)), 'free': []}
                self.take_heard(heard, told, ' '.join((a for _, _, a in batch)))
                fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
                for n in fresh:
                    found[n] = found.get(n, 0) + 1
                if not fresh and 1 in [_count(x) for x in _items(heard.get('crowded'))]:
                    # the shopper wants one thing per message: single questions from here on, the first item first
                    rounds = []
                    used[pairs[0][0]].discard(_words(pairs[0][1]))
                    follow.insert(0, (pairs[0][0], [pairs[0][1]], False))
                    self.trace.append({'kind': 'items_split', 'items': [n for n, _ in pairs]})
                continue
            if follow:
                number, labels, shown = follow.pop(0)
            elif pairs:
                (number, label), = pairs
                labels, shown = [label], False
            else:
                others = {n: v for n, v in lanes.items() if n != rested}
                number, labels = self.next_ask(others, found, spent, used) if others else (0, [])
                if not labels:
                    number, labels = self.next_ask(lanes, found, spent, used)
                shown = False
                if listing:
                    if not labels:
                        # every planned property went out once: list again the ones that drew no answer
                        number = next((n for n in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0))) if self.list_labels(lanes[n], used[n], answered[n], fronted[n])), 0)
                    labels = self.list_labels(lanes[number], used[number], answered[number], fronted[number]) if number else []
                if not labels:
                    break
            rested = 0
            used[number].update((_words(label) for label in labels))
            fronted[number].update((_words(label) for label in labels[:LIST_FRONT]))
            batch = self.send([(number, labels[0], self.ask_text(number, labels, shown))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if len(labels) >= 3:
                # a reply to a list answers one of its properties at most: a "don't mind" names none of them
                heard = {**heard, 'free': []}
                for entry in _items(heard.get('needs')):
                    prop = _stem(str((entry or {}).get('need') or '').split(':', 1)[0]) if isinstance(entry, dict) else frozenset()
                    answered[number] |= {_words(x) for x in lanes[number] if prop and (_names(prop, _stem(x)) or _names(_stem(x), prop))}
            self.take_heard(heard, told, ' '.join((a for _, _, a in batch)))
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh:
                continue
            split = 1 in [_count(x) for x in _items(heard.get('crowded'))]
            if split and len(labels) >= 3:
                # the shopper wants one property per message: single questions from here on, the list's head first
                listing = False
                used[number] -= {_words(x) for x in labels[1:]}
                follow.insert(0, (number, [labels[0]], False))
                self.trace.append({'kind': 'list_split', 'item': number})
                continue
            lost = bool(heard) and (not any((_items(heard.get(k)) for k in ('needs', 'drop', 'free')))) and (heard.get('budget') is None)
            # a split reply on a longer label keeps its head-word re-ask (.28 in real races), the best second try
            headed = split and len(labels[0].split()) >= 2 and _words(labels[0].split()[-1]) not in used[number]
            if many and (split or lost) and (not headed):
                # A reply that gives nothing (neither a value, a drop nor a "don't mind") or wants the message split
                # rarely turns into a value when the same item is asked again; the next message goes to another
                # item's next planned property, and this item waits one turn.
                rested = number
                self.trace.append({'kind': 'ask_rest', 'item': number, 'split': split})
                continue
            if not split:
                continue
            # After a reply that wants a single question per message, only the head word of a longer label is worth a
            # second try; the same label again, with or without example values, almost never draws a value
            # (real races 10-13), so the next planned property goes out instead.
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], False))
        found_items = [n for n in found if found[n] > 0]
        read_before = list(self.full)
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        widened = set(self.widened)
        pages = [self.full[i] for i in dict.fromkeys(self.widened) if i in self.full]
        self.widened = []
        if pages:
            # Pages Kyo read so the questions fit the store: the model has not seen them in its own results.
            summary += '\n[pages Kyo read while preparing the questions, not shown to you before (each used one step)]\n' + '\n'.join((self.show(page)[:1500] for page in pages))[:8000]
            proof = self.evidence(pages)
            if proof:
                summary += '\n' + proof
        if told:
            again = self.read_again([i for i in read_before if i not in widened])
            if again:
                summary += '\n' + again
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def read_again(self, ids: list[str]) -> str:
        """Listings opened before the shopper's answers, held against every requirement now in force (no step).
        Real races 5-14: in failed picks that had opened the paid listing, it was opened only before the hidden
        need was drawn in .35-.57 of them (r11, r13, r14) and never judged against that need again."""
        kinds = {i for n in range(1, len(self.items) + 1) for i in self.item_ids(n)}
        pool = [i for i in reversed(self.seen_order) if i in ids and (i in kinds or not kinds) and self.buyable(i) and i not in self.gone]
        if self.brief.get('pick') == 'cheapest':
            pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
        proof = self.evidence([self.full[i] for i in pool[:8]], False)
        if not proof:
            return ''
        order = ', cheapest first' if self.brief.get('pick') == 'cheapest' else ''
        return f"[listings you opened before these answers{order}, their keys named for a requirement in force: judge them again against the new requirements before searching further] " + proof

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """(search tool, the arguments a value search keeps, the free-text field it writes) from the newest search
        that wrote words; None when no search tool or no such field is known."""
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        worded = lambda a, k: isinstance(a.get(k), str) and re.search('[a-z]', a[k], re.I)
        for args in reversed(self.searched):
            fields = [k for k in named or texts if worded(args, k)]
            if not named:
                fields = [k for k in fields if self.free_field(k)]
            if fields and tool:
                field = fields[0]
                return (tool, self.kept_args(args, field, schema, props, [a for a in self.searched if worded(a, field)]), field)
        return None

    def free_field(self, key: str) -> bool:
        """A text field the searches so far filled with varied phrases, never with a category label."""
        cat_key = self.category_key()
        labels = {str(r.get(cat_key) or '').strip().lower() for r in self.rows.values()} - {''} if cat_key else set()
        sent = [a.get(key) for a in self.searched]
        return len({str(v) for v in sent if v}) >= 2 and any((len(str(v).split()) >= 2 for v in sent)) and (not any((str(v or '').strip().lower() in labels for v in sent)))

    def kept_args(self, args: dict[str, Any], field: str, schema: dict[str, Any], props: dict[str, Any], worded: list[dict[str, Any]]) -> dict[str, Any]:
        """The other arguments of a search worth keeping for a value search: required ones, and choice
        arguments (set to the choice the worded searches used most) that do not just name a requested kind."""
        required = schema.get('required') or []
        choice = lambda k: isinstance((props.get(k) or {}).get('enum'), list)
        keep = {k: v for k, v in args.items() if k != field and (choice(k) or k in required)}
        cat_key = self.category_key()
        shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
        shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
        long = lambda text: {w for w in _stem(text) if len(w) >= 4}
        kind_word = lambda v: _words(v) in shelves or any((long(v) & long(s) for s in shelves if s))
        keep = {k: v for k, v in keep.items() if k in required or not isinstance(v, str) or not kind_word(v)}
        for k in [k for k in keep if choice(k)]:
            used = [_dump(a[k]) for a in worded if k in a]
            if used:
                best = max(used, key=used.count)
                keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
        return keep

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def title_noun(self, number: int) -> str:
        """The word of the item's name that the titles of rows of its kind carry most often."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        titles = [' '.join((str(v) for k, v in r.items() if isinstance(v, str) and re.search('title|name|heading', k, re.I))).lower() for r in self.rows.values() if not cat_key or self.owns(item, str(r.get(cat_key) or ''))]
        words = [w for w in re.findall('[a-z]{3,}', item.name.lower()) if w not in ('and', 'the', 'for', 'with')]
        if not words:
            return ''
        stem = lambda w: w[:-1] if len(w) > 4 and w.endswith('s') else w
        return max(words, key=lambda w: (sum((1 for t in titles if stem(w) in t)), words.index(w)))

    def learn_rule(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """The store's word rule from first-page worded searches the episode has made (no extra step): two queries
        that share their first two words return the same rows -> only those count; adding words cuts the hit count
        -> every word must match; adding words raises it -> any word matches. Returns a note the first time."""
        plan = self.search_template()
        paging = (self.shapes.get(name) or {}).get('next_arg')
        if not plan or plan[0] != name or (paging and args.get(paging) not in (None, 0, '', 1)):
            return ''
        words = tuple(str(args.get(plan[2]) or '').lower().split())
        if not words:
            return ''
        hits = next((v for k, v in observation.items() if isinstance(v, int) and (not isinstance(v, bool)) and re.search('hit|count|total|found|match|tally', k, re.I)), None) if isinstance(observation, dict) else None
        other = _dump(sorted(((k, _dump(v)) for k, v in args.items() if k not in (plan[2], paging))))
        found = self.records_in(observation)
        ids = tuple((i for i, _ in found))
        heads = [' '.join((str(v) for k, v in r.items() if isinstance(v, str) and re.search('title|name|heading', k, re.I))).lower() for _, r in found]
        stem = lambda w: w[:-1] if len(w) > 4 and w.endswith('s') else w
        # a word every shown title already carries leaves the rows unchanged whatever the rule: no evidence either way
        # (an every-word store answers "round toe wedge" and "round toe wedge pumps" alike when all rows are pumps)
        apart = lambda extra: any((any((stem(w) not in h for w in extra)) for h in heads))
        votes: dict[str, int] = {}
        for o_other, o_words, o_ids, o_hits in self.rule_log:
            if o_other != other or o_words == words:
                continue
            if len(words) >= 3 and len(o_words) >= 3 and o_words[:2] == words[:2] and o_words[2:] != words[2:]:
                if o_ids != ids:
                    votes['not first2'] = votes.get('not first2', 0) + 1
                elif apart(set(words[2:]) ^ set(o_words[2:])):
                    votes['first2'] = votes.get('first2', 0) + 1
            # one query extends the other at its end: fewer hits -> words narrow, more -> words widen; past the second
            # word, the same rows and count -> only the first two words count
            pair = (o_words, words, o_hits, hits, o_ids, ids) if words[:len(o_words)] == o_words else (words, o_words, hits, o_hits, ids, o_ids) if o_words[:len(words)] == words else None
            if pair and pair[2] is not None and pair[3] is not None:
                if pair[2] != pair[3]:
                    key = 'and' if pair[3] < pair[2] else 'or'
                    votes[key] = votes.get(key, 0) + 1
                if len(pair[0]) >= 2:
                    if pair[2] != pair[3] or pair[4] != pair[5]:
                        votes['not first2'] = votes.get('not first2', 0) + 1
                    elif apart(pair[1][len(pair[0]):]):
                        votes['first2'] = votes.get('first2', 0) + 1
        self.rule_log = (self.rule_log + [(other, words, ids, hits)])[-16:]
        # an extra word that changes nothing is also what a store matching any word shows, so reading only the first two
        # words needs a narrowing seen too; one pair sharing its first two words with other rows rules it out
        narrow, widen = (votes.get('and', 0), votes.get('or', 0))
        rule = 'or' if widen > narrow else 'first2' if narrow and votes.get('first2', 0) and (not votes.get('not first2', 0)) else 'and' if narrow > widen else None
        if not rule or rule == self.word_rule:
            return ''
        self.word_rule = rule
        self.trace.append({'kind': 'word_rule', 'rule': rule, 'votes': votes})
        if sum((1 for t in self.trace if t.get('kind') == 'word_rule')) > 3:
            return ''
        self.rule_told = True
        return '\n[Kyo, from the searches so far] ' + {'first2': 'This store reads only the first two words of a query; the rest change nothing. Put first the two words a fitting title would carry: the wanted value, then the product noun.', 'and': 'This store matches only rows that hold every query word, so each extra word cuts the rows. Use two or three words a fitting title would carry: the wanted value and the product noun.', 'or': 'This store matches rows holding any query word, so each extra word adds rows of other kinds. Use two or three words a fitting title would carry, the wanted value first.'}[rule]

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            # real races 11-14: 2-3 word searches whose first words a fitting title carries surfaced the paying
            # listing first most often (.21-.39 per call vs .06-.18 for 4+ words): one value, then the product noun
            noun = self.title_noun(number)
            head = ' '.join(fresh[0].split()[:2])
            tries = [f'{head} {noun}' if noun and noun not in head.lower() else head, head]
            for words in list(dict.fromkeys(tries))[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: words[:30].strip()}, self.schemas.get(tool), 'args', faults)
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
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Kyo searched each new requirement value with the product noun and read the first fitting-kind rows] " + '\n'.join(seen)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                return (number, left[:1])
        return (0, [])

    def list_labels(self, lane: list[str], used: set[str], answered: set[str], fronted: frozenset[str] | set[str]=frozenset()) -> list[str]:
        """The properties one list question names, without the ones already answered, at most LIST_KEYS. Its first
        LIST_FRONT places go to properties never named in a first place yet (those never named before, then those named
        only further back), in the lane's order; the rest to the best-ranked properties left, so a later list never
        repeats an earlier one."""
        pool: list[str] = []
        for label in lane:
            w = _words(label)
            if w in answered or re.fullmatch('options?', w) or any((_words(x) == w for x in pool)):
                continue
            pool.append(label)
        # real race 14 lists: a property named in places 1-3 drew its need .79 (first list), in places 4+ .43; one named
        # only further back before drew .58 when moved to the front and .33 when left behind
        front = ([x for x in pool if _words(x) not in used] + [x for x in pool if _words(x) in used and _words(x) not in fronted])[:LIST_FRONT]
        return (front + [x for x in pool if x not in front])[:LIST_KEYS]

    def item_rounds(self, lanes: dict[int, list[str]]) -> list[list[int]]:
        """When the messages left are fewer than the items with planned properties, the items in request order split
        as evenly as possible over those messages, so every item is asked once; otherwise none."""
        items = [n for n in sorted(lanes) if lanes[n]]
        free = self.msgs_free()
        if free <= 0 or len(items) <= free:
            return []
        size, extra = divmod(len(items), free)
        out: list[list[int]] = []
        for k in range(free):
            take = size + (1 if k < extra else 0)
            out.append(items[:take])
            items = items[take:]
        return [g for g in out if g]

    def items_text(self, pairs: list[tuple[int, str]]) -> str:
        return ' '.join((f"For the {self.items[n - 1].name}, what {re.sub('[_]+', ' ', label).strip()} do you want?" for n, label in pairs))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 3:
            item = self.items[number - 1]
            shelf = f' ({item.shelf})' if item.shelf and _words(item.shelf) != _words(item.name) else ''
            names = [re.sub('[_]+', ' ', label).strip() for label in labels]
            return f"For the {item.name}{shelf}, what {', '.join(names[:-1])} or {names[-1]} do you want?"
        if len(labels) >= 2:
            return f'What {labels[0]} or {labels[1]} would you like for the {self.items[number - 1].name}?'
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not self.catch_all(key)] + [key for key in keys if self.catch_all(key)]

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
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
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if _listy(values) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        second: dict[str, Any] | None = None
        if P1_ON:
            # the same PLAN prompt to both models: a property both choose is a surer first question than one only one picks
            plan = self.ask_json(prompt, [HEAVY], 3500)
            if plan is None and (not (self.broke or self.no_credit or self.calls >= ALL_CALLS)):
                plan = self.ask_json(prompt, [LIGHT], 3500)
            elif plan is not None and self.cost <= SPEND_SOFT:
                second = self.ask_json(prompt, [LIGHT], 3500)
            plan = plan or {}
        else:
            plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}

        def read_picks(got: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]], list[int]]:
            picked: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
            hints: dict[int, set[str]] = {n: set() for n in picked}
            order: list[int] = []
            for pick in _items(got.get('ask')):
                if not isinstance(pick, dict):
                    continue
                number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
                label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
                if not label or len(label.split()) > 5:
                    continue
                # the bare purchase-option field of a listing is no property of the product: the question would earn a shrug
                if P2_ON and re.fullmatch('options?', _words(label)):
                    continue
                if all((_words(label) != _words(other) for other in picked[number])):
                    picked[number].append(label)
                if str(pick.get('hint') or '').strip():
                    hints[number].add(_words(label))
                    if number not in order:
                        order.append(number)
            return (picked, hints, order)
        lanes, hinted, order = read_picks(plan)
        self.plan_order += [n for n in order if n not in self.plan_order]
        lanes2, hinted2, order2 = read_picks(second) if second else ({}, {}, [])
        self.plan_order += [n for n in order2 if n not in self.plan_order]
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
            hinted[number] = {_words(spelled(h)) for h in hinted[number] | hinted2.get(number, set())}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            if lanes2.get(number):
                other_lane = [_words(spelled(pick)) for pick in lanes2[number]]
                both = [x for x in lane if _words(x) in other_lane]
                extra: list[str] = []
                for pick in lanes2[number]:
                    if all((_words(spelled(pick)) != _words(x) for x in lane + extra)):
                        extra.append(spelled(pick))
                # then the first sample's other picks in its order, then the second's, the more carried key first
                extra.sort(key=lambda x: -share_of(x))
                lane = both + [x for x in lane if x not in both] + extra
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not self.catch_all(k))][:GUARD_KEYS]:
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
            lanes[number] = [x for x in lanes[number] if not self.catch_all(x)] + [x for x in lanes[number] if self.catch_all(x)]
            if P3_ON and count <= 2 and lanes[number] and _words(lanes[number][0]) in hinted[number]:
                self.measure_first(number, lanes[number], hinted[number], keys, table, max(1, count // len(self.items)))
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}, **({'ask2': _clip(second.get('ask'), 120)} if second else {})})
        return lanes

    def measure_first(self, number: int, lane: list[str], hinted: set[str], keys: list[str], table: list[tuple[str, str, int, list[str]]], room: int) -> None:
        """A hinted first key whose values are measures (a story about where the product goes reads as a size) gives the
        first message to the most widely carried unhinted key not settled and not already going out."""
        values = next((v for label, key, _, v in table if self.owner_of(label) == number and key == _words(lane[0])), [])

        def measure(vals: list[str]) -> bool:
            return bool(vals) and sum((1 for v in vals if re.match(MEASURE_VALUE, v))) * 2 > len(vals)
        if not measure(values):
            return
        shown = {key: v for label, key, _, v in table if self.owner_of(label) == number}
        going = {_words(x) for x in lane[:room]}
        swap = next((k for k in keys if _words(k) not in hinted and _words(k) not in going and (not measure(shown.get(k, []))) and (not self.catch_all(k))), None)
        if swap is None:
            return
        lane[:] = [swap] + [x for x in lane if _words(x) != _words(swap)]
        self.trace.append({'kind': 'measure_first', 'item': number, 'first': swap, 'moved': lane[1]})

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.items[number - 1].name}'
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

    def written_budget(self, value: float, said: str) -> Any:
        """A limit the reader returned at another decimal scale than the figure the answers write (4844 read as 48.44)
        is the written figure, when exactly one figure is that shift of it and it lies nearer the limit in force. A
        limit no answer writes at all is the reader restating the request in another unit: the limit in force stays."""
        figures = {float(f.replace(',', '')) for f in re.findall('\\d[\\d,]*(?:\\.\\d+)?', said)} - {0.0}
        if any((abs(f - value) <= 1e-09 * max(1.0, f) for f in figures)):
            return value
        if not figures:
            self.trace.append({'kind': 'budget_unwritten', 'read': value})
            return self.budget
        shifted = [f for f in figures if any((abs(f / (value * 10 ** k) - 1) < 0.002 for k in (-3, -2, -1, 1, 2, 3)))]
        old = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) and (self.budget > 0) else None

        def off(x: float) -> float:
            return max(x / old, old / x) if old else 1.0
        if not shifted:
            self.trace.append({'kind': 'budget_unwritten', 'read': value})
            return self.budget
        if len(shifted) != 1 or (old and off(shifted[0]) >= off(value)):
            return value
        self.trace.append({'kind': 'budget_written', 'read': value, 'written': shifted[0]})
        return int(shifted[0]) if shifted[0].is_integer() else shifted[0]

    def take_heard(self, heard: dict[str, Any], told: list[str], said: str='') -> None:
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
            self.budget = self.written_budget(heard['budget'], said)

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({f'about item {n}' if n else 'about the items it names'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Do the steps the request asks for at this stage (a message to the shopper, a store call) before the
        ending goes out. Each round a model reads the record and names the call for a step not done yet; Kyo
        runs it. A reply that changes the task holds the ending. Returns (go on, what Kyo did)."""
        due = [p for p in self.process if p['when'] == stage]
        if not due or self.finished or (not self.gate_due(stage)):
            return (True, '')
        done: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            got = self.ask_json(self.gate_prompt(stage, due), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if done:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                news = '; '.join(told + ([hold] if hold else []))
                return (False, (self.gate_said(done) + '\n' if done else '') + f"The shopper's reply changes the task: {news}. Fix the cart for it, then check out again.")
            if not call and (not done) and (not nudged) and self.gate_unproven(stage, due, got):
                nudged = True
                continue
            line = self.gate_call(stage, call)
            if line is None:
                break
            done.append(line)
        return (True, self.gate_said(done) if done else '')

    @staticmethod
    def gate_said(done: list[str]) -> str:
        return 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(done)

    def end_tools(self) -> set[str]:
        return {tool for tool in (self.roles.get('buy'), self.roles.get('refuse')) if tool}

    def gate_prompt(self, stage: str, due: list[dict[str, Any]]) -> str:
        buying = stage == 'order'
        record = '\n'.join((f'{n}. {line}' for n, line in enumerate(self.done_log, 1))) or '(nothing yet)'
        notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else ''
        parts = [GATE.replace('STAGE', 'place the order' if buying else 'end without a purchase'), 'REQUEST:\n' + self.request, 'STEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in due)), 'LEDGER:\n' + self.ledger(), 'CART NOW: ' + self.cart_text(), 'RECORD (messages and step calls so far):\n' + record + notes, f"MESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if buying else 'ending')} itself takes one)", 'TOOLS:\n' + self.tool_lines(self.end_tools())]
        return '\n\n'.join(parts)

    def gate_unproven(self, stage: str, due: list[dict[str, Any]], got: dict[str, Any]) -> bool:
        """Steps the gate model calls done that no record line backs (for a step owing a message: no line made
        since the cart last changed). While a message and two steps are left, the record says so and the
        model is asked again for the call that does them."""
        if not self.roles.get('ask') or self.msg_used >= self.msg_limit or self.left() < 2:
            return False
        talk = stage == 'order' and any((p['messages'] > 0 for p in due))
        backed = {n for n in range(1, len(self.done_log) + 1) if not talk or (n <= len(self.done_steps) and self.done_steps[n - 1] > self.cart_at)}
        claims = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
        unproven = [e for e in claims if e.get('done') is not True or not backed.intersection(((_count(n) or 0) for n in _items(e.get('lines'))))]
        if not unproven:
            return False
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        names = '; '.join((str(e.get('step') or '') for e in unproven))[:200]
        self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows {names} done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)
        return True

    def gate_call(self, stage: str, call: dict[str, Any] | None) -> str | None:
        """Run the call the gate model named and return its record line; None ends the gate (no call: the
        stage is clear for this cart; or the call cannot run)."""
        if not call:
            self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
            return None
        name = str(call.get('tool') or '')
        talk = name == self.roles.get('ask')
        if name not in self.schemas or name in self.end_tools() or self.left() < 2 or (talk and self.msg_used >= self.msg_limit):
            return None
        faults: list[str] = []
        args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
        if faults:
            return None
        if talk:
            text = next((v for v in args.values() if isinstance(v, str)), '')
            sent = self.send([(1, '', text)], 1)
            if not sent:
                return None
            self.gate_msgs += 1
            return f'Q: {text}\nA: {sent[0][2]}'
        reply = self.act(name, args)
        if reply is None:
            return None
        seen = reply.get('observation')
        self.absorb(seen)
        self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        return f'{name} {_dump(args)} -> {self.show(seen)[:400]}'

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
        """A store note, read by a model. Listings it calls gone are marked unavailable; a note that changes the
        task changes the budget and the requirements. Returns what changed, for the model's next turn."""
        self.heard.update(lines)
        got = self.ask_json(NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines), [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = self.note_budget(got.get('budget')) + self.note_needs(got)
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and self.items:
            self.repick.update((min(max(1, _count(e.get('item')) or 1), len(self.items)) for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()))
        # A lower budget leaves the cheapest fit the cheapest while its line stays inside it (real races 5-14: a
        # still-fitting pick kept after a budget note passed .93, dropped .67), so only a line it breaks reopens the hunt.
        held = self.cart_ids()
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        broken = not held or cap is None or any((self.price_of(i) is None for i in held)) or sum((self.price_of(i) * max(1, self.held_units(i)) for i in held)) > cap
        narrower = gone or self.note_sold or any((c.startswith('dropped ') or (c.startswith('budget now') and broken) for c in changed))
        if narrower and self.brief.get('pick') == 'cheapest':
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return f"{summary} ({'; '.join(changed)})".strip() if changed else summary

    def note_budget(self, budget: Any) -> list[str]:
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0 or budget == self.budget:
            return []
        self.budget = budget
        return [f'budget now {budget}']

    def item_of(self, entry: dict[str, Any]) -> Item:
        """The item an entry's 1-based number names (clamped to the item list)."""
        return self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]

    def note_needs(self, got: dict[str, Any]) -> list[str]:
        """Requirements a note drops (matched exactly, else by property name) or adds."""

        def entries(key: str) -> list[tuple[Item, str]]:
            return [(self.item_of(e), str(e['need']).strip()) for e in _items(got.get(key)) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        changed: list[str] = []
        for item, need in entries('drop'):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            head = _words(need.split(':')[0])
            hit = [m for m in live if m == need] or [m for m in live if _words(m.split(':')[0]) == head][:1]
            item.dropped.extend(hit or [need])
            changed.append(f'dropped {need}')
        for item, need in entries('add'):
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f'added {need}')
        return changed

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any, head: bool=True) -> str:
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
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]) + (' [not the stated value: its words stand apart or in another order there]' if self.apart_needs(ident, [need]) else ''))
                parts += [tag + said for said in self.form_said(ident, need)]
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return ("[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " if head else '') + ' | '.join(lines)

    def form_said(self, ident: str, need: str) -> list[str]:
        """Attributes of this listing that state a plain required value in another writing (joined, hyphenated,
        split or another form of the same words), unless some attribute states it as written or only apart."""
        prop, _, value = need.partition(':')
        value = value.strip()
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not value or not block or self.apart_needs(ident, [need]):
            return []
        if any((_plain_words(value) == _plain_words(part) for v in block.values() for part in [str(v)] + re.split('[,/;|]', str(v)))):
            return []
        hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _form_match(value, v)]
        if hits and (ident, need) not in self.form_seen:
            self.form_seen.add((ident, need))
            self.trace.append({'kind': 'form_match', 'id': ident, 'need': need, 'hits': hits[:2]})
        return [f'{prop.strip()} "{value}" -> ' + ', '.join(hits[:2]) + ' [the same words as the stated value, written joined, hyphenated or in another form of the same word: this attribute states that value]'] if hits else []

    def form_lines(self, ids: list[str]) -> list[str]:
        many = len(self.items) > 1
        out = []
        for ident in ids:
            said = [(f'(item {n}) ' if many else '') + line for n, need in self.needs_now()[:8] for line in self.form_said(ident, need)]
            if said:
                out.append(f'{ident}: ' + '; '.join(said))
        return out

    def apart_needs(self, ident: str, needs: list[str] | None=None) -> list[str]:
        """Plain requirements this listing's attributes hold only apart: some key naming the property holds all
        the value's words with others between or in another order, and none holds them as the phrase."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        out = []
        for need in needs if needs is not None else [n for _, n in self.needs_now()]:
            prop, _, value = need.partition(':')
            shapes = [_spread(value.strip(), v) for k, v in block.items() if _names(_stem(prop), _stem(k))]
            if 'apart' in shapes and 'phrase' not in shapes:
                out.append(need)
        return out

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def fit_text(self, ident: str) -> str:
        """A record's plain text for fit codes: its own strings, then each nested block's key and value pairs."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        flat = [str(v) for v in record.values() if isinstance(v, str)]
        nested = [f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)]
        return ' '.join(flat) + ' ' + '. '.join(nested)

    def fit_index(self) -> dict[str, Any]:
        """Fit marks of every listing seen, and for each part code the device codes other pages say it fits."""
        marks = {ident: _fit_marks(self.fit_text(ident)) for ident in self.seen_order}
        notes: dict[str, set[str]] = {}
        for part, codes in (pair for mark in marks.values() for pair in mark['also']):
            notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

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
            owns = {m.get('own') for m in index['marks'].values()}
            for line in parts:
                fits = self.fit_set(line, index)
                rest = (index['marks'].get(line) or {}).get('more')
                if fits and (not fits & codes) and rest and rest not in owns:
                    # its own page leaves the device out and sends the rest of its list to a page not seen: a seen
                    # part that names the device on its own page is the surer pick (real race 13: orders whose part
                    # fit only through such a page paid 3 of 86, parts naming the device 13 of 94)
                    item = self.items[part - 1]
                    needs = [m for m in item.musts + item.told if m not in item.dropped]
                    # a plain word-valued requirement whose property an attribute of the candidate names, with none of
                    # the value's words in any such attribute, rules that candidate out
                    plain = [(_stem(n.partition(':')[0]), set(_plain_words(n.partition(':')[2])) - {'or', 'either', 'and'}) for n in needs if not re.search('\\d|\\b(?:not|no|other than|unlike|same|than|like|only)\\b', n.partition(':')[2], re.I)]
                    def against(ident: str) -> bool:
                        block = _fact_block(self.full.get(ident) or {}) or {}
                        return any(words and held and not any(words & set(_plain_words(v)) for v in held) for prop, words in plain for held in [[v for k, v in block.items() if _names(prop, _stem(k))]])
                    named = [i for i in self.item_ids(part) if i in self.full and self.fit_set(i, index) & codes and self.buyable(i) and i not in self.gone and i not in self.dropped and (not self.apart_needs(i, needs)) and (not against(i)) and (i not in cart)][:4]
                    if named:
                        faults.append(f"{line} ({item.name}) does not name {devices[0]} ({', '.join(sorted(codes))}) on its own page, which lists {', '.join(sorted(fits)[:4])} and sends the rest to part {rest}; seen parts of that kind whose own page names it: {', '.join(named)}. Keep the device and take one of them in place of this part when it meets the item's other requirements")
                    continue
                if fits and (not fits & codes):
                    good = [i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    hosts = [i for i in self.item_ids(main) if set(index['marks'].get(i, {}).get('codes') or []) & fits and (not self.sold_out(i))][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)) + (f"; seen {self.items[main - 1].name} listings this part works with: {', '.join(hosts)}" if hosts else ''))
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

    def line_faults(self) -> list[str]:
        """What the cart's held lines get wrong in code: an item without a line or with several, a wrong
        quantity, a line of a kind nobody asked for, a sold-out line, a total over the budget."""
        cat_key = self.category_key()
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not held or not cat_key:
            return []
        kind_of = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in held}
        claims = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, kind_of[i])] for i in held}
        labels = {str(r.get(cat_key) or '') for r in self.rows.values()}
        listed = [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels))]
        out: list[str] = []
        for n in listed:
            item = self.items[n - 1]
            mine = [i for i in held if n in claims[i]]
            only = [i for i in mine if claims[i] == [n]]
            if not mine:
                out.append(f'item {n} ({item.name}) has no line in the cart')
            elif len(only) > 1:
                out.append(f"item {n} ({item.name}) has {len(mine)} lines ({', '.join(mine)}); keep one")
            out += [f'{i} is held at {self.held_units(i)} unit(s); item {n} ({item.name}) wants {item.qty}' for i in only if self.held_units(i) != item.qty]
        if len(listed) == len(self.items):
            out += [f"{i} ({kind_of[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in held if not claims[i]]
        out += [f'{i} is sold out' for i in held if self.sold_out(i)]
        return out + self.budget_faults(held)

    def budget_faults(self, held: list[str]) -> list[str]:
        cost = {i: self.price_of(i) for i in held}
        cap = self.budget
        if not isinstance(cap, (int, float)) or None in cost.values():
            return []
        if self.brief.get('budget_scope') == 'each':
            return [f'{i} costs {cost[i]:g}, over the {cap:g} limit for each item' for i in held if cap < cost[i] <= 10 * cap]
        total = sum((cost[i] * self.held_units(i) for i in held))
        return [f'the total is {total:g}, over the budget of {cap:g}'] if cap + 1e-09 < total <= 10 * cap else []

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

    def cart_fields(self, name: str) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None, str | None]:
        """A cart tool's schema, properties, and its listing-id, units and mode argument names."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        specs = [(k, v) for k, v in props.items() if isinstance(v, dict)]
        texts = [k for k, v in specs if 'string' in _kinds(v) and 'enum' not in v]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in specs if 'integer' in _kinds(v)), None)
        mode = next((k for k, v in specs if isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
        return (schema, props, target, count, mode)

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        """Set wrong line quantities with the cart tool itself: one setting call, or a removal and a fresh add
        when the tool only adds; nothing when the calls would not leave room to finish."""
        name = self.roles.get('cart') or ''
        schema, props, target, count, mode = self.cart_fields(name)
        if not name or not target or (not count):
            return []
        sets = self.cart_sets(name)
        if not sets and (not mode) and wrong:
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if sets:
                calls.append({target: ident, count: want})
                continue
            options = [str(o) for o in props[mode]['enum']]
            off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
            on = next((o for o in options if o != off), options[0])
            calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
        if len(calls) + 3 > self.left():
            return []
        done: list[str] = []
        for args in calls:
            faults: list[str] = []
            shaped = _shape(args, schema, 'args', faults)
            reply = None if faults else self.act(name, shaped)
            if reply is None or self.finished:
                break
            seen = reply.get('observation')
            self.note_cart(name, shaped)
            self.cart_said = _dump(seen)
            self.observe(name, seen)
            done.append(_dump(shaped) + ' -> ' + self.show(seen)[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not ids:
            return ''
        apart = [(i, need) for i in ids for need in self.apart_needs(i)] if self.left() > self.tail() + 4 else []
        if apart:
            self.checked.add(key)
            self.verdicts[key] = False
            self.leads = [i for i in self.leads if i not in ids]
            self.trace.append({'kind': 'check_apart', 'cart': ids, 'needs': apart[:4]})
            said = [f"{i}: its {need.split(':', 1)[0].strip()} holds the words of \"{need.split(':', 1)[1].strip()}\" only apart or in another order, so it is another value" for i, need in apart[:3]]
            return 'A careful review says this cart is not right yet: ' + '; '.join(said) + '. Find a listing whose attribute states that value as written (search with the value and the product noun), put it in the cart instead, then check out.'
        if not self.judges(limit):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        forms = self.form_lines(ids)
        # A requirement read from other pages can ask for a value related to the one there (by a rule those
        # pages state): the review sees those pages, not only the value Kyo copied from the last one.
        live = {need for _, need in self.needs_now()}
        linked = list(dict.fromkeys((i for need, chain in self.ref_pages.items() if need in live for i in chain if i in self.full and i not in ids)))[-4:]
        if linked:
            prompt += "\n\nRECORDS OF THE PAGES THOSE REQUIREMENTS POINT AT (the chain in order; the requirement's own words say how the cart's value must relate to the value there, by any rule these pages state):\n" + '\n'.join((f'{i}: ' + self.record_of(i) for i in linked))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
            forms += self.form_lines([i for i in self.full if i not in ids and (not self.sold_out(i))])
        if forms:
            prompt += '\n\nWORD FORMS (Kyo read these attributes as stating a required value in another writing):\n' + '\n'.join(forms[:8])
        got = self.ask_json(prompt, self.judges(limit), 3000)
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

    def trail(self, room: int=6) -> tuple[str, bool]:
        """Follow requirements whose value sits on another listing, round by round: a model reads the records
        into chains of links; Kyo opens the listings a chain needs next, or searches for one it has no row
        of. Returns the pages and values found, and whether an unfinished chain holds the order (a few
        times). No round starts after the clock stop. Rounds count over the episode, up to room."""
        pending = self.pointers_open()
        tool = self.offer_tool()
        target = self.single_target(tool)
        pages: list[str] = []
        lost: list[str] = []
        rounds = min(room, 6 if self.left() > 15 else 4)
        while pending and self.trails < rounds and self.cost < SPEND_HARD and (not self.late()):
            self.trails += 1
            models = [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else [])
            got = self.ask_json(self.trail_prompt(pending), models, 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach, lost, gaps = self.read_chains(got, pending)
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                known = self.trace[-1] if self.trace[-1].get('kind') == 'link_known' else None
                if known:
                    pages.append(f"{known['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished:
                break
            pages += self.open_links(tool, target, reach[:max(0, min(3, 6 - len(pages)))])
        text = '[Kyo opened the pages a requirement points at]\n' + '\n'.join(pages) + '\n' if pages else ''
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        if not lost or self.trail_blocks >= (3 if self.left() > 15 else 2) or self.left() <= 5:
            return (text, False)
        self.trail_blocks += 1
        return (text + 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out.", True)

    def trail_prompt(self, pending: list[str]) -> str:
        newest = list(reversed(self.seen_order))
        kept = [self.full.get(i) or self.rows.get(i) for i in newest if i in self.opened]
        records = ['OPENED ' + _dump(_clip(record, 300))[:1400] for record in kept if record][:45]
        for ident in newest:
            if len(records) >= 60:
                break
            if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
        return TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n' + '\n'.join(records)

    def read_chains(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, tuple[str, ...]]]]:
        """Sort the model's chains: a walked chain gives its value; one with an unopened row says what to open
        next; one with a link Kyo has no row of is lost, with what to search for it."""
        reach: list[str] = []
        lost: list[str] = []
        gaps: list[tuple[str, str, tuple[str, ...]]] = []
        for chain in _items(got.get('chains')):
            if not isinstance(chain, dict):
                continue
            named = str(chain.get('need') or '').strip()
            need = named if named in pending else pending[0] if len(pending) == 1 else None
            if not need:
                continue
            links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
            ids = [str(l.get('id') or '').strip() for l in links]
            value = str(chain.get('value') or '').strip()
            if links and all((i in self.opened for i in ids)) and (chain.get('done') is True and value or not str(links[-1].get('find') or '').strip()):
                self.ref_values[need] = value or f'as stated on {ids[-1]}'
                self.ref_pages[need] = ids
                self.version += 1
                continue
            closed = [i for i in ids if i in self.rows and i not in self.opened]
            if closed:
                reach.extend((i for i in closed if i not in reach))
                continue
            gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
            about = str(gap.get('about') or '') or 'the listing its value comes from'
            words = str(gap.get('find') or '').strip()
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
            gaps.append((about, words, tuple((i for i in ids if i))))
        return (reach, lost, gaps)

    def open_links(self, tool: str, target: str, ids: list[str]) -> list[str]:
        """Open each listing with the page tool; stops at the first call that cannot run."""
        shown: list[str] = []
        for ident in ids:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            if reply is None or self.finished:
                break
            self.observe(tool, reply.get('observation'))
            shown.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        return shown

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

    def find_link(self, about: str, words: str, before: tuple[str, ...]=()) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        # an earlier link of the chain is the page that names this one: its words describe the link, it is not it
        for ident in [i for i in self.opened_shop(about) if i not in before]:
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
            args = _shape({**base, field: self.kind_last(tool, field, words, named)}, self.schemas.get(tool), 'args', faults)
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

        # a row whose summary leaves out the values is still of the link's kind when its category label is
        # one the link's description names; the search's own order then ranks it
        cat_key = self.category_key()
        told = _roots(re.sub('[^a-z0-9]+', ' ', (about + ' ' + words).lower()))

        def labelled(ident: str) -> bool:
            label = _roots(re.sub('[^a-z0-9]+', ' ', str((self.rows.get(ident) or {}).get(cat_key) or '').lower())) if cat_key else set()
            return bool(label) and label <= told

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower())) or labelled(i)]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def kind_last(self, tool: str, field: str, words: str, shop: set[str]) -> str:
        """Search words for a field that keeps only its first characters: the shop's name and the values go
        ahead of the words that only name the kind of product, which Kyo checks on the rows itself."""
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(field)
        cap = spec.get('maxLength') if isinstance(spec, dict) else None
        cat_key = self.category_key()
        if not isinstance(cap, int) or len(words) <= cap or not cat_key:
            return words
        kinds = {w for row in self.rows.values() for w in _roots(re.sub('[^a-z0-9]+', ' ', str(row.get(cat_key) or '').lower()))}

        def kind_only(token: str) -> bool:
            low = re.sub('[^a-z0-9]+', ' ', token.lower())
            return bool(_roots(low)) and _roots(low) <= kinds and (not set(re.findall('[a-z0-9]{3,}', low)) & shop)
        tokens = words.split()
        return ' '.join([t for t in tokens if not kind_only(t)] + [t for t in tokens if kind_only(t)])

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
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        """A sold-out cart line's replacements, as a review model finds them among the other records seen
        (cheapest first when the pick is the cheapest), at most 4."""
        if not self.judges():
            return []
        pool = [i for i in reversed(self.seen_order) if i not in lost and i not in self.gone and (not self.sold_out(i))][:25]
        if not pool:
            return []
        records = [_dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for i in pool]
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        taken = set(lost) | set(self.gone) | set(self.cart_ids())
        for entry in _items((got or {}).get('fits')):
            ident, price = (str(entry.get('id')), entry.get('price')) if isinstance(entry, dict) else (str(entry), None)
            if ident not in self.rows or ident in taken:
                continue
            taken.add(ident)
            fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        order = sorted(fits, key=lambda f: f[0]) if self.brief.get('pick') == 'cheapest' else fits
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return [i for _, i in order][:4]

    def audit(self, keep: int | None=None) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows, keep):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        # Read pages are the only records the review can pass, and unread rows are the only ones it
        # can send Kyo to open, so each kind keeps its own room and neither crowds out the other.
        order = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        pages = [i for i in order if i in self.full][:20]
        rows = [i for i in order if i not in self.full][:30]
        records = ['OPENED ' + _dump(_clip(self.full[i], 200))[:1200] for i in pages]
        records += ['ROW ' + _dump(_clip(self.rows.get(i), 120))[:500] for i in rows]
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (read pages first, then rows; newest first in each):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """The review's fitting listings per item number. The review sometimes keys them by the item's name
        ("tablets") instead of its number; those are mapped to the item they name, so the leads can be bought."""
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for key, ids in fits.items():
            number = self.item_number(key)
            # a key naming no item still gives leads, under its own words, as before
            slot = named.setdefault(str(number) if number else str(key), [])
            slot += [str(i) for i in _items(ids) if str(i) in self.rows and (not self.sold_out(str(i))) and str(i) not in slot]
            del slot[4:]
        return {k: v for k, v in named.items() if v}

    def item_number(self, key: Any) -> int:
        """The requested item a key names: its 1-based number, or a name or store label that is the item's own;
        the only item when there is one; 0 when none."""
        number = _count(key)
        if number is None:
            found = re.findall('\\d+', str(key))
            number = int(found[0]) if len(found) == 1 and re.fullmatch('\\s*(?:item\\s*)?#?\\s*\\d+\\s*', str(key), re.I) else None
        if number is not None:
            return number if 1 <= number <= len(self.items) else 0
        said = _words(str(key))
        for n, item in enumerate(self.items, 1):
            if said and (said in (_words(item.name), _words(item.shelf or '')) or self.owns(item, str(key))):
                return n
        return 1 if len(self.items) == 1 else 0

    def open_rows(self, rows: list[str], keep: int | None=None) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            # one at a time, the review reads 4 rows while steps and the clock are plentiful (an unread row may be
            # the fit the ending would forgo), else 2
            room = 4 if self.left() >= 15 and self.spent() <= CLOCK_STOP - 60 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
        else:
            return False
        if keep is not None:
            # a late review opens only what still leaves its cart lines and the order room
            plan = plan[:max(0, self.left() - keep)]
        if not plan or self.left() < len(plan) + (5 if keep is None else keep):
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
        """The cart calls that leave exactly these lines (and take the dropped ones off), or None when this
        store's cart tool cannot express that."""
        name = self.roles.get('cart') or ''
        _, props, target, count, mode = self.cart_fields(name)
        whole = self.whole_cart_call(props, lines)
        if whole is not _NO:
            return whole
        if not target:
            return None
        units = lambda n: {count: n} if count else {}
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **units(have)})
                continue
            if not count:
                return None
            if self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif isinstance(props[count].get('minimum'), int) and props[count]['minimum'] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **units(want)})
        return calls

    def whole_cart_call(self, props: dict[str, Any], lines: dict[str, int]) -> Any:
        """A cart tool taking a list of line objects gets the whole cart in one call (None when its line has no
        id field); _NO when the tool takes no such list."""
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if not (isinstance(shape, dict) and isinstance(shape.get('properties'), dict)):
                continue
            inner = shape['properties']
            id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
            count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
            if not id_key:
                return None
            return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        return _NO

    def complete_cart(self) -> str:
        """Once an independent check confirmed a lead, Kyo builds that cart itself and checks out instead of
        ending without a purchase. Returns the store's replies, or '' when it did not happen."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        lines, drop = self.lead_cart()
        calls = self.cart_calls(lines, drop) if lines else None
        if not calls or len(calls) + 1 > self.left():
            return ''
        ids = list(lines)
        self.fill_facts(ids, len(calls) + 3)
        if not self.passes(ids):
            return ''
        placed = self.place(cart, calls)
        replies = self.buy_confirmed(buy, ids) if placed else []
        if not replies:
            return ''
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ', so Kyo set the cart (' + '; '.join(placed) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def lead_cart(self) -> tuple[dict[str, int], list[str]]:
        """The lines to buy (the review's first buyable pick per item, plus cart lines of kinds not covered
        yet) and the cart lines to take off; nothing when the lines miss an item or the confirmed lead."""
        key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(key) or '').strip().lower() if key else ident
        picks: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number) or 0
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in picks}:
                picks[pick] = self.items[index - 1].qty
        if self.confirmed not in picks:
            return ({}, [])
        covered = {kind(p) for p in picks}
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in picks and (not self.sold_out(i)) and (kind(i) not in covered)}
        if len(picks) + len({kind(i) for i in keep}) < len(self.items):
            return ({}, [])
        return ({**keep, **picks}, [i for i in now if i not in picks and i not in keep])

    def passes(self, ids: list[str]) -> bool:
        """The independent check of these listings as the cart (it runs once per cart and version)."""
        self.check_cart(ids, SPEND_HARD)
        return bool(self.verdicts.get(self.check_key(ids)))

    def place(self, cart: str, calls: list[dict[str, Any]]) -> list[str]:
        """Run the cart calls; the calls made, or [] when one could not run."""
        placed: list[str] = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            reply = None if faults else self.act(cart, args)
            if reply is None or self.finished:
                return []
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            placed.append(_dump(args))
        return placed

    def buy_confirmed(self, buy: str, ids: list[str]) -> list[str]:
        """Check out the confirmed cart; once more after a refusal when the re-read lines still pass."""
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return []
        replies: list[str] = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                if not self.passes(ids) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in ids)):
                    break
            go, said = self.gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': 'completed', 'lines': ids, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        return replies

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
        if keep is not None and self.read_room():
            # An item without a cart line may need one listing read before it is carted; the narrowed tools left only
            # the ending (Arena r11/r14: the model stalled there and the episode ended without a purchase).
            keep.add(self.facts_tool())
            if self.left() >= self.tail() + 2 + self.read_room():
                # Arena b27-b32: 8 of 17 endings without a purchase on order tasks came after turns without a call
                # while the narrowed tools held no search
                keep.add(self.roles.get('search'))
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def read_room(self) -> int:
        """Items with no cart line, when one read, a cart call for each of them and the ending still fit."""
        missing = max(0, len(self.items) - len(self.cart_ids()))
        return missing if missing and self.facts_tool() and self.left() >= self.tail() + 1 + missing else 0

    def fit_picks(self) -> dict[int, str]:
        """Per item, the first listing the review found meeting every requirement of it (or, without a review,
        the cart line the agent put in for it)."""
        cat_key = self.category_key()
        carted = [i for i in self.cart_ids() if cat_key]
        picks: dict[int, str] = {}
        for number, item in enumerate(self.items, 1):
            mine = [i for i in carted if self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]
            pick = next((i for i in self.lead_items.get(str(number), []) + mine if i in self.rows and (not self.sold_out(i)) and (not self.apart_needs(i))), None)
            if pick:
                picks[number] = pick
        return picks

    def offer_note(self, room: int) -> str:
        """When the request asks to be offered what comes closest if nothing fits, the remark of an ending
        without a purchase names, by id, the first listing the review found meeting every requirement of its
        item (or, without a review, the cart line the agent put in for that item); a listing neither found
        fitting is never named. When no item has one, it names for each item the cheapest of the listings
        that miss the least (stock and budget counted), which any price margin the request allows
        above the cheapest of those always admits. '' when there is none."""
        if self.brief.get('quiet_decline') is not False:
            return ''
        picks = self.fit_picks()
        named = [f'{picks[n]} for the {item.name}' for n, item in enumerate(self.items, 1) if n in picks]
        missing = [item.name for n, item in enumerate(self.items, 1) if n not in picks]
        if not named:
            return self.closest_note(room)
        while named:
            text = 'Ending with no purchase: no listing here satisfies the full order. Nearest matches: ' + '; '.join(named) + ' (each one satisfies its own item).'
            more = ' No ' + ' or '.join(missing) + ' listing I read satisfies all of its requirements.' if missing else ''
            if len(text) + len(more) <= room:
                return text + more
            if len(text) <= room:
                return text
            named.pop()
        return ''

    def rank_closest(self, only: list[str] | None=None) -> dict[int, list[tuple[int, float, str, list[str]]]]:
        """Per item, its read listings as (misses counted, price, id, misses), least missed then cheapest
        first. A model lists the attribute requirements each record does not show met; Kyo adds one for a
        listing out of stock and one for a line over the budget. With only, those pages are judged and
        merged into the earlier ranking."""
        cat_key = self.category_key()
        ranked: dict[int, list[tuple[int, float, str, list[str]]]] = {n: list(rows) for n, rows in (self.closest or {}).items()} if only is not None else {}
        pool = [i for i in (only if only is not None else self.full) if i in self.full and i in self.rows]
        pages = sorted(pool, key=lambda i: (self.price_of(i) if self.price_of(i) is not None else float('inf')))[:16]
        models = self.judges(SPEND_HARD)
        if not pages or not models or not self.items:
            return self.closest or {}
        records = [('SOLD OUT ' if not self.buyable(i) else '') + _dump(_clip(self.full[i], 200))[:1000] for i in pages]
        prompt = CLOSEST + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nLISTING RECORDS:\n' + '\n'.join(records)
        got = self.ask_json(prompt, models, 8000)
        self.closest_ids.update(pages)
        for entry in _items((got or {}).get('listings')):
            if not isinstance(entry, dict) or str(entry.get('id')) not in pages:
                continue
            ident = str(entry.get('id'))
            number = _count(entry.get('item')) or (1 if entry.get('item') is None and len(self.items) == 1 else 0)
            label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
            if number < 1 or number > len(self.items) or (label and self.owner_of(label) not in (0, number)):
                continue
            ranked[number] = [r for r in ranked.get(number, []) if r[2] != ident]
            misses = [str(m) for m in _items(entry.get('misses'))]
            price = self.price_of(ident)
            if not self.buyable(ident):
                misses.append('in stock')
            if price is not None and isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) and (price * max(1, self.items[number - 1].qty) > self.budget):
                misses.append('budget')
            ranked[number].append((len(misses), price if price is not None else float('inf'), ident, misses))
        for rows in ranked.values():
            rows.sort(key=lambda r: (r[0], r[1]))
        self.closest = ranked
        self.trace.append({'kind': 'closest', 'got': {n: [r[:3] for r in rows[:4]] for n, rows in ranked.items()}})
        return ranked

    def closest_note(self, room: int) -> str:
        fresh = [i for i in self.full if i not in self.closest_ids]
        # the ranking is a long model call: near the clock it could outlast the episode and lose the ending itself
        if self.late() or self.spent() > CLOCK_STOP - 40:
            ranked = self.closest or {}
        else:
            ranked = self.rank_closest() if self.closest is None else self.rank_closest(fresh) if fresh else self.closest
        unit = f" {self.brief.get('budget_unit')}" if self.brief.get('budget_unit') else ''
        named = []
        for number, item in enumerate(self.items, 1):
            rows = ranked.get(number) or []
            if not rows:
                continue
            short, price, ident, misses = rows[0]
            cost = f'{price:g}{unit}, ' if price != float('inf') else ''
            named.append(f"{ident} for the {item.name} ({cost}it misses only: {'; '.join(misses)[:120]})" if short else f'{ident} for the {item.name} ({cost}the closest I found)')
        while named:
            text = 'Ending with no purchase: no listing satisfies every requirement. Nearest matches: ' + '; '.join(named) + '.'
            if len(text) <= room:
                return text
            named = [n.split(' (', 1)[0] for n in named] if any((' (' in n for n in named)) else named[:-1]
        return ''

    def closest_hunt(self) -> None:
        """Before an ending without a purchase that must offer the closest listing (no item has a fitting one),
        rank the read listings, then read the cheapest unread rows of the same kind priced below the cheapest
        of those that miss the least, after one listing of that kind under that price with a
        price-capped search the store has answered, and rank again: rows come in no price order, and the
        offer is judged by price among the closest."""
        if self.hunted or self.brief.get('quiet_decline') is not False or self.fit_picks() or self.finished or self.spent() > CLOCK_STOP - 60:
            return
        self.hunted = True
        if not self.price_key:
            # the offer is judged by price and stock among the closest: learn where this store's rows keep
            # them, and which searches capped the price, from the searches made so far
            for name, args, seen in self.search_log:
                self.read_shape(name, seen)
            for name, args, seen in self.search_log:
                self.learn_cap(name, args, seen)
        ranked = self.rank_closest()
        cat_key = self.category_key()
        read: list[str] = []
        for number, rows in sorted(ranked.items()):
            best = [r for r in rows if r[0] == rows[0][0] and r[1] != float('inf')]
            if not best or not cat_key or self.left() < self.tail() + 5 or self.spent() > CLOCK_STOP - 30:
                continue
            floor = min((r[1] for r in best))
            kind = str((self.rows.get(best[0][2]) or {}).get(cat_key) or '').strip().lower()
            calls = [c for c in self.cap_calls if kind and kind in c[3]]
            if calls:
                tool, key, args, _ = min(calls, key=lambda c: len(c[2]))
                paging = (self.shapes.get(tool) or {}).get('next_arg')
                faults: list[str] = []
                call = _shape({**{k: v for k, v in args.items() if k != paging}, key: int(floor) if float(floor).is_integer() else floor}, self.schemas.get(tool), 'args', faults)
                if not faults and tool + _dump(call) not in self.read_cache:
                    reply = self.act(tool, call)
                    if reply is None:
                        return
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    self.read_cache[tool + _dump(call)] = (self.version, self.step, self.show(seen))
                    self.trace.append({'kind': 'closest_search', 'item': number, 'args': call, 'rows': len(self.records_in(seen))})
            pool = [i for i in self.seen_order if i not in self.full and self.buyable(i) and str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind and (self.price_of(i) if self.price_of(i) is not None else float('inf')) < floor]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            self.fill_facts(pool[:CHEAPER_OPENS], self.tail() + 2)
            read += [i for i in pool[:CHEAPER_OPENS] if i in self.full]
        if read and (not self.late()):
            self.rank_closest(read)

    def hold_closest(self, job: Job) -> str:
        self.closest_hunt()
        return ''

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            offer = self.offer_note(spec.get('maxLength', 400) if isinstance(spec.get('maxLength'), int) else 400)
            if offer:
                out[key] = offer
                self.trace.append({'kind': 'offer_note', 'text': offer})
                continue
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Ending with no purchase: no listing here satisfies every requirement you gave.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing-taking call whose one free text argument holds a title or another field's text instead of an
        id gets the one listing that text starts; several or none is a problem the model is told about."""
        takers = {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None}
        if name not in takers or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ({'string', 'array'} & set(_kinds(v))) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')
        key = free[0]
        if key not in args:
            return (dict(args), '')
        many = isinstance(args[key], list)
        fixed = [self.listing_named(v) for v in (args[key] if many else [args[key]])]
        problem = next((p for _, p in fixed if p), '')
        if problem:
            return (args, problem)
        return ({**args, key: [v for v, _ in fixed] if many else fixed[0][0]}, '')

    def listing_named(self, value: Any) -> tuple[Any, str]:
        if isinstance(value, str) and value.strip() in self.rows:
            return (value.strip(), '')
        if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
            return (value, '')
        start = value.strip().lower()
        hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items()))]
        if len(hits) == 1:
            return (hits[0], '')
        if hits:
            return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
        return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')

    def cart_sets(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Keep Kyo's own record of the cart after a cart call: a call that lists whole lines replaces it, one
        that names lines adds, sets or (with a removal mode) takes units off them."""
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        modes = lambda key: (spec.get(key) or {}).get('enum')
        removal = any((isinstance(modes(k), list) and len(modes(k)) >= 2 and v in modes(k)[1:] for k, v in args.items()))
        lines, whole = self.lines_named(spec, args)
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = _next_units(self.cart_lines.get(ident, 0), count, removal, sets)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def lines_named(self, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
        """The (listing, units) pairs a cart call names, and whether it lists the whole cart (lines as objects)."""
        units = lambda values: next((_count(v) for v in values if _count(v) is not None and (not isinstance(v, str))), None)
        loose = units(args.values())
        found: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                found += [(value, loose)] if value in self.rows else []
                continue
            if not isinstance(value, list):
                continue
            shape = (spec.get(key) or {}).get('items') or {}
            whole = whole or (not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape))
            for line in value:
                if isinstance(line, dict):
                    whole = True
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    got = units(line.values())
                    if ident:
                        found.append((ident, 1 if got is None else got))
                elif isinstance(line, str) and line in self.rows:
                    found.append((line, loose))
        return (found, whole)

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
        # stores can refuse fewer ids than the schema allows (real r8: every call with 6+ ids, r9/r12: 3+), so a size
        # the store refused caps the schema's own limit
        if isinstance(spec.get('maxItems'), int):
            return min(spec['maxItems'], max(1, self.compare_bad - 1)) if self.compare_bad else spec['maxItems']
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
        """Answer each tool call of one model turn. A call passes Kyo's desk in three stages: screening answers
        calls that would waste a step or break a store rule without running them; the closing policy holds
        an ending until its checks pass; the rest runs and what it shows is recorded."""
        batch = self.join_opens(calls)
        before = set(self.opened)
        replies = [(call['id'], self.desk(call, len(batch))) for call in batch]
        early = self.early_trail(self.opened - before) if replies else ''
        if early:
            replies[-1] = (replies[-1][0], replies[-1][1] + '\n' + early)
        return replies

    def early_trail(self, fresh: set[str]) -> str:
        """The model has just opened a page of a shop that a pending requirement points into: Kyo walks that
        chain now, as a checkout would, with its own value search, instead of leaving the chain's first page
        to a pick among summary rows. Once per episode, leaving rounds for the checkout's walk."""
        if self.trails or not fresh or self.finished or self.late() or self.left() <= 15:
            return ''
        pending = self.pointers_open()
        if not any((fresh & set(self.opened_shop(need)) for need in pending)):
            return ''
        self.trace.append({'kind': 'early_trail', 'step': self.step, 'pending': pending})
        return self.trail(4)[0]

    def desk(self, call: dict[str, Any], batch: int) -> str:
        if call.get('joined'):
            return 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'
        job = self.screen(call.get('function') or {}, batch)
        if isinstance(job, str):
            return job
        before = set(self.rows)
        self.last_seen = None
        said = self.closing(job) or self.perform(job)
        if job.searching:
            self.note_page(job, before)
        return said

    def own_words(self, args: dict[str, Any], skip: str | None) -> set[str]:
        """The words of a search's text arguments that do not name a requested kind (its item names, store
        labels, and the category labels of rows that belong to a requested item)."""
        kind: set[str] = set()
        for item in self.items:
            kind |= _roots(item.name) | _roots(item.shelf or '')
        cat_key = self.category_key()
        for label in {str(r.get(cat_key) or '') for r in self.rows.values()} if cat_key else set():
            if self.owner_of(label):
                kind |= _roots(label)
        text = ' '.join((str(v) for k, v in args.items() if k != skip and isinstance(v, str)))
        return _roots(re.sub('[^A-Za-z0-9]+', ' ', text)) - kind

    def note_page(self, job: Job, before: set[str]) -> None:
        """Record what one search call brought: the scalar values its result handed out (a next-page token or
        offset is one of them) and whether it showed a new row of a requested kind that holds the search's own
        words (any new row of a requested kind when its words only name the kind)."""
        seen = self.last_seen
        offered = {_dump(v) for v in seen.values() if isinstance(v, (str, int)) and (not isinstance(v, bool))} if isinstance(seen, dict) else set()
        paged = self.page_of(job)
        words = self.own_words(job.args, paged[0] if paged else None)
        cat_key = self.category_key()
        found = False
        for ident, record in self.records_in(seen):
            if ident in before:
                continue
            row = self.rows.get(ident) or record
            if cat_key and (not self.owner_of(str(row.get(cat_key) or ''))):
                continue
            if not words or words & _roots(re.sub('[^A-Za-z0-9]+', ' ', _dump(row))):
                found = True
                break
        self.page_log = (self.page_log + [(job.name, dict(job.args), offered, found)])[-40:]

    def page_of(self, job: Job) -> tuple[str, str] | None:
        """(paging argument, the search it pages) when this call pages an earlier one: the same tool and the
        same arguments but one, whose value that earlier result handed out (or the tool's known paging
        argument)."""
        known = (self.shapes.get(job.name) or {}).get('next_arg')
        for name, args, offered, _ in reversed(self.page_log):
            if name != job.name:
                continue
            diff = [k for k in set(args) | set(job.args) if args.get(k) != job.args.get(k)]
            if len(diff) != 1 or job.args.get(diff[0]) in (None, '', 0, '0', 1, '1') or isinstance(job.args.get(diff[0]), bool):
                continue
            if diff[0] == known or _dump(job.args[diff[0]]) in offered:
                return (diff[0], _dump({k: v for k, v in job.args.items() if k != diff[0]}))
        return None

    def page_stop(self, job: Job, batch: int) -> str:
        """P7: a search whose last pages brought no new row of a requested kind holding its words is not
        paged on; the model is sent to search with other words."""
        if job.ending or job.name in {tool for role, tool in self.roles.items() if role != 'search'}:
            return ''
        paged = self.page_of(job)
        if not paged:
            return ''
        key, base = paged
        run = [found for name, args, _, found in self.page_log if name == job.name and _dump({k: v for k, v in args.items() if k != key}) == base]
        empty = 0
        for found in reversed(run):
            if found:
                break
            empty += 1
        if empty < EMPTY_PAGES:
            return ''
        self.trace.append({'kind': 'page_stop', 'step': self.step, 'tool': job.name, 'args': _clip(job.args, 80), 'empty': empty})
        return f'Not run (no step used): the last {empty} pages of this search brought no new listing of a requested kind that holds its words, so its later pages hold only the store\'s other rows. Search with other words instead: the values the listing should show, or its seller\'s name together with those values.'

    def readers(self) -> set[str]:
        return {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}

    def screen(self, fn: dict[str, Any], batch: int) -> Job | str:
        """The call as a Job, or the answer that stands in for running it (no step is used)."""
        if self.finished:
            return 'Not run: the episode is over.'
        name = fn.get('name')
        allowed = sorted((t['function']['name'] for t in self.offered()))
        if name == self.roles.get('ask'):
            return "Not run (no step used): Kyo's interviewer handles every message to the shopper."
        if name not in self.schemas:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(allowed)}."
        if name not in allowed:
            return 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(allowed) + '.'
        raw = fn.get('arguments')
        args = raw if isinstance(raw, dict) else _json_obj(raw) if (raw or '').strip() else {}
        if args is None:
            return 'Not run (no step used): the arguments were not one JSON object.'
        faults: list[str] = []
        args = _shape(args, self.schemas[name], 'args', faults)
        if faults:
            return 'Not run (no step used): ' + '; '.join(faults[:6])
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return f'Not run (no step used): {wrong}.'
        job = Job(name, args, self.is_buy(name, args) or self.is_refuse(name, args), self.compare_target() if name == self.roles.get('compare') else None)
        for rule in (self.fit_room, self.first_page, self.gone_target, self.absent_line, self.alone, self.tail_kept, self.repeated, self.cached, self.shown, self.page_stop):
            said = rule(job, batch)
            if said:
                return said
        return job

    def fit_room(self, job: Job, batch: int) -> str:
        """A side-by-side view shows at most compare_room listings once the store has refused more: the first go, the rest wait."""
        ids, room = (job.listed(), self.compare_room())
        if ids is not None and len(ids) > room and self.compare_bad:
            job.extra = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, ids[room:]))}."
            job.args = {**job.args, job.many: ids[:room]}
        return ''

    def first_page(self, job: Job, batch: int) -> str:
        """A search with words no earlier search sent, but with a paging value set, would skip that query's best
        rows: it goes out from its first page."""
        paging = (self.shapes.get(job.name) or {}).get('next_arg') if job.name == self.roles.get('search') else None
        value = job.args.get(paging) if paging else None
        if value in (None, '', 0, '0', 1, '1') or isinstance(value, bool):
            return ''
        base = {k: v for k, v in job.args.items() if k != paging}
        if any(({k: v for k, v in s.items() if k != paging} == base for s in self.searched)):
            return ''
        schema = self.schemas.get(job.name) or {}
        if paging in (schema.get('required') or []):
            spec = (schema.get('properties') or {}).get(paging) or {}
            if not ({'integer', 'number'} & set(_kinds(spec))):
                return ''
            base[paging] = spec.get('minimum', 0) if isinstance(spec.get('minimum'), int) else 0
        job.args = base
        job.extra += f'\n[Kyo] Sent from the first page: {paging} {value!r} belonged to an earlier query, and a new query starts at its best rows.'
        return ''

    def gone_target(self, job: Job, batch: int) -> str:
        target = self.single_target(job.name)
        ident = str(job.args.get(target)) if target else ''
        if not target or job.name not in (self.roles.get('open'), self.roles.get('offer')) or ident not in self.gone:
            return ''
        return f'Not run (no step used): {job.args[target]} sold out at step {self.gone[ident]} and will not come back. Replace its cart line with another fitting listing.'

    def absent_line(self, job: Job, batch: int) -> str:
        """Taking units off a line the cart does not hold."""
        if job.name != self.roles.get('cart') or not any(((_count(v) or 0) < 0 for v in job.args.values() if not isinstance(v, str))):
            return ''
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
        held = (self.cart_lines, view, self.cart_said)
        absent = next((v for v in job.args.values() if isinstance(v, str) and v in self.rows and (not any((v in place for place in held)))), None)
        return f'Not run (no step used): {absent} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text() if absent else ''

    def alone(self, job: Job, batch: int) -> str:
        return 'Not run (no step used): send the finishing call on its own, after reading your other results.' if job.ending and batch > 1 else ''

    def tail_kept(self, job: Job, batch: int) -> str:
        if job.ending or self.left() > self.tail() or not self.end_tools():
            return ''
        return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."

    def repeated(self, job: Job, batch: int) -> str:
        if job.ending or job.name == self.roles.get('cart') or job.mark() not in self.recent:
            return ''
        return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'

    def cached(self, job: Job, batch: int) -> str:
        hit = self.read_cache.get(job.mark()) if job.name in self.readers() else None
        if not hit or hit[0] != self.version:
            return ''
        return f'Not run (no step used): the same call at step {hit[1]} returned this, and nothing has changed since:\n' + hit[2][:3000]

    def shown(self, job: Job, batch: int) -> str:
        """Reading a listing that has shown everything since the last change; a side-by-side view leaves those listings out."""
        target = self.single_target(job.name)
        if target and job.name in self.readers() and job.name != self.roles.get('compare') and self.shown_already(job.name, str(job.args.get(target))):
            return f'Not run (no step used): {job.args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(job.args[target]))
        ids = job.listed()
        if not ids:
            return ''
        fresh = [i for i in ids if not self.shown_already(job.name, str(i))]
        if not fresh:
            return 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in ids))[:4000]
        if len(fresh) < len(ids):
            job.extra += f"\n[Kyo] Left out {', '.join((str(i) for i in ids if i not in fresh))}: shown in full already, nothing changed since."
            job.args = {**job.args, job.many: fresh}
        return ''

    def closing(self, job: Job) -> str:
        """The closing policy: the checks an ending must pass, in order; the first that holds answers the call."""
        if not job.ending:
            return ''
        roomy = self.left() > 2
        if self.is_buy(job.name, job.args):
            checks = [self.hold_wider, self.hold_changed, self.hold_sold, self.hold_chain, self.hold_misfit, self.hold_lines, self.hold_review, self.hold_stale] if roomy else []
        else:
            early = roomy and self.left() > 5
            checks = ([self.hold_audit, self.hold_leads] if early else []) + [self.hold_second_end, self.hold_late_end] + ([] if early else [self.hold_late_review]) + [self.hold_unsearched, self.hold_closest, self.end_by_lead]
        for check in checks + [self.hold_gate]:
            said = check(job)
            if said:
                return said
        return ''

    def hold_wider(self, job: Job) -> str:
        wider = self.wider_hold()
        return wider + '\nCart now: ' + self.cart_text() if wider else ''

    def hold_changed(self, job: Job) -> str:
        """Re-read stale cart lines, then look at the cart once: a change the model has not seen holds the order."""
        job.reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        tool = self.roles.get('cart_view')
        if not (self.cart_touched and self.cart_seen_at < self.cart_at and tool and (self.left() > 3)):
            return ''
        faults: list[str] = []
        args = _shape({}, self.schemas.get(tool), 'args', faults)
        view = None if faults else self.act(tool, args)
        if view is None:
            return ''
        self.cart_view, self.cart_seen_at = (view.get('observation'), self.step)
        change = self.observe(tool, self.cart_view)
        if not change:
            return ''
        self.queued = []
        return 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'

    def hold_sold(self, job: Job) -> str:
        swap = self.swap_hint()
        held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 and (not swap) else []
        if not swap and (not held):
            return ''
        self.held += 1 if held else 0
        return 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_text()

    def hold_chain(self, job: Job) -> str:
        job.chain, hold = self.trail()
        return 'Not run yet (no step used). ' + job.chain + '\nCart now: ' + self.cart_text() if hold else ''

    def hold_misfit(self, job: Job) -> str:
        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
        if not misfit:
            return ''
        self.fit_holds += 1
        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
        return 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()

    def hold_lines(self, job: Job) -> str:
        """Cart lines that do not match the request. When only quantities are off, Kyo sets them itself."""
        faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
        if not faults:
            return ''
        self.line_holds += 1
        if all((' is held at ' in f for f in faults)):
            fixed = self.fix_counts(self.count_fixes())
            faults = self.line_faults() if fixed else faults
            if not faults:
                return "Not run yet (no step used). Kyo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()
        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
        return 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()

    def hold_review(self, job: Job) -> str:
        read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
        verdict = self.check_cart()
        if not verdict:
            return ''
        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
        return 'Not run yet. ' + job.chain + seen + verdict + '\nCart now: ' + self.cart_text()

    def hold_stale(self, job: Job) -> str:
        if not job.reread or self.cost <= SPEND_SOFT:
            return ''
        return "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + job.reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'

    def hold_audit(self, job: Job) -> str:
        review = self.audit()
        return 'Not run yet (no step used). ' + review if review else ''

    def hold_leads(self, job: Job) -> str:
        """Listings the review thinks fit hold the ending: first a push to buy them, then an independent check of the first."""
        if not self.leads or self.left() <= 4 or self.refuse_blocks > 1:
            return ''
        self.refuse_blocks += 1
        if self.refuse_blocks == 1:
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'
        first = self.leads[0]
        if not self.passes([first]) or first not in self.leads:
            return ''
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."

    def hold_second_end(self, job: Job) -> str:
        if not NO_SECOND_END or not self.end_refused or self.end_blocks >= 2:
            return ''
        self.end_blocks += 1
        self.trace.append({'kind': 'second_end_held', 'refused': self.end_refused})
        return 'Not run (no step used): the store already turned down ending without a purchase in this episode, and it does not accept that ending later either. Apply what its reply said, put the best fitting listing you have read for each item in the cart at the requested quantity, and check out.\nCart now: ' + self.cart_text()

    def hold_late_end(self, job: Job) -> str:
        """A cart once built for every item is put back and checked out instead (once). Real races 5-14: 118 failed
        episodes dropped a carted paid listing still in stock and in budget and then ended without a purchase,
        against at most 37 right endings after a cart; the hold had waited for the last 5 steps."""
        if self.late_held:
            return ''
        lines = [i for i in self.built if not self.sold_out(i)]
        if self.left() > 5:
            # before the last steps only a cart no requirement was added to since, inside the budget in force
            cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
            if {need for _, need in self.needs_now()} - self.built_needs or (cap is not None and sum((self.price_of(i) or 0) for i in lines) > cap):
                return ''
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return ''
        self.late_held = True
        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
        redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
        return 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'

    def hold_unsearched(self, job: Job) -> str:
        if self.leads or self.pushed or self.left() <= self.max_steps // 3:
            return ''
        self.pushed = True
        return f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."

    def hold_late_review(self, job: Job) -> str:
        done = self.decline_review()
        return done + (' [episode finished]' if self.finished else '') if done else ''

    def decline_review(self) -> str:
        """Before an ending without a purchase that no review has examined (sent with 5 steps left or fewer,
        past the review in the closing checks, or Kyo's own ending with no full cart), Kyo runs the review
        once while the cart lines, the order and a few model calls still fit: it reads the listings seen,
        opening unread rows only while that room stays free, and buys a review pick only after the
        independent check passes the whole cart. Real races 11-14: most of the know line's declines on
        payable tasks came with 6 steps left or fewer; love's study of its own put 7 of 9 at exactly 5."""
        if self.audited or self.completed or self.finished or not self.roles.get('cart') or not self.roles.get('buy'):
            return ''
        if self.left() < len(self.items) + 2 or self.spent() > CLOCK_STOP - 30 or self.cost > SPEND_HARD:
            return ''
        self.audit(len(self.items) + 1)
        self.trace.append({'kind': 'decline_review', 'left': self.left(), 'leads': list(self.leads)})
        if not self.leads:
            return ''
        self.confirmed = self.leads[0]
        return self.complete_cart()

    def end_by_lead(self, job: Job) -> str:
        """A confirmed lead is bought by Kyo itself; otherwise the ending goes out with a short reason."""
        done = self.complete_cart()
        if done:
            return done + (' [episode finished]' if self.finished else '')
        job.args = self.quiet(job.name, job.args)
        return ''

    def hold_gate(self, job: Job) -> str:
        if self.left() < 2:
            return ''
        go, said = self.gate('order' if self.is_buy(job.name, job.args) else 'end')
        if not go:
            return 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
        if said:
            job.extra = '\n[Kyo] ' + said + job.extra
        return ''

    def perform(self, job: Job) -> str:
        """Run the call, keep what it shows, and write the answer the model reads."""
        reply = self.act(job.name, job.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        seen = reply.get('observation')
        self.last_seen = seen
        self.side_view_seen(job, seen)
        self.recent = (self.recent + [job.mark()])[-3:]
        for key, value in job.args.items():
            values = value if isinstance(value, list) else [value]
            if values and all((isinstance(v, str) and v in self.rows for v in values)):
                self.id_args.add((job.name, key))
        self.absorb(seen)
        paged = self.learn_view(job, seen)
        self.learn_cart(job, seen)
        if self.is_refuse(job.name, job.args) and (not reply.get('finished')) and (not self.finished):
            self.end_refused += 1
        if self.is_buy(job.name, job.args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
        return self.answer(job, reply, paged)

    def side_view_seen(self, job: Job, seen: Any) -> None:
        """Learn how many listings the side-by-side view shows from what it returned."""
        ids = job.listed()
        if not ids or len(ids) < 2:
            return
        text = _dump(seen)
        if any((str(i) in text for i in ids)):
            self.compare_ok = max(self.compare_ok, len(ids))
            return
        self.compare_bad = min(self.compare_bad or 999, len(ids))
        job.extra += f'\n[Kyo] The side-by-side view showed nothing for {len(ids)} listings: send at most {self.compare_room()} per call.'

    def learn_view(self, job: Job, seen: Any) -> str:
        """What a reading call teaches: which view carries the attributes, which tool shows the offer, the
        searches that returned rows. Returns the extra pages read for a search."""
        name = job.name
        if name == self.roles.get('compare'):
            rows = self.records_in(seen)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in self.readers() - {self.roles.get('search')}:
            self.note_view(name, seen)
        job.searching = not job.ending and name not in {tool for role, tool in self.roles.items() if role != 'search'}
        paged = self.more_pages(name, job.args, seen) if job.searching else ''
        if job.searching:
            self.learn_cap(name, job.args, seen)
            if self.brief.get('quiet_decline') is False:
                self.search_log = (self.search_log + [(name, dict(job.args), seen)])[-12:]
        steady = {self.roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask')}
        listing = self.ident(seen) if isinstance(seen, dict) and self.single_target(name) and name not in steady else None
        if listing:
            self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(seen))
            if name == self.roles.get('open') and (not self.facts_from[name]):
                spare = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False and t not in steady | {self.roles.get('search')}), None)
                if spare:
                    self.roles['open'] = spare
            if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                self.roles['offer'] = name
            if name == self.offer_tool():
                self.offer_seen[listing] = self.step
        if name == self.roles.get('search') and self.records_in(seen):
            paged += self.learn_rule(name, job.args, seen)
            self.searched = (self.searched + [dict(job.args)])[-8:]
        return paged

    def learn_cart(self, job: Job, seen: Any) -> None:
        """Keep the cart's lines after a cart call; remember the lines taken off and the fullest cart built."""
        if job.name == self.roles.get('cart_view'):
            self.cart_view, self.cart_seen_at = (seen, self.step)
        if job.name != self.roles.get('cart'):
            return
        self.note_cart(job.name, job.args)
        self.cart_said = _dump(seen)
        self.recent = []
        counts = [_count(v) for v in job.args.values() if not isinstance(v, str) and _count(v) is not None]
        for ident in (v for v in job.args.values() if counts and isinstance(v, str) and v in self.rows):
            if counts[0] > 0:
                self.dropped.discard(ident)
            elif not _in_rows(seen, ident):
                self.dropped.add(ident)
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.built):
            self.built = lines
            self.built_needs = {need for _, need in self.needs_now()}

    def answer(self, job: Job, reply: dict[str, Any], paged: str) -> str:
        """The result as the model reads it, with what Kyo noticed added below it."""
        seen = reply.get('observation')
        plain = self.show(seen) + paged + job.extra
        if job.searching:
            plain += self.cheapest_here([seen] + self.paged_obs) + self.cap_ignored(job.name, job.args, seen)
        notes = [self.evidence(seen), self.zero_count_hint(job, seen), self.read_store_note(seen), self.after_refusal(job, reply), self.swap_hint(), self.queued_notes(), self.cheaper_search()]
        if job.name in self.readers():
            self.read_cache[job.mark()] = (self.version, self.step, plain)
        return plain + ''.join(('\n' + note for note in notes if note)) + (' [episode finished]' if reply.get('finished') else '')

    def zero_count_hint(self, job: Job, seen: Any) -> str:
        if job.name != self.roles.get('cart') or not any((_count(v) == 0 for v in job.args.values() if not isinstance(v, str))):
            return ''
        if not any((isinstance(v, str) and v in self.rows and _in_rows(seen, v) for v in job.args.values())):
            return ''
        return '[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'

    def read_store_note(self, seen: Any) -> str:
        self.note_sold = False
        lines = self.loose_lines(seen)
        meaning = self.take_note(lines) if lines else ''
        return f'[note read] {meaning}. Apply this change to your picks before checking out.' if meaning else ''

    def after_refusal(self, job: Job, reply: dict[str, Any]) -> str:
        """After a refused checkout, or a note that something sold out, re-read the cart's lines."""
        buying = self.is_buy(job.name, job.args)
        refused = buying and (not reply.get('finished'))
        again = refused and any((line in self.spoken(reply.get('observation')) for line in self.sold_lines))
        loose = self.note_sold and (refused or (not self.note_named and (not buying)))
        found = self.find_sold() if (loose or again) and self.cart_ids() and (not self.finished) else ''
        self.note_sold = False
        return "[Kyo re-read the cart's lines after the refusal]\n" + found if found else ''

    def queued_notes(self) -> str:
        if not self.queued:
            return ''
        text = "[notes seen on Kyo's own calls] " + ' | '.join(self.queued)
        self.queued = []
        return text

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # a purchase that misses a requested item earns nothing: such a cart ends without a purchase instead
        short = holding and refuse is not None and len(self.cart_ids()) < len(self.items)
        if short:
            self.trace.append({'kind': 'short_cart_end', 'cart': self.cart_ids()})
        if buy and (not short) and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
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
        if refuse and (short or not holding):
            done = self.decline_review()
            if self.finished:
                return ''
            if done:
                return done
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

    def late_pick(self, number: int, held: list[str], index: dict[str, Any]) -> str | None:
        """A read page for a requested item that passes Kyo's code checks: of that item's kind, buyable, not
        taken off before, its requirements' words not held only apart, and its fit marks naming the device
        codes the item's requirements give or the device line in the cart (when the page names any). The
        cheapest when the pick is cheapest, else the newest read."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        needs = [m for m in item.musts + item.told if m not in item.dropped]
        codes = {c for need in needs for c in _codes(need)}
        for part, main in self.bound:
            if part == number:
                codes |= {c for i in held if self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) == main for c in index['marks'].get(i, {}).get('codes') or []}
        owns = {m.get('own') for m in index['marks'].values()}
        found = []
        for ident in reversed(list(self.full)):
            if self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) != number or ident in held:
                continue
            if not self.buyable(ident) or ident in self.gone or ident in self.dropped or self.apart_needs(ident, needs):
                continue
            fits = self.fit_set(ident, index)
            rest = (index['marks'].get(ident) or {}).get('more')
            if codes and fits and (not fits & codes) and (not (rest and rest not in owns)):
                continue
            found.append(ident)
        if self.brief.get('pick') == 'cheapest':
            found.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
        return found[0] if found else None

    def late_add(self) -> None:
        """P8: at the clock stop, a cart that misses only some requested items gets, for each missing item, a
        read page that passes Kyo's code checks and then the cart check (one model call, cut to the time
        left), when the time and steps left allow the calls; the checkout then goes the usual way. Otherwise
        the cart stays as it is, and a cart that misses an item ends without a purchase."""
        cart, buy, refuse = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'))
        cat_key = self.category_key()
        if self.finished or self.broke or self.no_credit or (not self.late()) or (not cart) or (not buy) or (not refuse) or (not cat_key):
            return
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and (not self.sold_out(i))]
        owner = {i: self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in held}
        missing = [n for n in range(1, len(self.items) + 1) if n not in owner.values()]
        if not held or not missing or 0 in owner.values() or len(set(owner.values())) != len(held):
            return
        index = self.fit_index()
        picks = {n: self.late_pick(n, held, index) for n in missing}
        trace: dict[str, Any] = {'kind': 'late_add', 'step': self.step, 'cart': held, 'picks': picks}
        self.trace.append(trace)
        if not all(picks.values()):
            trace['why'] = 'no page passes'
            return
        lines = {**{i: self.held_units(i) for i in held}, **{str(p): self.items[n - 1].qty for n, p in picks.items()}}
        prices = [self.price_of(i) for i in lines]
        if isinstance(self.budget, (int, float)) and None not in prices:
            each = self.brief.get('budget_scope') == 'each'
            if (each and max(prices) > self.budget) or ((not each) and sum((p * lines[i] for p, i in zip(prices, lines))) > self.budget + 1e-09):
                trace['why'] = 'budget'
                return
        calls = self.cart_calls(lines, [])
        room = EPISODE_SECONDS - (time.monotonic() - self.began) - 8
        if not calls or len(calls) + 3 > self.left() or room < 6:
            trace['why'] = 'no room'
            return
        ids = list(lines)
        # one model call at most, cut to the time left
        calls_before, timeout = (self.calls, getattr(_llm, 'timeout', None))
        self.calls = max(self.calls, ALL_CALLS - 1)
        if isinstance(timeout, (int, float)):
            _llm.timeout = max(4, int(min(timeout, room)))
        try:
            ok = self.passes(ids)
        finally:
            used = self.calls - max(calls_before, ALL_CALLS - 1)
            self.calls = calls_before + max(0, used)
            if isinstance(timeout, (int, float)):
                _llm.timeout = timeout
        trace['checked'] = ok
        if not ok or self.late_over():
            return
        trace['placed'] = self.place(cart, calls)

    def late_over(self) -> bool:
        return EPISODE_SECONDS - (time.monotonic() - self.began) < 6

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
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            self.late_add()
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (not self.late()):
                listed = self.cheaper_search()
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + ('\n' + listed if listed else '') + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        retried = False
        nudged_tail = False
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap):
            if self.late():
                self.trace.append({'kind': 'clock_stop', 'step': self.step})
                break
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
                    names = [t['function']['name'] for t in self.offered()]
                    if not nudged_tail and self.read_room() and len(names) < len([t for t in self.tools if t['function']['name'] != self.roles.get('ask')]):
                        # Turns without a call on the last steps' narrowed tools while an item has no cart line (Arena
                        # b27-b32: 8 of 17 endings without a purchase on order tasks): once, the open tools are named.
                        nudged_tail = True
                        idle = 0
                        self.chat[-1] = {'role': 'user', 'content': f"[Kyo] Your last turns made no tool call. Open to you now: {', '.join(names)}. An item still has no cart line: use them to find and read a listing that fits, cart it and check out; end without a purchase only if your search showed that nothing fits.\n" + self.status()}
                        self.trace.append({'kind': 'idle_tail_nudge', 'step': self.step, 'tools': names})
                        continue
                    # Turns without a call this early (no listing page read, under half the steps used) are a model
                    # that wrongly believes the tools are gone: one more round with the tool names before the ending.
                    if retried or self.full or self.step * 2 >= self.max_steps or self.left() <= 2:
                        break
                    retried = True
                    idle = 0
                    names = ', '.join((t['function']['name'] for t in self.offered()))
                    self.chat[-1] = {'role': 'user', 'content': f'[Kyo] Your last turns made no tool call. The store tools are working and available now: {names}. Call one of them in this turn; nothing has been read in full yet, so open the listings in each category.\n' + self.status()}
                    self.trace.append({'kind': 'idle_retry', 'step': self.step})
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8):
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
        episode = Kyo(problem_data)
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
