from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
PUZZLED = re.compile('simpler|plain(?:er)? (?:terms|words)|didn.?t (?:understand|follow|catch|get|land)|could not catch|not following|lost (?:me|your)|none the wiser|word it|say it (?:again|another)|try (?:it )?again|another go|once more|come again|pardon|adrift|confus|explain|struggling|above my understanding|over the bonnet|one ear|can.?t tell what|no idea what|what (?:exactly )?do you mean|jargon|static|read it .{0,30}nothing|words swam|went by in a blur|put it differently|another way|muddy palette|gray wash|grey wash|not sinking in|lost my place|repeat your last|didn.?t catch your meaning|fewer fancy|brain is mush', re.I)
ONE_AT_A_TIME = re.compile('one (?:by one|at a time|thing|item|query|prompt|subject|question)|split them|separately|smaller bites|a trickle|one (?:topic|point) (?:a|per)|just ask me the first|slow the pace|do one .{0,12} and then|(?:keep|sticking) to (?:just )?one|questions came|too many (?:things|questions)|a lot to juggle|get tangled|forgetting half', re.I)
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
SCARCE_MESSAGES = 2
LIST_SIZE = 8
LIST_MESSAGES = 4
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(retry_delay=1, max_retries=1, timeout=50)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(retry_delay=1, max_retries=3, timeout=95)
GUIDE = 'You are Halo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Halo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Halo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Halo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Halo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Halo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Halo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.'
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
    held_1 = list((re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))))
    return held_1

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
    also = list(((hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)))
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    held_2 = {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}
    return held_2

def _dump(value: Any) -> str:
    try:
        held_3 = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
        return held_3
    except (TypeError, ValueError):
        held_4 = str(value)
        return held_4

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if 7 < depth:
        return '...'
    elif isinstance(value, str):
        if limit >= len(value):
            return value
        held_5 = value[:limit] + '...'
        return held_5
    elif not isinstance(value, dict):
        if isinstance(value, list):
            held_7 = list((_clip(v, limit, depth + 1) for v in value))
            return held_7
        return value
    else:
        held_6 = dict(((k, _clip(v, limit, depth + 1)) for k, v in value.items()))
        return held_6

def _json_obj(text: Any) -> dict[str, Any] | None:
    if isinstance(text, str):
        text = text.strip()
        fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
        if fence:
            text = fence.group(1)
        try:
            got = json.loads(text)
            if isinstance(got, dict):
                return got
            return None
        except ValueError:
            pass
        left, right = (text.find('{'), text.rfind('}'))
        if 0 <= left < right:
            try:
                got = json.loads(text[left:right + 1])
                if isinstance(got, dict):
                    return got
                return None
            except ValueError:
                return None
        return None
    else:
        return None

def _words(key: str) -> str:
    held_8 = str(key)
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', held_8)
    held_9 = re.sub('[\\s_.\\-]+', ' ', key).strip().lower()
    return held_9

def _snake(key: str) -> str:
    held_10 = _words(key).replace(' ', '_')
    return held_10

def _stem(text: str) -> frozenset[str]:
    held_11 = frozenset((w[:-1] if 3 < len(w) and w.endswith('s') else w for w in _words(text).split()))
    return held_11

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        if re.search('(?:ch|sh|ss|x|z)es$', w):
            w = w[:-2]
        else:
            w = w[:-1] if 3 < len(w) and w.endswith('s') and (not w.endswith('ss')) else w
        if 3 <= len(w):
            if w not in ('and', 'for', 'the', 'with'):
                out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    held_12 = frozenset((re.sub('our$', 'or', w[:-1] if 3 < len(w) and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))
    return held_12

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not (not prop or not key):
        held_13 = key == prop or key >= prop or (2 <= len(key) and prop >= key)
        return held_13
    else:
        return False

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if not isinstance(kind, str):
        held_15 = list((k for k in kind or [] if isinstance(k, str)))
        return held_15
    else:
        held_14 = [kind]
        return held_14

def _fits_kind(value: Any, kind: str) -> bool:
    if 'string' == kind:
        held_16 = isinstance(value, str)
        return held_16
    elif 'integer' == kind:
        held_17 = isinstance(value, int) and (not isinstance(value, bool))
        return held_17
    elif not 'number' == kind:
        if 'boolean' == kind:
            held_19 = isinstance(value, bool)
            return held_19
        if 'array' == kind:
            held_20 = isinstance(value, list)
            return held_20
        if 'object' == kind:
            held_21 = isinstance(value, dict)
            return held_21
        if 'null' == kind:
            held_22 = None is value
            return held_22
        return True
    else:
        held_18 = isinstance(value, (int, float)) and (not isinstance(value, bool))
        return held_18

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if isinstance(schema, dict):
        kinds = _kinds(schema)
        if kinds:
            if not any((_fits_kind(value, k) for k in kinds)):
                for kind in kinds:
                    if 'integer' == kind and isinstance(value, float) and value.is_integer():
                        value = int(value)
                        break
                    elif kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                        number = float(value)
                        if 'integer' == kind or number.is_integer():
                            value = int(number)
                        else:
                            value = number
                        break
                    elif not ('string' == kind and isinstance(value, (int, float)) and (not isinstance(value, bool))):
                        if 'boolean' == kind:
                            if isinstance(value, str):
                                if value.strip().lower() in ('true', 'false'):
                                    value = value.strip().lower() == 'true'
                                    break
                        if 'array' == kind:
                            if not isinstance(value, list):
                                if None is not value:
                                    value = [value]
                                    break
                    else:
                        value = str(value)
                        break
                if not any((_fits_kind(value, k) for k in kinds)):
                    held_24 = '/'.join(kinds)
                    held_23 = '{} should be {}'.format(where, held_24)
                    faults.append(held_23)
                    return value
        options = schema.get('enum')
        if isinstance(options, list):
            if value not in options:
                if isinstance(value, str):
                    low = value.strip().lower()
                    match = list((o for o in options if isinstance(o, str) and o.lower() == low))
                    if match:
                        value = match[0]
                if value not in options:
                    held_25 = '{}: allowed values are {}'.format(where, options)
                    faults.append(held_25)
        if isinstance(value, str):
            if isinstance(schema.get('maxLength'), int):
                if schema['maxLength'] < len(value):
                    value = value[:schema['maxLength']]
        if isinstance(value, str):
            if isinstance(schema.get('minLength'), int):
                if schema['minLength'] > len(value):
                    held_26 = '{} is too short'.format(where)
                    faults.append(held_26)
        if isinstance(value, (int, float)):
            if not isinstance(value, bool):
                if isinstance(schema.get('minimum'), (int, float)):
                    if schema['minimum'] > value:
                        held_27 = '{} is below {}'.format(where, schema['minimum'])
                        faults.append(held_27)
                if isinstance(schema.get('maximum'), (int, float)):
                    if schema['maximum'] < value:
                        held_28 = '{} is above {}'.format(where, schema['maximum'])
                        faults.append(held_28)
        if isinstance(value, list):
            if isinstance(schema.get('maxItems'), int):
                if schema['maxItems'] < len(value):
                    held_29 = '{} allows at most {} entries'.format(where, schema['maxItems'])
                    faults.append(held_29)
            if isinstance(schema.get('minItems'), int):
                if schema['minItems'] > len(value):
                    held_30 = '{} needs at least {} entries'.format(where, schema['minItems'])
                    faults.append(held_30)
            if isinstance(schema.get('items'), dict):
                value = list((_shape(v, schema['items'], '{}[{}]'.format(where, i), faults) for i, v in enumerate(value)))
        if isinstance(value, dict):
            if isinstance(schema.get('properties'), dict):
                props = schema.get('properties')
            else:
                props = {}
            if schema.get('additionalProperties') is False:
                for name in list((n for n in value if n not in props)):
                    held_31 = '{}.{} is not an allowed argument'.format(where, name)
                    faults.append(held_31)
            for name in schema.get('required') or []:
                if name not in value:
                    held_32 = '{}.{} is required'.format(where, name)
                    faults.append(held_32)
            value = dict(((k, _shape(v, props.get(k), '{}.{}'.format(where, k), faults) if k in props else v) for k, v in value.items()))
        return value
    else:
        return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    elif not (isinstance(value, (str, dict)) and value):
        held_34 = []
        return held_34
    else:
        held_33 = [value]
        return held_33

def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    elif isinstance(value, int):
        return value
    elif not (isinstance(value, float) and value.is_integer()):
        if isinstance(value, str):
            if re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
                held_36 = int(float(value))
                return held_36
        return None
    else:
        held_35 = int(value)
        return held_35

def _id_like(value: Any, named: bool=False) -> bool:
    if not (not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip()):
        held_37 = named or bool(re.search('\\d', value))
        return held_37
    else:
        return False

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = list((row.get(key) for row in rows))
        if all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            values = list((v.strip() for v in values))
            if not (1 < len(rows) and len(set(values)) < len(values)):
                score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
                if None is best or best[0] < score:
                    best = (score, key)
            else:
                continue
        else:
            continue
    if best:
        held_38 = best[1]
        return held_38
    return None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if 5 < depth:
        return False
    elif isinstance(node, list):
        held_39 = any((isinstance(v, dict) and ident in list((str(x).strip() for x in v.values() if isinstance(x, str))) for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
        return held_39
    elif not isinstance(node, dict):
        return False
    else:
        held_40 = any((_in_rows(v, ident, depth + 1) for v in node.values()))
        return held_40

def _llm_within(seconds: float) -> ProxyClient:
    if not 50 <= seconds:
        limit = max(5, int(seconds) // 5 * 5)
        if limit not in _llm_short:
            _llm_short[limit] = ProxyClient(retry_delay=1, max_retries=1, timeout=limit)
        held_41 = _llm_short[limit]
        return held_41
    else:
        return _llm

def _flat_value(text: str) -> Any:
    bits = list((b for b in re.split(';\\s+', text) if b.strip()))
    if bits and all((re.match('\\s*[A-Za-z][\\w .\\-/]{0,39}=', b) for b in bits)) and (2 <= len(bits) or ' ' not in text.split('=', 1)[0]):
        held_42 = dict(((b.split('=', 1)[0].strip(), b.split('=', 1)[1].strip()) for b in bits))
        return held_42
    elif re.fullmatch('-?\\d+', text):
        held_43 = int(text)
        return held_43
    elif not re.fullmatch('-?\\d+\\.\\d+', text):
        if text in ('True', 'False'):
            held_45 = 'True' == text
            return held_45
        if text in ('None', 'null'):
            return None
        return text
    else:
        held_44 = float(text)
        return held_44

def _flat_record(text: Any) -> dict[str, Any] | None:
    if not (not isinstance(text, str) or ' | ' not in text):
        out: dict[str, str] = {}
        last = None
        for part in text.split(' | '):
            found = re.match('\\s*([A-Za-z_][\\w .\\-]{0,39}):(?:\\s(.*)|$)', part, re.S)
            if found and found.group(1).strip() not in out:
                last = found.group(1).strip()
                out[last] = (found.group(2) or '').strip()
            elif not None is not last:
                return None
            else:
                out[last] += ' | ' + part
        if not (3 > len(out) or not any((' ' not in v and re.search('\\d', v) and re.search('[A-Za-z]', v) and (3 <= len(v) <= 64) for v in out.values()))):
            held_46 = dict(((k, _flat_value(v)) for k, v in out.items()))
            return held_46
        else:
            return None
    else:
        return None

def _flat_records(node: Any, depth: int=0) -> Any:
    if 6 < depth:
        return node
    elif isinstance(node, str):
        record = _flat_record(node)
        if None is record:
            return node
        return record
    elif not isinstance(node, list):
        if isinstance(node, dict):
            held_48 = dict(((k, _flat_records(v, depth + 1)) for k, v in node.items()))
            return held_48
        return node
    else:
        held_47 = list((_flat_records(v, depth + 1) for v in node))
        return held_47

def _fact_block(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        best = None
        for value in node.values():
            if isinstance(value, dict):
                if 2 <= len(value):
                    scalars = dict(((k, v) for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))))
                    if 2 <= len(scalars):
                        if None is best or len(best) < len(scalars):
                            best = scalars
        return best
    else:
        return None

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Halo:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = list((t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)))
        self.schemas = dict(((t['function']['name'], t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}}) for t in self.tools))
        held_49 = self.view.get('max_steps')
        steps = _count(held_49)
        if None is steps:
            if isinstance(self.view.get('max_steps'), float):
                steps = int(self.view['max_steps'])
        if steps and 0 < steps:
            self.max_steps = steps
        else:
            self.max_steps = 30
        self.started = time.monotonic()
        self.step = 0
        held_50 = self.view.get('finished')
        self.finished = bool(held_50)
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
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
        self.told_ids: set[str] = set()
        self.refused_change = ''
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.ptr_block = False
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
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        held_51 = self.max_steps - self.step
        return held_51

    def clock(self) -> float:
        held_52 = time.monotonic() - self.started
        return held_52

    def model_room(self) -> float:
        held_53 = TASK_SECONDS - TIME_MARGIN - self.clock()
        return held_53

    def msgs_free(self) -> int:
        held_54 = self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)
        return held_54

    def cart_mark(self) -> str:
        held_55 = _dump(sorted(self.cart_lines.items()))
        return held_55

    def page_tools(self) -> set[str]:
        held_56 = self.roles.get('offer')
        if not held_56:
            held_56 = self.roles.get('open')
        page = held_56
        held_57 = set((t for t in (page, self.roles.get('compare')) if t))
        return held_57

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if not known:
            tool = self.roles.get('open')
            if self.facts_from.get(tool or '') is not False and self.single_target(tool):
                return tool
            return None
        else:
            return known

    def compare_facts(self) -> bool | None:
        held_58 = self.facts_from.get(self.roles.get('compare') or '')
        return held_58

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        held_59 = False is seen or (None is seen and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))
        return held_59

    def gate_due(self, stage: str) -> bool:
        held_60 = any((stage == p['when'] for p in self.process)) and (stage, self.cart_mark() + '#{}'.format(self.version)) not in self.gate_ok
        return held_60

    def tail(self) -> int:
        talk = any(('order' == p['when'] and 0 < p['messages'] for p in self.process))
        if talk and self.msg_limit > self.msg_used and self.gate_due('order'):
            return 2
        return 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if not (limit >= self.cost and SPEND_HARD >= self.cost + HEAVY_ROOM):
            if SPEND_HARD >= self.cost:
                held_62 = [LIGHT]
                return held_62
            held_63 = []
            return held_63
        else:
            held_61 = [HEAVY, LIGHT]
            return held_61

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if ALL_CALLS <= self.calls:
                return None
            elif not self.model_room() < CALL_FLOOR:
                self.calls = self.calls + 1
                tries = 0
                while True:
                    message, status = self.call_model(model, messages, max_tokens, tools)
                    if not None is not message:
                        held_64 = None is status
                        if not held_64:
                            held_64 = status in (402, 408, 425, 429)
                        if not held_64:
                            held_65 = isinstance(status, int)
                            if held_65:
                                held_65 = 500 <= status
                            held_64 = held_65
                        passing = held_64
                        if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or WAIT_TOTAL < self.waited + WAITS[tries]:
                            break
                        elif not self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                            pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                            self.waited += WAITS[tries]
                            tries = tries + 1
                            time.sleep(pause)
                        else:
                            break
                    else:
                        return message
                if 402 == status:
                    self.no_credit = True
                    return None
            else:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/'):
            if 1 < len(messages):
                sent = list((dict(m) for m in messages))
                marks = [0] + ([len(sent) - 1] if 1 < len(sent) else [])
                for index in marks:
                    content = sent[index].get('content')
                    if isinstance(content, str):
                        if content:
                            sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if not (isinstance(data, dict) and data.get('choices')):
            held_70 = result.error
            if not held_70:
                held_70 = {}
            error = held_70
            status = error.get('status')
            if isinstance(data, dict):
                if isinstance(data.get('error'), dict):
                    held_71 = data['error'].get('code')
                    status = _count(held_71)
                    error = {'body': data['error'].get('message')}
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
            held_72 = (None, status)
            return held_72
        else:
            held_66 = data.get('usage')
            if not held_66:
                held_66 = {}
            usage = held_66
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            held_67 = data['choices'][0].get('message')
            if not held_67:
                held_67 = {}
            message = held_67
            if not (message.get('content') or message.get('tool_calls')):
                self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
                held_69 = (None, 0)
                return held_69
            else:
                held_68 = (message, 200)
                return held_68

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            held_73 = (message or {}).get('content')
            got = _json_obj(held_73)
            if None is not got:
                return got
            elif self.broke or self.no_credit or ALL_CALLS <= self.calls:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if not isinstance(result.data, dict):
                held_74 = result.error
                if not held_74:
                    held_74 = {}
                error = held_74
                status = error.get('status')
                body = str(error.get('body') or '')
                self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
                if 409 == status:
                    low = body.lower()
                    if not ('finish' in low or 'quarantin' in low or 'finaliz' in low):
                        found = re.search('expected (\\d+)', body)
                        if found:
                            self.step = int(found.group(1)) - 1
                            payload['step'] = self.step + 1
                            continue
                        elif not ('reuse' in low or 'different action' in low):
                            return None
                        else:
                            self.step = self.step + 1
                            payload['step'] = self.step + 1
                            continue
                    else:
                        self.finished = True
                        return None
                elif status in (400, 401, 402, 403, 404, 422):
                    if 402 == status:
                        self.broke = True
                    return None
            else:
                data = result.data
                if 'observation' in data:
                    data['observation'] = _flat_records(data['observation'])
                if isinstance(data.get('step'), int):
                    self.step = data['step']
                else:
                    self.step = self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools():
                    if not (isinstance(seen, dict) and seen.get('error')):
                        for value in args.values():
                            self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
        return None

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            held_75 = '- {}: {} | parameters: {}'.format(fn['name'], fn.get('description', ''), _dump(fn.get('parameters') or {}))
            lines.append(held_75)
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        held_76 = self.ask_json(prompt, [LIGHT], 4000)
        if not held_76:
            held_76 = {}
        brief = held_76
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            held_77 = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
            if not held_77:
                held_77 = brief
            brief = held_77
        elif not self.no_credit:
            if self.shared_shelf(brief):
                held_78 = self.ask_json(prompt, [HEAVY], 2500)
                if not held_78:
                    held_78 = {}
                again = held_78
                if _items(again.get('items')):
                    if isinstance(again.get('roles'), dict):
                        if again['roles'].get('buy'):
                            brief = again
        self.brief = brief
        if isinstance(brief.get('roles'), dict):
            roles = brief.get('roles')
        else:
            roles = {}
        self.roles = dict(((k, v) for k, v in roles.items() if isinstance(v, str) and v in self.schemas))
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy:
            if refuse == buy:
                if not brief['buy_args'] or not brief['refuse_args'] or brief['refuse_args'] == brief['buy_args']:
                    ask = 'The store tool {} both checks out and ends without a purchase, depending on its arguments. Its schema: '.format(buy) + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
                    held_79 = self.ask_json(ask, [LIGHT, HEAVY], 1500)
                    if not held_79:
                        held_79 = {}
                    fixed = held_79
                    if isinstance(fixed.get('buy_args'), dict):
                        if isinstance(fixed.get('refuse_args'), dict):
                            if fixed['refuse_args'] != fixed['buy_args']:
                                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _items(brief.get('items')):
            if isinstance(entry, dict):
                held_80 = entry.get('quantity')
                quantity = _count(held_80)
                if quantity and 0 < quantity:
                    count = quantity
                else:
                    count = 1
                musts = list((str(m) for m in _items(entry.get('musts')) if m))
                held_82 = str(entry.get('name') or 'the product')
                held_83 = str(entry.get('shelf') or '')
                held_81 = Item(held_82, held_83, count, musts)
                self.items.append(held_81)
            else:
                continue
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if not None is twin:
                became[old] = merged.index(twin) + 1
                twin.musts += list((m for m in item.musts if m not in twin.musts))
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
            else:
                merged.append(item)
                became[old] = len(merged)
        self.items = merged
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part:
                    if main:
                        if main != part:
                            if 1 <= part <= len(self.items):
                                if 1 <= main <= len(self.items):
                                    if (part, main) not in self.bound:
                                        self.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text:
                if (number, text) not in self.refs:
                    self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        if isinstance(budget, str):
            if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
                budget = float(budget)
        if isinstance(budget, (int, float)):
            if not isinstance(budget, bool):
                if 0 < budget:
                    self.budget = budget
        held_84 = brief.get('messages')
        limit = _count(held_84)
        if None is not limit and 0 <= limit:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        held_85 = brief.get('steps')
        steps = _count(held_85)
        if None is not steps:
            if 10 <= steps < self.max_steps:
                self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict):
                if str(entry.get('step') or '').strip():
                    when = str(entry.get('when') or '').strip().lower()
                    self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = list((p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']))
            if talk:
                self.reserve = min(max(talk), self.msg_limit)
            else:
                self.reserve = 0

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = list((_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)))
        shelves = list((s for s in shelves if s))
        held_86 = len(shelves) != len(set(shelves))
        return held_86

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = list((need for n, need in self.refs if number == n and need in item.musts))
            rest = list((m for m in item.musts if m not in pointers))
            if not (not pointers or not rest):
                prompt = UNMIX + '\n\nREQUEST:\n' + self.request + '\n\nPRODUCT: {}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n'.format(item.name) + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
                held_87 = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
                if not held_87:
                    held_87 = {}
                got = held_87
                moved = list((m for m in rest if m in list((str(x).strip() for x in _items(got.get('describe_other'))))))
                if moved:
                    item.musts = list((m for m in item.musts if m not in moved))
                    joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
                    item.musts = list((joined if pointers[0] == m else m for m in item.musts))
                    self.refs = list(((n, joined if (number, pointers[0]) == (n, need) else need) for n, need in self.refs))
                    self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})
                else:
                    continue
            else:
                continue

    def described(self, number: int, need: str) -> bool:
        if not (need in list((m for _, m in self.refs)) or need in list((str(x) for x in _items(self.brief.get('lookups'))))):
            prop, _, value = need.partition(':')
            if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
                return False
            elif not re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
                item = self.items[number - 1]
                if any((set((w for w in _stem(other.name) if 4 <= len(w))) & _stem(value) for other in self.items if item is not other)):
                    return False
                cat_key = self.category_key()
                keys: set[str] = set()
                read = 0
                for record in self.full.values():
                    if not (cat_key and (not self.owns(item, str(record.get(cat_key) or '')))):
                        held_88 = _fact_block(record)
                        if not held_88:
                            held_88 = {}
                        block = held_88
                        if block:
                            read = read + 1
                            keys |= set((_words(k) for k in block))
                    else:
                        continue
                held_89 = 3 <= read and (not any((_stem(k) & _stem(prop) for k in keys)))
                return held_89
            else:
                return False
        else:
            return False

    def ledger(self) -> str:
        parts = []
        many = 1 < len(self.items)
        for number, item in enumerate(self.items, 1):
            live = list((m for m in item.musts + item.told if m not in item.dropped))
            told = list((m for m in live if self.described(number, m)))
            needs = list((m + (' (the page it points at gives: {})'.format(self.ref_values[m]) if m in self.ref_values else '') for m in live if m not in told))
            if many:
                head = 'item {}: {}'.format(number, item.name)
            else:
                head = 'item: {}'.format(item.name)
            if item.shelf:
                head += ' (store label: {})'.format(item.shelf)
            head += ', quantity {}'.format(item.qty)
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            parts.append(text)
        brief = self.brief
        if None is not self.budget:
            if brief.get('budget_scope') == 'each':
                scope = 'for each item'
            else:
                scope = 'in total'
            held_90 = 'budget: at most {} {} {}'.format(self.budget, brief.get('budget_unit') or '', scope).replace('  ', ' ')
            parts.append(held_90)
        pick = brief.get('pick')
        if 'cheapest' == pick:
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif not pick in ('most', 'least'):
            parts.append('pick: any listing that fits everything')
        else:
            held_91 = 'pick: the {} {} among fitting listings'.format(pick, brief.get('pick_measure') or '').replace('  ', ' ')
            parts.append(held_91)
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        cart_tool = self.roles.get('cart')
        held_92 = (self.schemas.get(cart_tool or '') or {}).get('properties')
        if not held_92:
            held_92 = {}
        cart_props = held_92
        count_arg = next((k for k, v in cart_props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        if count_arg:
            if re.search('\\b(?:line|row|index|slot|position|place)\\b', _words(count_arg)):
                if not re.search('qty|quant|count|unit|piece|amount|how many|wanted|needed|copies', _words(count_arg)):
                    parts.append('cart tool {}: its argument {} is a number of units, not a line position: '.format(cart_tool, count_arg) + ('the units the line holds (0 takes the line out)' if self.cart_sets(cart_tool) else 'the units the call adds or removes'))
        for wish in _items(brief.get('wishes')):
            held_93 = 'wish (tie-breaker only): {}'.format(wish)
            parts.append(held_93)
        for rule in _items(brief.get('rules')):
            held_94 = 'rule from the request: {}'.format(rule)
            parts.append(held_94)
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Halo does it when you check out)', 'end': 'at an ending without a purchase (Halo does it then)'}[entry['when']]
            held_95 = 'step the request asks for, {}: {}'.format(when, entry['step'])
            parts.append(held_95)
        facts = self.facts_tool()
        if facts:
            if self.facts_from.get(facts) is True or self.roles.get('offer'):
                bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
                pages = sorted((t for t in self.page_tools() if facts != t))
                parts.append('listing attributes come from {}'.format(facts) + ('; {} show none'.format(', '.join(bare)) if bare else '') + ("; a listing's page (what a chain of pages or a page check means) is shown by {}".format(' or '.join(pages)) if pages else ''))
        held_96 = '\n'.join(('- ' + p for p in parts))
        return held_96

    def status(self) -> str:
        left = self.left()
        text = '[Halo] step {} of {} used, {} left.'.format(self.step, self.max_steps, left)
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif 6 >= left or (len(self.items) + 5 >= left and 0 == self.buys):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif 0 == self.buys and 12 >= left:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif 0 == self.buys:
            if not self.cart_touched:
                if self.max_steps // 2 <= self.step:
                    text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock() >= WIND_DOWN:
            if left > self.tail() + 1:
                text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        live = set((need for _, need in self.needs_now()))
        lookups = list((need for _, need in self.refs if need in live and need not in self.ref_values))
        if lookups:
            if 0 == self.buys:
                text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        if self.bound and 0 == self.buys:
            plan = self.fit_plan()
        else:
            plan = ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or 3 > len(node):
            return None
        elif not (self.id_key and _id_like(node.get(self.id_key), True)):
            nested = any((isinstance(v, dict) for v in node.values()))
            if in_list or 0 == depth or nested:
                found = _learn_id([node])
                if found:
                    if 'id' in str(found).lower() or node[found].strip() in self.rows:
                        held_98 = node[found].strip()
                        return held_98
                return None
            else:
                return None
        else:
            held_97 = node[self.id_key].strip()
            return held_97

    def learn_rows(self, node: list[Any]) -> bool:
        rows = list((n for n in node if isinstance(n, dict) and 3 <= len(n)))
        if not rows or len(node) / 2 > len(rows):
            return False
        elif not (self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows))):
            found = _learn_id(rows)
            if not found:
                return False
            else:
                self.id_key = found
                return True
        else:
            return True

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if 6 < depth:
                return
            elif isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if not ident:
                    for value in node.values():
                        walk(value, depth + 1)
                else:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any((old.get(k) is True and False is v for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif any((old.get(k) is False and True is v for k, v in node.items())):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if not (6 < depth or not isinstance(schema, dict)):
                for option in schema.get('enum') or []:
                    if isinstance(option, str):
                        held_99 = option.strip().lower()
                        found.add(held_99)
                for value in schema.values():
                    if isinstance(value, dict):
                        walk(value, depth + 1)
            else:
                return
        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str):
                    if 0 < len(value) <= 40:
                        if not re.search('\\d{3}', value):
                            counts.setdefault(key, set()).add(value)
                            present[key] = present.get(key, 0) + 1
        held_100 = len(self.rows)
        total = max(1, held_100)
        known = self.labels()
        best = None
        for key, values in counts.items():
            if not (0.8 * total > present[key] or 40 < len(values)):
                listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
                mean = sum((len(v) for v in values)) / len(values)
                score = (-listed, len(values), -present[key], -mean)
                if None is best or best[0] > score:
                    best = (score, key)
            else:
                continue
        if best:
            held_101 = best[1]
            return held_101
        return None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if record:
                held_102 = _fact_block(record)
                if not held_102:
                    held_102 = {}
                block = held_102
                if cat_key:
                    label = str(record.get(cat_key) or 'listings')
                else:
                    label = 'listings'
                group = groups.setdefault(label, {'n': 0, 'keys': {}})
                group['n'] += 1
                done: set[str] = set()
                for key, value in block.items():
                    name = _words(key)
                    if not (not name or name in done):
                        done.add(name)
                        slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                        slot['n'] += 1
                        text = str(value).strip().lower()
                        slot['per'][ident] = text
                        if text not in slot['values']:
                            slot['values'].append(text)
                    else:
                        continue
            else:
                continue
        wanted = list(((_words(item.shelf or ''), _words(item.name), set((w for w in _stem(item.shelf or '') | _stem(item.name) if 4 <= len(w)))) for item in self.items))
        matched = set((label for label in groups if any((_words(label) in (shelf, name) or set((w for w in _stem(label) if 4 <= len(w))) & long for shelf, name, long in wanted)) or self.owner_of(label)))
        if matched:
            groups = dict(((label, group) for label, group in groups.items() if label in matched))
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                held_103 = slot['per'].values()
                values = list(held_103)
                if values:
                    mean = sum((len(v) for v in values)) / len(values)
                    if not (40 < mean or sum((1 for v in values if ' / ' in v)) * 2 >= len(values)):
                        recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if v is not o))))) > 0 for v in values))
                        if not (4 <= len(values) and len(set(values)) == len(values) and (12 < mean) and (len(values) > recur * 2)):
                            twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                            if None is not twin:
                                for ident, text in slot['per'].items():
                                    twin['per'].setdefault(ident, text)
                                    if text not in twin['values']:
                                        twin['values'].append(text)
                                twin['n'] = len(twin['per'])
                                continue
                            kept.append((name, slot))
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], list((v[:40] for v in slot['values'][:4]))))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if number:
            for must in self.items[number - 1].musts:
                prop, _, value = must.partition(':')
                if not re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                    held_105 = value.lower()
                    held_104 = re.findall('[a-z0-9]{3,}', held_105)
                    words = set(held_104)
                    held_107 = ' '.join(values).lower()
                    held_106 = re.findall('[a-z0-9]{3,}', held_107)
                    shown = set(held_106)
                    if prop.strip():
                        if _words(prop) != name:
                            if _stem(prop) & _stem(name):
                                if words:
                                    if not shown >= words:
                                        held_108 = ' [not the stated {}: another property]'.format(prop.strip())
                                        return held_108
                else:
                    continue
            return ''
        else:
            return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not (not a or not b or (not a & b)):

            def agree(x: str, y: str) -> bool:
                wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
                short, long = (wx, wy) if len(wy) >= len(wx) else (wy, wx)
                held_109 = y == x or (bool(short) and set(short) <= set(long) and (2 * len(short) + 2 >= len(long)))
                return held_109
            both = list((i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()))
            same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
            if not (b >= a or a >= b):
                held_110 = 2 <= len(both) and len(both) == same
                return held_110
            elif not both:
                held_112 = bool((set(spec['values']) & set(slot['values'])) - {''})
                return held_112
            else:
                held_111 = len(both) < same * 2
                return held_111
        else:
            return False

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            if cat_key:
                label = str(self.rows[ident].get(cat_key) or '')
            else:
                label = ''
            groups.setdefault(label, []).append(ident)
        wanted = list(((_words(item.shelf or ''), _words(item.name), set((w for w in _stem(item.shelf or '') | _stem(item.name) if 4 <= len(w)))) for item in self.items))
        matched = list((ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or set((w for w in _stem(label) if 4 <= len(w))) & long for shelf, name, long in wanted)) or self.owner_of(label)))
        held_113 = matched
        if not held_113:
            held_114 = groups.values()
            held_113 = list(held_114)
        pool = held_113
        held_115 = sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]
        return held_115

    def single_target(self, tool: str | None) -> str | None:
        held_116 = self.schemas.get(tool or '')
        if not held_116:
            held_116 = {}
        schema = held_116
        held_117 = schema.get('properties')
        if not held_117:
            held_117 = {}
        props = held_117
        texts = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)))
        held_118 = next((k for k in schema.get('required') or [] if k in texts), None)
        if not held_118:
            if 1 == len(texts):
                held_118 = texts[0]
            else:
                held_118 = None
        target = held_118
        if not (not tool or not target or any((target != k for k in schema.get('required') or []))):
            return target
        else:
            return None

    def price_of(self, ident: str) -> float | None:
        if self.price_key:
            value = (self.rows.get(ident) or {}).get(self.price_key)
        else:
            value = None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
                value = float(value)
        if isinstance(value, (int, float)) and (not isinstance(value, bool)):
            held_119 = float(value)
            return held_119
        return None

    def buyable(self, ident: str) -> bool:
        if not self.sold_out(ident):
            if self.stock_key:
                value = (self.rows.get(ident) or {}).get(self.stock_key)
            else:
                value = None
            held_120 = not (False is value or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (0 >= value)))
            return held_120
        else:
            return False

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if not ident:
                found: list[tuple[str, dict[str, Any]]] = []
                for value in observation.values():
                    if isinstance(value, list):
                        found += list(((i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i))
                    elif isinstance(value, dict):
                        found += list(((i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i))
                return found
            else:
                held_121 = [(ident, observation)]
                return held_121
        elif not isinstance(observation, list):
            held_123 = []
            return held_123
        else:
            held_122 = list(((i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i))
            return held_122

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if self.version == version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        held_124 = bool(keys) and self.version == version and (have >= keys) and (ident not in self.gone)
        return held_124

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            held_125 = self.shapes[name]
            return held_125
        elif isinstance(observation, dict):
            lists = list((k for k, v in observation.items() if isinstance(v, list) and 2 <= len(v) and all((isinstance(r, dict) for r in v))))
            if lists:
                self.shapes[name] = None
                sample = dict(((k, v[:3] if k in lists else v) for k, v in observation.items()))
                prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
                held_126 = self.ask_json(prompt, [LIGHT, HEAVY], 2500)
                if not held_126:
                    held_126 = {}
                got = held_126
                if got.get('rows') in lists:
                    rows = got.get('rows')
                else:
                    rows = None
                if not rows:
                    return None
                first = observation[rows][0]
                held_127 = (self.schemas.get(name) or {}).get('properties')
                if not held_127:
                    held_127 = {}
                props = held_127

                def named(key: Any, where: dict[str, Any]) -> str | None:
                    if isinstance(key, str) and key in where:
                        return key
                    return None
                shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
                if shape['count']:
                    if isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int):
                        shape['count'] = None
                self.shapes[name] = shape
                self.price_key = self.price_key or shape['price']
                self.stock_key = self.stock_key or shape['stock']
                self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
                return shape
            else:
                return None
        else:
            return None

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not (not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict))):
            rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
            if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
                return None
            elif not (None is not value and (isinstance(value, bool) or not isinstance(value, (str, int)))):
                held_128 = (count, len(rows), shape['next_arg'], value)
                return held_128
            else:
                return None
        else:
            return None

    def price_sorted(self, observation: Any) -> bool:
        prices = list((self.price_of(i) for i, _ in self.records_in(observation)))
        held_129 = 3 <= len(prices) and None not in prices and all((b >= a for a, b in zip(prices, prices[1:])))
        return held_129

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        if pick in ('cheapest', 'most', 'least'):
            plan = self.paging(name, observation)
        else:
            plan = None
        if plan:
            count, shown, param, value = plan
            given = args.get(param)
            if isinstance(given, int) and (not isinstance(given, bool)):
                later = given not in (None, 0, 1)
            else:
                later = bool(str(given or '').strip())
            if None is value or later or shown >= count:
                return ''
            elif not ('cheapest' == pick and self.price_sorted(observation)):
                pages = -(-(count - shown) // shown)
                if 2 < pages:
                    if AUTO_PAGES >= self.pages_read + 2:
                        if self.left() - self.tail() - 2 >= 12:
                            pages = 2
                if 2 < pages or AUTO_PAGES < self.pages_read + pages or self.left() - self.tail() - pages < 12:
                    held_130 = "\n[Halo] This search matched {} listings and showed {}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {}={}.".format(count, shown, param, _dump(value))
                    return held_130
                texts = []
                for _ in range(pages):
                    faults: list[str] = []
                    call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
                    if not faults:
                        reply = self.act(name, call)
                        if not (None is reply or self.finished):
                            self.pages_read = self.pages_read + 1
                            seen = reply.get('observation')
                            self.observe(name, seen)
                            self.paged_obs.append(seen)
                            shown_text = self.show(seen)
                            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
                            held_132 = _dump(value)
                            held_131 = '{}={}: {}'.format(param, held_132, shown_text)
                            texts.append(held_131)
                            after = self.paging(name, seen)
                            if after:
                                value = after[3]
                            else:
                                value = None
                            if None is value:
                                break
                        else:
                            break
                    else:
                        break
                self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
                if None is not value and texts:
                    rest = "\n[Halo] {} listings matched and Halo has read {} rows of them; narrow the search with the requirement's value or page on to see the rest.".format(count, shown * (len(texts) + 1))
                else:
                    rest = ''
                if texts:
                    held_133 = '\n[Halo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest
                    return held_133
                return ''
            else:
                return ''
        else:
            return ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        held_134 = (self.schemas.get(name) or {}).get('properties')
        if not held_134:
            held_134 = {}
        props = held_134

        def priced(key: str) -> bool:
            if isinstance(props.get(key), dict):
                spec = props.get(key)
            else:
                spec = {}
            held_135 = bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
            return held_135
        caps = list((k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)))
        if not (not caps or not self.price_key):
            caps.sort(key=lambda k: args[k])
            cap = args[caps[0]]
            over = list((i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap))
            if over:
                modes = dict(((k, v) for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)))
                others = dict(((k, list((o for o in props[k]['enum'] if v != o))) for k, v in modes.items()))
                if any(others.values()):
                    hint = '; list the category with ' + ', '.join(('{}={}'.format(k, _dump(o[0])) for k, o in others.items() if o)) + ' and the same {} to see only cheaper listings'.format(caps[0])
                else:
                    hint = ''
                held_136 = '\n[Halo] This search did not apply {}={:g}: {} of its rows cost more{}.'.format(caps[0], cap, len(over), hint)
                return held_136
            else:
                return ''
        else:
            return ''

    def cheapest_here(self, observations: list[Any]) -> str:
        if not (self.brief.get('pick') != 'cheapest' or not self.price_key):
            found = set((i for seen in observations for i, _ in self.records_in(seen)))
            rows = list(((self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None))
            if not 2 > len(rows):
                held_137 = '\n[Halo] buyable rows here, cheapest first: ' + ', '.join(('{} ({:g})'.format(i, p) for p, i in sorted(rows)[:6]))
                return held_137
            else:
                return ''
        else:
            return ''

    def wider_hold(self) -> str:
        if not (self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items)):
            self.wider_held = True
            lines = list(('{} at {:g}'.format(i, self.price_of(i)) for i in self.cart_ids() if self.price_of(i) is not None))
            held_138 = "Not run yet (no step used). {} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line".format(self.left()) + (' ({})'.format('; '.join(lines)) if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'
            return held_138
        else:
            return ''

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not (not tool or not target):
            texts = []
            for ident in list((i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i)))):
                if not (self.left() <= keep or self.finished or self.facts_from.get(tool) is False):
                    faults: list[str] = []
                    args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.act(tool, args)
                        if not None is reply:
                            seen = reply.get('observation')
                            self.observe(tool, seen)
                            self.note_view(tool, seen)
                            if isinstance(seen, dict):
                                if self.ident(seen):
                                    self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
                            texts.append('{}: '.format(ident) + self.show(seen)[:1500])
                        else:
                            break
                    else:
                        break
                else:
                    break
            if texts:
                self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
            return texts
        else:
            held_139 = []
            return held_139

    def show_links(self, ids: list[str], keep: int) -> None:
        plan = self.value_template()
        if not (not plan or (2 > len(self.items) and 2 > len(ids))):
            tool, base, field = plan
            for ident in ids:
                marks = self.fit_index()['marks']
                code = (marks.get(ident) or {}).get('more')
                if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())):
                    continue
                elif not (self.left() <= keep or self.finished):
                    self.link_tried.add(code)
                    faults: list[str] = []
                    args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.act(tool, args)
                        if None is reply:
                            return
                        held_140 = reply.get('observation')
                        self.observe(tool, held_140)
                        self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
                        marks = self.fit_index()['marks']
                        held_141 = list((i for i, mark in marks.items() if mark.get('own') == code))
                        held_142 = max(keep, self.tail() + 6)
                        self.fill_facts(held_141, held_142)
                    else:
                        return
                else:
                    return
        else:
            return

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        for tool, schema in self.schemas.items():
            held_143 = (schema or {}).get('properties')
            if not held_143:
                held_143 = {}
            props = held_143
            caps = list((k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and re.search(CAP_ARG, _snake(k), re.I) and (not re.search(CAP_NOT, k, re.I))))
            shelves = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(SHELF_ARG, _snake(k), re.I)))
            if not (1 != len(caps) or 1 != len(shelves)):
                base: dict[str, Any] = {}
                for key, spec in props.items():
                    if isinstance(spec, dict):
                        options = spec.get('enum')
                    else:
                        options = None
                    if not (not isinstance(options, list) or 2 > len(options)):
                        pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and (not any((shelves[0] != k and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())))), None)
                        named = list((o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)))
                        if None is not pick:
                            pick = pick
                        else:
                            pick = named[0] if 1 == len(named) else None
                        if not None is pick:
                            base[key] = pick
                        else:
                            return None
                    else:
                        continue
                held_144 = (tool, base, shelves[0], caps[0])
                return held_144
            else:
                continue
        return None

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not (not plan or not cat_key):
            tool, base, shelf, cap = plan
            for line in cart:
                price = self.price_of(line)
                label = str((self.rows.get(line) or {}).get(cat_key) or '')
                if None is price or 2 > price or (not label) or ((line, price) in self.capped):
                    continue
                elif not (self.left() <= self.tail() + 5 or self.finished):
                    self.capped.add((line, price))
                    faults: list[str] = []
                    args = _shape({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.act(tool, args)
                        if None is reply:
                            return
                        held_145 = reply.get('observation')
                        self.observe(tool, held_145)
                        self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
                        held_146 = reply.get('observation')
                        page = self.paging(tool, held_146)
                        if page:
                            if None is not page[3]:
                                if page[1] < page[0] <= 2 * page[1]:
                                    if self.left() > self.tail() + 5:
                                        if not self.price_sorted(reply.get('observation')):
                                            call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                                            if faults:
                                                more = None
                                            else:
                                                more = self.act(tool, call)
                                            if None is not more:
                                                held_147 = more.get('observation')
                                                self.observe(tool, held_147)
                    else:
                        return
                else:
                    return
        else:
            return

    def cheaper_rows(self) -> list[str]:
        if not self.brief.get('pick') != 'cheapest':
            mark = self.cart_mark() + '#{}'.format(self.version)
            cat_key = self.category_key()
            if not (mark in self.cheaper_marks or not cat_key):
                self.cheaper_marks.add(mark)
                cart = self.cart_ids()
                self.list_below(cart)
                picks: list[str] = []
                for line in cart:
                    price = self.price_of(line)
                    kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
                    if not (None is price or not kind):
                        pool = list((i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)))
                        pool.sort(key=lambda i: self.price_of(i) or 0.0)
                        picks += list((i for i in pool[:2] if i not in picks))
                    else:
                        continue
                held_150 = self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)
                return held_150
            else:
                held_149 = []
                return held_149
        else:
            held_148 = []
            return held_148

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if lines:
            meaning = self.take_note(lines)
            if meaning:
                held_151 = '(seen on {}) {}'.format(name, meaning)
                self.queued.append(held_151)
            return meaning
        else:
            return ''

    def widen(self) -> None:
        tool = self.roles.get('compare')
        held_152 = self.schemas.get(tool or '')
        if not held_152:
            held_152 = {}
        schema = held_152
        held_153 = schema.get('properties')
        if not held_153:
            held_153 = {}
        props = held_153
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((target != k for k in schema.get('required') or [])) or self.compare_bare():
            self.widen_singly()
            return
        elif not self.left() <= 12:
            if isinstance(props[target].get('maxItems'), int):
                cap = props[target].get('maxItems')
            else:
                cap = 4
            ranked = self.item_groups()
            share = max(1, cap // max(1, len(ranked)))
            picks: list[str] = []
            for ids in ranked:
                picks += list((i for i in ids if i not in self.full))[:share]
            picks = picks[:cap]
            if not 2 > len(picks):
                faults: list[str] = []
                args = _shape({target: picks}, schema, 'args', faults)
                if faults:
                    return
                before = len(self.full)
                reply = self.act(tool, args)
                if None is not reply:
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
                    elif before == len(self.full):
                        if 2 < len(picks):
                            if self.left() > 12:
                                faults = []
                                args = _shape({target: picks[:2]}, schema, 'args', faults)
                                if not faults:
                                    reply = self.act(tool, args)
                                    if None is not reply:
                                        held_154 = reply.get('observation')
                                        self.observe(tool, held_154)
            else:
                return
        else:
            return

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not (not tool or not target):
            wanted = set((w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())))
            if 2 >= len(self.items):
                each = WIDEN_EACH
            else:
                each = 4
            lanes: list[list[str]] = []
            heads: list[int] = []
            for ids in self.item_groups():
                opened = sum((1 for i in ids if i in self.full))
                pool = list((i for i in ids if i not in self.full and self.buyable(i)))
                if self.brief.get('pick') == 'cheapest':
                    pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
                pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
                lanes.append(pool[:max(0, each - opened)])
                held_155 = max(0, 4 - opened)
                heads.append(held_155)
            first = list((lane[k] for k in range(4) for lane, head in zip(lanes, heads) if min(head, len(lane)) > k))
            more = list((lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)))
            cap = max(5, 2 * len(lanes) + 2)
            read = self.fill_facts(first[:cap][:8], 22)
            self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)
        else:
            return

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        elif not (self.left() <= 8 or self.clock() > ASK_BY):
            opened = len(self.full)
            if not 3 * len(self.items) + 1 <= opened:
                held_156 = max(8, self.max_steps // 4) <= self.step
                return held_156
            else:
                return True
        else:
            return False

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        held_157 = self.schemas.get(name or '')
        if not held_157:
            held_157 = {}
        schema = held_157
        held_158 = schema.get('properties')
        if not held_158:
            held_158 = {}
        props = held_158
        texts = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)))
        held_159 = next((k for k in schema.get('required') or [] if k in texts), None)
        if not held_159:
            if texts:
                held_159 = texts[0]
            else:
                held_159 = None
        target = held_159
        if target:
            args: dict[str, Any] = {target: text}
            for key in schema.get('required') or []:
                if key not in args:
                    held_160 = props.get(key)
                    if not held_160:
                        held_160 = {}
                    spec = held_160
                    if not (isinstance(spec.get('enum'), list) and spec['enum']):
                        return None
                    else:
                        args[key] = spec['enum'][0]
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            if faults:
                return None
            return args
        else:
            return None

    def interview(self) -> str:
        self.asked = True
        if not self.msgs_free() <= 0:
            self.widen()
            if LIST_MESSAGES >= self.msg_limit:
                self.listing = False
                self.trace.append({'kind': 'one_property_each', 'messages': self.msg_limit})
            count = self.msg_limit - self.msg_used
            self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
            self.lane_found = dict(((n, 0) for n in self.lanes))
            self.lane_spent = dict(((n, 0) for n in self.lanes))
            self.lane_used = dict(((n, set()) for n in self.lanes))
            self.follow_up = []
            sent: list[tuple[int, str, str]] = []
            told: list[str] = []
            while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
                batch = self.ask_next(told)
                if batch:
                    sent += batch
                else:
                    break
            found_items = list((n for n in self.lane_found if 0 < self.lane_found[n]))
            if found_items:
                searched = self.value_search(found_items)
            else:
                searched = ''
            if told:
                self.version = self.version + 1
            lines = list(('Q: {}\nA: {}'.format(q, a) for _, q, a in sent))
            summary = "Halo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
            summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
            summary += '. Apply them to every pick (search again with the new values where needed).'
            pointed = list((need for _, need in self.refs if any((need in t for t in told))))
            if pointed:
                summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
            if searched:
                summary += '\n' + searched
            if self.queued:
                summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
                self.queued = []
            return summary
        else:
            return ''

    def late_question(self) -> str:
        if self.late_asked or not self.roles.get('ask') or self.msg_limit <= self.msg_used:
            return ''
        elif any(('order' == p['when'] and 0 < p['messages'] for p in self.process)):
            return ''
        elif not (self.left() < LATE_STEPS or self.clock() > LATE_BY or (not self.judges())):
            self.late_asked = True
            if not self.lanes:
                self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
                self.lane_found = dict(((n, 0) for n in self.lanes))
                self.lane_spent = dict(((n, 0) for n in self.lanes))
                self.lane_used = dict(((n, set()) for n in self.lanes))
            told: list[str] = []
            budget = self.budget
            batch = self.ask_next(told, LATE_STEPS - 2)
            if not batch:
                self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
                return ''
            if budget != self.budget:
                held_161 = 'budget now {}'.format(self.budget)
                told.append(held_161)
            number, question, answer = batch[0]
            self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
            if not told:
                return ''
            self.version = self.version + 1
            summary = 'Halo asked the shopper one more question before checking out (one step):\nQ: {}\nA: {}\nNew firm requirements from this answer: '.format(question, answer) + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
            pointed = list((need for _, need in self.refs if any((need in t for t in told))))
            if pointed:
                summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
            searched = self.value_search([number])
            held_162 = summary + ('\n' + searched if searched else '')
            return held_162
        else:
            return ''

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        held_163 = self.schemas.get(tool or '')
        if not held_163:
            held_163 = {}
        schema = held_163
        held_164 = schema.get('properties')
        if not held_164:
            held_164 = {}
        props = held_164
        texts = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)))
        named = list((k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)))
        for args in reversed(self.searched):
            fields = list((k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)))
            if not named:
                if self.category_key():
                    labels = set((str(r.get(self.category_key()) or '').strip().lower() for r in self.rows.values())) - {''}
                else:
                    labels = set()
                fields = list((k for k in fields if len(set((str(a.get(k)) for a in self.searched if a.get(k)))) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))))
            if fields:
                if tool:
                    field = fields[0]
                    keep = dict(((k, v) for k, v in args.items() if field != k and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))))
                    cat_key = self.category_key()
                    shelves = set((_words(item.shelf or '') for item in self.items)) | set((_words(item.name) for item in self.items))
                    shelves |= set((_words(str(row.get(cat_key) or '')) for row in self.rows.values())) if cat_key else set()
                    keep = dict(((k, v) for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any((set((w for w in _stem(v) if 4 <= len(w))) & set((w for w in _stem(s) if 4 <= len(w))) for s in shelves if s))))))
                    worded = list((a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)))
                    for k in list(keep):
                        if isinstance((props.get(k) or {}).get('enum'), list):
                            used = list((_dump(a[k]) for a in worded if k in a))
                            if used:
                                best = max(used, key=used.count)
                                keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                    held_165 = (tool, keep, field)
                    return held_165
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        out = []
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if not (number != n or need in list((m for _, m in self.refs)) or ':' not in need):
                value = need.split(':', 1)[1].strip()
                edge = re.search(FLOOR, value, re.I)
                if edge:
                    if need in told:
                        out.append((edge.group(1).strip(), True))
                    continue
                elif not (not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I)):
                    out.append((value, need in told))
                else:
                    continue
            else:
                continue
        return out

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        plan = self.search_template()
        tool = self.roles.get('search')
        if not (plan or not tool or (not self.searched)):
            held_166 = self.schemas.get(tool)
            if not held_166:
                held_166 = {}
            schema = held_166
            held_167 = schema.get('properties')
            if not held_167:
                held_167 = {}
            props = held_167
            held_168 = schema.get('required')
            if not held_168:
                held_168 = []
            required = held_168
            named = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(QUERY_WORDS, _snake(k), re.I)))
            last = self.searched[-1]
            enums = dict(((k, v['enum']) for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (2 <= len(v['enum']))))
            modes = []
            for key, options in enums.items():
                reads = list((o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)))
                others = list((o for o in options if o != last.get(key)))
                if 1 == len(reads):
                    modes.append((key, reads[0]))
                elif key in required:
                    if 2 == len(options):
                        if 1 == len(others):
                            modes.append((key, others[0]))
            if not (1 != len(named) or 1 < len(modes) or (enums and (not modes))):
                base = dict(((k, last[k]) for k in required if k in last and named[0] != k))
                base.update(modes)
                held_169 = (tool, base, named[0])
                return held_169
            else:
                return None
        else:
            return plan

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        if not (not plan or self.left() < 22):
            tool, base, field = plan
            seen: list[str] = []
            budget = 8
            for number in numbers:
                pairs = self.value_words(number)
                item = self.items[number - 1]
                values = list((v for v, _ in pairs))
                fresh = list((v for v, told in pairs if told))
                if not (not fresh or 2 >= budget):
                    lead = list((v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)))
                    values = lead + list((v for v in values if v not in lead))
                    stated = list((i for i, v in enumerate(values) if v not in fresh))
                    if stated:
                        drop = stated[-1]
                    else:
                        drop = 0
                    if 1 < len(values):
                        tries = [values, values[:drop] + values[drop + 1:]]
                    else:
                        tries = [[values[0], item.name], values] if lead else [values]
                    for words in tries[:2]:
                        if words:
                            faults: list[str] = []
                            args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                            if not (faults or self.finished):
                                reply = self.act(tool, args)
                                budget = budget - 1
                                if not None is reply:
                                    observation = reply.get('observation')
                                    self.observe(tool, observation)
                                    cat_key = self.category_key()
                                    found = list((i for i, _ in self.records_in(observation)))
                                    if isinstance(self.budget, (int, float)):
                                        cap = self.budget
                                    else:
                                        cap = None
                                    prices = list((self.price_of(i) for i in found))
                                    if None is not cap:
                                        if prices:
                                            if all((None is not p and 10 * cap < p for p in prices)):
                                                cap = None
                                    rows = list((i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (None is cap or (self.price_of(i) or 0) <= cap)))
                                    home = self.home_label(number)
                                    if cat_key:
                                        if home:
                                            rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                                    seen.append('{}: '.format(_dump(args.get(field))) + (', '.join(rows[:6]) or 'no row of that kind'))
                                    if rows:
                                        opened = self.fill_facts(list((i for i in rows if i not in self.full))[:min(3, budget)], self.tail() + 16)
                                        budget -= len(opened)
                                        if opened:
                                            seen.append('opened: ' + ' | '.join(opened)[:2400])
                                        break
                                else:
                                    break
                            else:
                                break
                        else:
                            continue
                else:
                    continue
            if seen:
                self.trace.append({'kind': 'value_search', 'seen': list((s[:200] for s in seen))})
                held_170 = "[Halo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)
                return held_170
            else:
                return ''
        else:
            return ''

    def home_label(self, number: int) -> str:
        cat_key = self.category_key()
        if cat_key:
            item = self.items[number - 1]
            for pool in (self.full, self.rows):
                counts: dict[str, int] = {}
                for record in pool.values():
                    label = str(record.get(cat_key) or '')
                    if label:
                        if self.owns(item, label):
                            counts[_words(label)] = counts.get(_words(label), 0) + 1
                for name in (item.shelf, item.name):
                    if _words(name) in counts:
                        held_171 = _words(name)
                        return held_171
                if counts:
                    held_172 = max(counts, key=lambda label: counts[label])
                    return held_172
            return ''
        else:
            return ''

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = dict(((n, i) for i, n in enumerate(self.plan_order)))
        held_173 = SCARCE_MESSAGES >= self.msg_limit
        if held_173:
            held_173 = not self.listing
        scarce = held_173
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = list((label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))))
            if left:
                held_174 = (number, left[:LIST_SIZE if self.listing else 1])
                return held_174
        held_175 = (0, [])
        return held_175

    def told_about(self, number: int, label: str) -> bool:
        mine = _spelled(label)
        held_176 = any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))
        return held_176

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if not 2 <= len(labels):
            held_178 = self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)
            return held_178
        else:
            held_177 = 'What {} or {} would you like for the {}?'.format(', '.join(labels[:-1]), labels[-1], self.item_said(number))
            return held_177

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args:
                if text in args.values():
                    held_179 = (labels[:size], text)
                    return held_179
        held_180 = ([], '')
        return held_180

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        for _ in range(LIST_SIZE):
            self.follow_up = list((f for f in self.follow_up if not self.told_about(f[0], f[1][0])))
            if not self.follow_up:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                shown = False
            else:
                number, labels, shown = self.follow_up.pop(0)
            if labels:
                fitted, text = self.fit_ask(number, labels, shown)
                if not fitted:
                    if shown:
                        fitted, text = self.fit_ask(number, labels, False)
                if not fitted:
                    self.lane_used.setdefault(number, set()).add(_words(labels[0]))
                    self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
                else:
                    break
            else:
                held_181 = []
                return held_181
        else:
            held_182 = []
            return held_182
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if batch:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
            had = list((len(item.told) for item in self.items))
            heard = self.hear(batch, (LIGHT, HEAVY))
            held_185 = heard.get('free')
            held_184 = _items(held_185)
            free_said = bool(held_184)
            if 1 < len(labels):
                heard.pop('free', None)
            self.take_heard(heard, told)
            new = list((need for item, size in zip(self.items, had) for need in item.told[size:]))
            for n, (item, size) in enumerate(zip(self.items, had), 1):
                if size < len(item.told):
                    self.lane_found[n] = self.lane_found.get(n, 0) + 1
            crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
            if not crowded:
                if not new:
                    if ONE_AT_A_TIME.search(str(batch[0][2])):
                        crowded = True
                        self.listing = False
                        self.trace.append({'kind': 'one_at_a_time', 'item': number, 'labels': len(labels)})
            if 1 < len(labels):
                if not new:
                    if not crowded:
                        if not free_said:
                            if PUZZLED.search(str(batch[0][2])):
                                crowded = True
                                self.trace.append({'kind': 'puzzled_list', 'item': number})
            self.settle(number, labels, shown, new, crowded)
            return batch
        else:
            held_183 = []
            return held_183

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        held_186 = set()
        used = self.lane_used.setdefault(number, held_186)
        if 1 == len(labels):
            held_187 = _words(labels[0])
            used.add(held_187)
            if new or not crowded:
                return
            elif not (number not in self.one_product and self.name_parts(number)):
                waiting = list((label for label in self.lanes.get(number, []) if _words(label) not in used and (not self.told_about(number, label))))
                if not waiting:
                    head = labels[0].split()[-1]
                    if len(labels[0].split()) >= 2 and _words(head) not in used:
                        self.follow_up.insert(0, (number, [head], False))
                    elif not shown:
                        if len(self.samples(number, labels[0])) >= 2:
                            self.follow_up.insert(0, (number, labels, True))
                    return
                else:
                    self.trace.append({'kind': 'moved_on', 'item': number, 'from': labels[0], 'left': len(waiting)})
                    return
            else:
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
        elif not (crowded and (not new)):
            if new:
                hit = self.answered_label(labels, new)
            else:
                hit = -1
            if 0 <= hit:
                ahead = labels[hit + 1:]
            else:
                ahead = labels[:0:-1]
            used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
            rest = list((label for label in self.lanes.get(number) or [] if label not in labels))
            self.lane_misses[number] = 0 if 0 <= hit else self.lane_misses.get(number, 0) + 1
            self.lanes[number] = rest + ahead if self.lane_misses[number] and 0 == self.lane_misses[number] % 2 else ahead + rest
            self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})
        else:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            seen.setdefault(_words(key), set()).update((w for v in values for w in _roots(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = list((len(_spelled(label) & _spelled(prop)) for label in labels))
            if max(scores, default=0) == 0:
                scores = list((len(seen.get(_words(label), set()) & _roots(value)) for label in labels))
            if max(scores, default=0) > 0:
                held_189 = max(scores)
                held_188 = scores.index(held_189)
                hits.append(held_188)
        if hits:
            held_190 = min(hits)
            return held_190
        held_191 = -1
        return held_191

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = list((_stem(m.split(':', 1)[0]) for m in item.musts))
        rows = list(((label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number))
        if 1 == len(self.items):
            if not rows:
                rows = list(((label, key, carried, len(values)) for label, key, carried, values in table))
        rows.sort(key=lambda r: (-r[2], -r[3]))
        keys = list((key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))))
        held_192 = list((key for key in keys if not re.search(CATCH_ALL, key))) + list((key for key in keys if re.search(CATCH_ALL, key)))
        return held_192

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if not (2 > len(self.items) or not cat_key):
            labels = set((str(record.get(cat_key) or '') for record in self.full.values())) - {''}
            strays = list((label for label in labels if not self.owner_of(label)))
            orphans = list((n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))))
            if strays:
                if 1 == len(orphans):
                    held_193 = dict(((label, orphans[0]) for label in strays))
                    self.adopted.update(held_193)
                    self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})
        else:
            return

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        elif not label in self.adopted:
            words = _words(label)
            long = _roots(label)
            best, top = (0, 0.0)
            for number, item in enumerate(self.items, 1):
                if words == _words(item.shelf or ''):
                    score = 3.0
                elif not words == _words(item.name):
                    mine = _roots(item.shelf or '') | _roots(item.name)
                    score = len(long & mine) / max(1, len(long | mine))
                else:
                    score = 2.0
                if top < score:
                    best, top = (number, score)
            return best
        else:
            held_194 = self.adopted[label]
            return held_194

    def owns(self, item: Item, label: str) -> bool:
        held_195 = bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)
        return held_195

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        held_196 = sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))
        return held_196

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append('item {}: {} - settled by the request: '.format(number, item.name) + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if last != label:
                held_197 = 'category {!r}:'.format(label)
                listing.append(held_197)
                last = label
            listing.append('  {}. {} (on {}): '.format(index, name, carried) + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = list((str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()))
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        held_198 = self.ask_json(prompt, [HEAVY, LIGHT], 3500)
        if not held_198:
            held_198 = {}
        plan = held_198
        lanes: dict[int, list[str]] = dict(((n, []) for n in range(1, len(self.items) + 1)))
        hinted: dict[int, set[str]] = dict(((n, set()) for n in lanes))
        for pick in _items(plan.get('ask')):
            if isinstance(pick, dict):
                held_199 = max(1, _count(pick.get('item') or 1) or 1)
                held_200 = len(self.items)
                number = min(held_199, held_200)
                label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
                if not label or len(label.split()) > 5:
                    continue
                elif not _words(label) in ('option', 'options', 'variant', 'variants', 'variation', 'variations'):
                    if all((_words(label) != _words(other) for other in lanes[number])):
                        lanes[number].append(label)
                    if str(pick.get('hint') or '').strip():
                        held_201 = _words(label)
                        hinted[number].add(held_201)
                        if number not in self.plan_order:
                            self.plan_order.append(number)
                else:
                    self.trace.append({'kind': 'options_label_skipped', 'item': number})
                    continue
            else:
                continue
        for number in lanes:
            keys = self.lane_keys(number, table)
            held_202 = self.records_of_item(number)
            read = max(1, held_202)
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    held_203 = own.get(key, 0.0)
                    own[key] = max(held_203, carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                held_204 = next((k for k in own if plain(k) == plain(label)), label)
                return held_204

            def share_of(label: str) -> float:
                if not _words(label) in own:
                    near = list((s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))))
                    if near:
                        held_206 = max(near)
                        return held_206
                    return 0.0
                else:
                    held_205 = own[_words(label)]
                    return held_205
            hinted[number] = set((_words(spelled(h)) for h in hinted[number]))
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    held_207 = spelled(pick)
                    lane.append(held_207)
            for key in list((k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))))[:GUARD_KEYS]:
                if not any((_words(key) == _words(other) for other in lane[:3])):
                    weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                    if not (None is weak and 3 <= len(lane)):
                        placed = list((other for other in lane if _words(other) != _words(key)))
                        if not (None is not weak and 1 < len(self.items)):
                            held_211 = len(placed)
                            held_210 = min(held_211, weak if None is not weak else len(placed))
                            placed.insert(held_210, key)
                        else:
                            moved = placed[weak]
                            placed[weak] = key
                            held_209 = len(placed)
                            held_208 = min(3, held_209)
                            placed.insert(held_208, moved)
                        if not any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])):
                            lane = placed
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            lanes[number] = lane
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = list((x for x in lanes[number] if not re.search(CATCH_ALL, _words(x)))) + list((x for x in lanes[number] if re.search(CATCH_ALL, _words(x))))
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': dict(((n, v[:8]) for n, v in lanes.items()))})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        name = self.items[number - 1].name
        if re.search('\\band\\b', name, re.I):
            parts = list((p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()))
            if 2 <= len(parts) and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)):
                return parts
            held_213 = []
            return held_213
        else:
            held_212 = []
            return held_212

    def item_said(self, number: int) -> str:
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            held_214 = item.name
            return held_214
        elif not number not in self.one_product:
            cat_key = self.category_key()
            rows = list((row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))))
            named = list((_roots(' '.join((str(v) for k, v in row.items() if cat_key != k and isinstance(v, str)))) for row in rows))
            hits = dict(((p, sum((1 for words in named if _roots(p.split()[-1]) & words))) for p in parts))
            held_216 = max(parts, key=lambda p: (hits[p], -parts.index(p)))
            return held_216
        else:
            held_215 = re.sub('\\band\\b', 'or', item.name, flags=re.I)
            return held_215

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        held_217 = self.item_said(number)
        text = 'What {} would you like for the {}'.format(label, held_217)
        held_218 = text + (', for example {} or {}?'.format(samples[0], samples[1]) if samples else '?')
        return held_218

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if want == name:
                held_219 = list((v for v in values if v and v not in settled and (30 >= len(v))))[:2]
                return held_219
        held_220 = []
        return held_220

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if not (self.finished or self.left() <= floor or self.msg_limit <= self.msg_used):
                args = self.message_args(text)
                if args:
                    reply = self.act(tool, args)
                    if not None is reply:
                        self.msg_used = self.msg_used + 1
                        answer = reply.get('observation')
                        words = self.spoken(answer)
                        sent.append((number, text, words))
                        held_221 = 'step {}: message to the shopper {!r} -> reply {!r} (cart then: {})'.format(self.step, text, words[:300], self.cart_mark())
                        self.done_log.append(held_221)
                        self.done_steps.append(self.step)
                        self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
                        if isinstance(answer, dict):
                            if answer.get('error'):
                                break
                    else:
                        break
                else:
                    break
            else:
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = 1 < len(self.items)
        for entry in _items(heard.get('needs')):
            if not (not isinstance(entry, dict) or not str(entry.get('need') or '').strip()):
                held_222 = max(1, _count(entry.get('item')) or 1)
                held_223 = len(self.items)
                number = min(held_222, held_223)
                need = str(entry['need']).strip()
                if need not in self.items[number - 1].told:
                    self.items[number - 1].told.append(need)
                    told.append('item {}: {}'.format(number, need) if many else need)
                if entry.get('points') is True:
                    if (number, need) not in self.refs:
                        self.refs.append((number, need))
            else:
                continue
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict):
                if str(entry.get('property') or '').strip():
                    item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                    gone = _words(str(entry['property']).split(':')[0])
                    hit = list((m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone))
                    item.dropped.extend(hit)
                    told.extend(('dropped {}'.format(m) for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict):
                if entry.get('property'):
                    held_224 = max(1, _count(entry.get('item')) or 1)
                    held_225 = len(self.items)
                    number = min(held_224, held_225)
                    self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)):
            if not isinstance(heard.get('budget'), bool):
                if 0 < heard['budget']:
                    self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join(('item {}: {}'.format(n, it.name) for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(('{}. (about item {}) Q: {}\nA: {}'.format(i, n, q, a) for i, (n, q, a) in enumerate(sent, 1)))
        held_227 = list(models)
        held_226 = self.ask_json(prompt, held_227, 2500)
        if not held_226:
            held_226 = {}
        got = held_226
        if isinstance(got, dict):
            if isinstance(got.get('needs'), list):
                asked = ' '.join((str(q).lower() for _, q, _ in sent))
                answered = ' '.join((str(a).lower() for _, _, a in sent))
                filler = {'at', 'least', 'most', 'the', 'and', 'or', 'not', 'no', 'more', 'less', 'than', 'of', 'a', 'an'}
                kept = []
                for entry in got['needs']:
                    if isinstance(entry, dict):
                        if entry.get('points') is not True:
                            value = str(entry.get('need') or '').split(':', 1)[-1].strip().lower()
                            words = list((w for w in re.findall('[a-z0-9][a-z0-9.\\-]*', value) if 2 <= len(w) and w not in filler))
                            if value:
                                if value in asked:
                                    if words:
                                        if not any((w in answered for w in words)):
                                            self.trace.append({'kind': 'echo_dropped', 'need': str(entry.get('need'))[:120]})
                                            continue
                    kept.append(entry)
                got['needs'] = kept
        return got

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = list((p for p in self.process if stage == p['when']))
        if not (not steps or self.finished or (not self.gate_due(stage))):
            ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
            tools = list(('- {}: {} | parameters: {}'.format(t['function']['name'], t['function'].get('description', ''), _dump(t['function'].get('parameters') or {})) for t in self.tools if t['function']['name'] not in ending))
            said: list[str] = []
            nudged = False
            for _ in range(GATE_ROUNDS):
                prompt = GATE.replace('STAGE', 'place the order' if 'order' == stage else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join(('- {}'.format(p['step']) for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(('{}. {}'.format(i, line) for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + '\n\nMESSAGES LEFT: {}   STEPS LEFT: {} (the {} itself takes one)'.format(max(0, self.msg_limit - self.msg_used), self.left(), 'order' if 'order' == stage else 'ending') + '\n\nTOOLS:\n' + '\n'.join(tools)
                got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
                if not None is got:
                    told: list[str] = []
                    hold = ''
                    if said:
                        self.take_heard({'needs': got.get('needs')}, told)
                        hold = str(got.get('hold') or '').strip()
                    if isinstance(got.get('call'), dict):
                        call = got.get('call')
                    else:
                        call = None
                    self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
                    if not (told or hold):
                        name = str((call or {}).get('tool') or '')
                        shown = list((e for e in _items(got.get('steps')) if isinstance(e, dict)))
                        held_230 = 'order' == stage
                        if held_230:
                            held_230 = any((0 < p['messages'] for p in steps))
                        talk = held_230
                        lines = range(1, len(self.done_log) + 1)
                        proof = set((i for i in lines if not talk or (len(self.done_steps) >= i and self.done_steps[i - 1] > self.cart_at)))
                        unproven = list((e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))))
                        if not (not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_limit > self.msg_used) and (self.left() >= 2)):
                            if not call or name not in self.schemas or name in ending:
                                if not call:
                                    self.gate_ok.add((stage, self.cart_mark() + '#{}'.format(self.version)))
                                break
                            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_limit <= self.msg_used):
                                break
                            faults: list[str] = []
                            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
                            if faults:
                                break
                            if name == self.roles.get('ask'):
                                text = next((v for v in args.values() if isinstance(v, str)), '')
                                batch = self.send([(1, '', text)], 1)
                                if batch:
                                    self.gate_msgs = self.gate_msgs + 1
                                    held_231 = 'Q: {}\nA: {}'.format(text, batch[0][2])
                                    said.append(held_231)
                                    continue
                                else:
                                    break
                            reply = self.act(name, args)
                            if None is reply:
                                break
                            seen = reply.get('observation')
                            self.absorb(seen)
                            held_232 = 'step {}: {} {} -> {} (cart then: {})'.format(self.step, name, _dump(args), self.show(seen)[:300], self.cart_mark())
                            self.done_log.append(held_232)
                            self.done_steps.append(self.step)
                            held_234 = _dump(args)
                            held_233 = '{} {} -> {}'.format(name, held_234, self.show(seen)[:400])
                            said.append(held_233)
                        else:
                            nudged = True
                            self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                            self.done_log.append('(Halo: no line made since the cart last changed at step {} shows '.format(self.cart_at) + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + ' done, and {} message(s) and {} steps are left: give the call that does it)'.format(self.msg_limit - self.msg_used, self.left()))
                            self.done_steps.append(-1)
                            continue
                    else:
                        self.version = self.version + 1
                        held_229 = (False, ('Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
                        return held_229
                else:
                    break
            held_235 = (True, 'Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) if said else '')
            return held_235
        else:
            held_228 = (True, '')
            return held_228

    def spoken(self, observation: Any) -> str:
        if not isinstance(observation, str):
            found: list[str] = []

            def walk(node: Any, depth: int) -> None:
                if 4 < depth:
                    return
                elif isinstance(node, str) and len(node.split()) >= 2:
                    found.append(node)
                elif isinstance(node, dict):
                    for value in node.values():
                        walk(value, depth + 1)
                elif isinstance(node, list):
                    for value in node:
                        walk(value, depth + 1)
            walk(observation, 0)
            if found:
                held_236 = ' '.join(found)[:800]
                return held_236
            held_237 = _dump(observation)[:400]
            return held_237
        else:
            return observation

    def is_record(self, line: str) -> bool:
        parts = list((p for p in re.split('\\s\\|\\s|;\\s', str(line)) if re.match('\\s*[\\w .\\-]{2,40}:\\s', p)))
        held_238 = 2 <= len(parts)
        return held_238

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict):
            if not self.ident(observation):
                for key, value in observation.items():
                    if isinstance(value, str) and len(value.split()) >= 5:
                        lines.append(value)
                    elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                        lines.extend((v for v in value if len(v.split()) >= 5))
                    elif isinstance(value, dict):
                        if not self.ident(value, False, 1):
                            for inner in value.values():
                                if isinstance(inner, str):
                                    if len(inner.split()) >= 6:
                                        lines.append(inner)
        held_239 = list((line for line in lines if line not in self.heard and (not self.is_record(line))))
        return held_239

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        held_240 = self.ask_json(prompt, [LIGHT, SPARE], 2500)
        if not held_240:
            held_240 = {}
        got = held_240
        gone = list((str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))))
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold:
            if 1 == len(lines):
                self.sold_lines.update(lines)
        if not got.get('kind') not in (None, 'change'):
            changed = []
            budget = got.get('budget')
            if isinstance(budget, (int, float)):
                if not isinstance(budget, bool):
                    if 0 < budget:
                        if self.budget != budget:
                            self.budget = budget
                            held_241 = 'budget now {}'.format(budget)
                            changed.append(held_241)
            added: set[str] = set()
            drops = list((e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()))
            adds = list((e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()))
            for entry in drops:
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                target = str(entry['need']).strip()
                live = list((m for m in item.musts + item.told if m not in item.dropped))
                hit = list((m for m in live if target == m))
                if not hit:
                    hit = list((m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])))[:1]
                item.dropped.extend(hit or [target])
                held_242 = 'dropped {}'.format(target)
                changed.append(held_242)
            for entry in adds:
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                need = str(entry['need']).strip()
                if need in item.dropped:
                    item.dropped.remove(need)
                if need not in item.musts + item.told:
                    item.told.append(need)
                added.add(need)
                held_243 = 'added {}'.format(need)
                changed.append(held_243)
            if gone:
                changed.append('unavailable: ' + ', '.join(gone))
            if got.get('price_moved') is True:
                changed.append('a price changed')
            summary = str(got.get('summary') or '').strip()
            if self.brief.get('pick') == 'cheapest':
                if drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed)):
                    self.wider_held = False
                    self.cheaper_marks = set()
            if changed or summary:
                self.version = self.version + 1
                self.recent = []
                self.notes.append(summary or '; '.join(changed))
            held_244 = (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()
            return held_244
        else:
            return ''

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
            if 6 < depth:
                return
            elif isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if not (ident and _fact_block(node) and (ident not in ids)):
                    for value in node.values():
                        walk(value, depth + 1)
                else:
                    ids.append(ident)
                    return
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.needs_now()
        if not (not ids or not needs):
            many = 1 < len(self.items)
            lines = []
            for ident in ids[:12]:
                held_245 = _fact_block(self.full.get(ident) or {})
                if not held_245:
                    held_245 = {}
                block = held_245
                parts = []
                for number, need in needs[:8]:
                    prop = need.split(':', 1)[0].strip()
                    hits = list(('{}={}'.format(k, str(v)[:50]) for k, v in block.items() if _names(_stem(prop), _stem(k))))
                    if many:
                        tag = '(item {}) '.format(number)
                    else:
                        tag = ''
                    if hits:
                        parts.append('{}{} -> '.format(tag, prop) + ', '.join(hits[:3]))
                if parts:
                    lines.append('{}: '.format(ident) + '; '.join(parts))
            if lines:
                held_246 = "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)
                return held_246
            else:
                return ''
        else:
            return ''

    def fact_ids(self, observation: Any) -> set[str]:
        found: set[str] = set()

        def walk(node: Any, depth: int) -> None:
            if 6 < depth:
                return
            elif isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if not (ident and _fact_block(node)):
                    for value in node.values():
                        walk(value, depth + 1)
                else:
                    found.add(ident)
                    return
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return found

    def cart_text(self) -> str:
        if not (None is not self.cart_view and self.cart_at <= self.cart_seen_at):
            held_248 = "lines by Halo's record of the cart calls (listing: units): " + _dump(self.cart_lines)
            return held_248
        else:
            held_247 = _dump(_clip(self.cart_view, 300))[:3000]
            return held_247

    def held_units(self, ident: str) -> int:
        if None is not self.cart_view:
            if self.cart_at <= self.cart_seen_at:
                stack: list[Any] = [self.cart_view]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        if ident in list((v.strip() for v in node.values() if isinstance(v, str))):
                            counts = list((_count(v) for k, v in node.items() if not isinstance(v, str) and _count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))))
                            if 1 == len(counts):
                                held_249 = counts[0]
                                return held_249
                        stack += list((v for v in node.values() if isinstance(v, (dict, list))))
                    elif isinstance(node, list):
                        stack += node
        held_250 = self.cart_lines.get(ident, 1)
        return held_250

    def cart_ids(self) -> list[str]:
        if not (None is not self.cart_view and self.cart_at <= self.cart_seen_at):
            held_252 = list(self.cart_lines)
            return held_252
        else:
            text = _dump(self.cart_view)
            held_251 = list((i for i in self.rows if i in text))
            return held_251

    def record_of(self, ident: str) -> str:
        held_253 = self.full.get(ident)
        if not held_253:
            held_253 = self.rows.get(ident)
        record = held_253
        if record:
            held_254 = _dump(_clip(record, 400))[:3000]
            return held_254
        return '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        held_255 = reversed(self.seen_order)
        order = list(held_255)
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if not (ident in skip or self.sold_out(ident)):
                held_256 = self.full.get(ident)
                if not held_256:
                    held_256 = self.rows.get(ident)
                record = held_256
                lines.append(_dump(_clip(record, 160))[:900])
                if limit <= len(lines):
                    break
            else:
                continue
        held_257 = '\n'.join(lines)
        return held_257

    def fit_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.seen_order:
            held_258 = self.full.get(ident)
            if not held_258:
                held_258 = self.rows.get(ident)
            if not held_258:
                held_258 = {}
            record = held_258
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join(('{} {}'.format(k, v) for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        held_259 = {'marks': marks, 'notes': notes}
        return held_259

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        held_260 = index['marks'].get(ident)
        if not held_260:
            held_260 = {}
        mark = held_260
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        held_261 = list((i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))))
        return held_261

    def fit_faults(self) -> list[str]:
        if self.bound:
            index = self.fit_index()
            cart = self.cart_ids()
            faults = []
            for part, main in self.bound:
                devices = list((i for i in cart if i in self.item_ids(main)))
                parts = list((i for i in cart if i in self.item_ids(part)))
                if not (1 != len(devices) or not parts):
                    codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
                    if codes:
                        for line in parts:
                            fits = self.fit_set(line, index)
                            if fits:
                                if not fits & codes:
                                    good = list((i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)))[:4]
                                    faults.append('{} ({}) works with {}, not with {} ({})'.format(line, self.items[part - 1].name, ', '.join(sorted(fits)[:4]), devices[0], ', '.join(sorted(codes))) + ('; seen parts of that kind that work with it: {}'.format(', '.join(good)) if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
                    else:
                        continue
                else:
                    continue
            return faults
        else:
            held_262 = []
            return held_262

    def fit_plan(self, short: bool=False) -> str:
        if self.bound:
            index = self.fit_index()
            lines = []
            for main in sorted(set((m for _, m in self.bound))):
                parts = list((p for p, m in self.bound if main == m))
                tally: dict[str, dict[int, int]] = {}
                for p in parts:
                    for i in self.item_ids(p):
                        if not self.sold_out(i):
                            for code in self.fit_set(i, index):
                                tally.setdefault(code, {}).setdefault(p, 0)
                                tally[code][p] += 1
                        else:
                            continue
                ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
                devices = dict(((code, list((i for i in self.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or [])))[:2]) for code, _ in ranked))
                text = '; '.join(('{}: '.format(code) + ', '.join(('{} {}'.format(n, self.items[p - 1].name) for p, n in sorted(per.items()))) + (' (device seen: {})'.format(', '.join(devices[code])) if devices[code] else ' (device not seen yet)') for code, per in ranked))
                if text:
                    held_263 = "codes the {} candidates' parts name -> {}".format(self.items[main - 1].name, text)
                    lines.append(held_263)
            if lines:
                if short:
                    held_264 = ' | '.join(lines)
                    return held_264
                held_265 = "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)
                return held_265
            else:
                return ''
        else:
            return ''

    def missing_items(self) -> list[int]:
        cat_key = self.category_key()
        ids = list((i for i in self.cart_ids() if (self.held_units(i) or 0) > 0))
        if not (not cat_key or len(self.items) <= len(ids)):
            labels = list((str(r.get(cat_key) or '') for r in self.rows.values()))
            held = list((str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids))
            held_267 = list((n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))))
            return held_267
        else:
            held_266 = []
            return held_266

    def states_needs(self, ident: str, number: int) -> bool:
        held_268 = _fact_block(self.full.get(ident) or {})
        if not held_268:
            held_268 = {}
        block = held_268
        for owner, need in self.needs_now():
            if not number != owner:
                prop, _, value = need.partition(':')
                split = lambda text: set(re.sub('[,/;|]', ' ', _words(text)).split())
                words = split(value)
                if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                    return False
                elif not any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                    return False
            else:
                continue
        return True

    def fill_short_cart(self) -> bool:
        held_269 = self.roles.get('cart')
        if not held_269:
            held_269 = ''
        cart = held_269
        bound = set((n for pair in self.bound for n in pair))
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = list((i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and (self.price_of(i) is not None) and self.states_needs(i, number)))
            fits.sort(key=lambda i: self.price_of(i) or 0.0)
            if fits and number not in bound:
                pick = fits[0]
            else:
                pick = None
            if pick:
                if isinstance(self.budget, (int, float)):
                    held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
                    cost = (self.price_of(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                    if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                        pick = None
            if pick:
                calls = self.cart_calls({**dict(((i, self.cart_lines[i]) for i in self.cart_ids() if i in self.cart_lines)), pick: item.qty}, [])
            else:
                calls = None
            if not (not calls or len(calls) + 1 > self.left()):
                for args in calls:
                    faults: list[str] = []
                    held_270 = self.schemas.get(cart)
                    args = _shape(args, held_270, 'args', faults)
                    if not (faults or self.act(cart, args) is None):
                        self.note_cart(cart, args)
                    else:
                        return False
                self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
            else:
                return False
        return True

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if not (2 > len(mine) or mine <= _roots(label) or (not cat_key) or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf))):
            held_271 = next((seen for seen in sorted(set((str(r.get(cat_key) or '') for r in self.rows.values()))) if seen and mine <= _roots(seen)), '')
            return held_271
        else:
            return ''

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        ids = list((i for i in self.cart_ids() if (self.held_units(i) or 0) > 0))
        if not (not ids or not cat_key):
            label = dict(((i, str((self.rows.get(i) or {}).get(cat_key) or '')) for i in ids))
            known = set((n for n, item in enumerate(self.items, 1) if any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))))
            owner = dict(((i, list((n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])))) for i in ids))
            faults: list[str] = []
            for n, item in enumerate(self.items, 1):
                if not n not in known:
                    lines = list((i for i in ids if n in owner[i]))
                    if not lines:
                        held_273 = 'item {} ({}) has no line in the cart'.format(n, item.name)
                        faults.append(held_273)
                    elif len(list((i for i in lines if owner[i] == [n]))) > 1:
                        held_274 = 'item {} ({}) has {} lines ({}); keep one'.format(n, item.name, len(lines), ', '.join(lines))
                        faults.append(held_274)
                    for i in lines:
                        have = self.held_units(i)
                        if owner[i] == [n]:
                            if item.qty != have:
                                held_275 = '{} is held at {} unit(s); item {} ({}) wants {}'.format(i, have, n, item.name, item.qty)
                                faults.append(held_275)
                        fuller = self.fuller_label(n, label[i])
                        if fuller:
                            held_276 = '{} is from {}, but the store has the label {}, which names item {} ({}) in full: take the line from there'.format(i, label[i], fuller, n, item.name)
                            faults.append(held_276)
                else:
                    continue
            if len(self.items) == len(known):
                for i in ids:
                    if not owner[i]:
                        held_277 = '{} ({}) is not a kind the shopper asked for: take it out'.format(i, label[i] or 'unknown kind')
                        faults.append(held_277)
            for i in ids:
                if self.sold_out(i):
                    held_278 = '{} is sold out'.format(i)
                    faults.append(held_278)
            prices = dict(((i, self.price_of(i)) for i in ids))
            if isinstance(self.budget, (int, float)):
                if all((None is not p for p in prices.values())):
                    if not self.brief.get('budget_scope') == 'each':
                        total = sum((prices[i] * self.held_units(i) for i in ids))
                        if self.budget + 1e-09 < total <= 10 * self.budget:
                            held_279 = 'the total is {:g}, over the budget of {:g}'.format(total, self.budget)
                            faults.append(held_279)
                    else:
                        over = list((i for i in ids if self.budget < prices[i] <= 10 * self.budget))
                        faults += list(('{} costs {:g}, over the {:g} limit for each item'.format(i, prices[i], self.budget) for i in over))
            return faults
        else:
            held_272 = []
            return held_272

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if cat_key:
            out = []
            for ident in self.cart_ids():
                held_281 = self.held_units(ident)
                if not held_281:
                    held_281 = 0
                have = held_281
                owners = list((n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))))
                mine = list((i for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and 1 == len(owners) and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))))
                if 0 < have:
                    if 1 == len(owners):
                        if 1 == len(mine):
                            if have != self.items[owners[0] - 1].qty:
                                if not self.sold_out(ident):
                                    out.append((ident, have, self.items[owners[0] - 1].qty))
            return out
        else:
            held_280 = []
            return held_280

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        held_282 = self.roles.get('cart')
        if not held_282:
            held_282 = ''
        name = held_282
        held_283 = self.schemas.get(name)
        if not held_283:
            held_283 = {}
        schema = held_283
        held_284 = schema.get('properties')
        if not held_284:
            held_284 = {}
        props = held_284
        texts = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)))
        held_285 = next((k for t, k in self.id_args if name == t and k in texts), None)
        if not held_285:
            held_285 = next((k for k in schema.get('required') or [] if k in texts), None)
        if not held_285:
            if 1 == len(texts):
                held_285 = texts[0]
            else:
                held_285 = None
        target = held_285
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (2 <= len(v['enum']))), None)
        if not (not name or not target or (not count)):
            calls: list[dict[str, Any]] = []
            for ident, have, want in wrong:
                if self.cart_sets(name):
                    calls.append({target: ident, count: want})
                elif not mode:
                    held_287 = []
                    return held_287
                else:
                    options = list((str(o) for o in props[mode]['enum']))
                    off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                    on = next((o for o in options if off != o), options[0])
                    calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            if not len(calls) + 3 > self.left():
                done = []
                for args in calls:
                    faults: list[str] = []
                    args = _shape(args, schema, 'args', faults)
                    if not faults:
                        reply = self.act(name, args)
                        if not (None is reply or self.finished):
                            self.note_cart(name, args)
                            held_289 = reply.get('observation')
                            self.cart_said = _dump(held_289)
                            held_290 = reply.get('observation')
                            self.observe(name, held_290)
                            done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:300])
                        else:
                            break
                    else:
                        break
                self.trace.append({'kind': 'count_fix', 'calls': done})
                return done
            else:
                held_288 = []
                return held_288
        else:
            held_286 = []
            return held_286

    def check_key(self, ids: list[str]) -> str:
        held_291 = '|'.join(sorted(ids)) + '#{}'.format(self.version)
        return held_291

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = None is not ids
        if lead:
            ids = ids
        else:
            ids = self.cart_ids()
        key = self.check_key(ids)
        if not (key in self.checked or not self.judges(limit) or (not ids)):
            self.checked.add(key)
            prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
            if not lead:
                prompt += '\n\nCART:\n' + self.cart_text()
            else:
                quantity = '; '.join(('item {}: quantity {}'.format(n, it.qty) for n, it in enumerate(self.items, 1)))
                prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
            prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(('{}: '.format(i) + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
            if self.brief.get('pick') in ('cheapest', 'most', 'least'):
                prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
            judges = self.judges(limit)
            got = None
            if 1 < len(judges):
                if not lead:
                    if not self.audited:
                        if self.brief.get('pick') not in ('cheapest', 'most', 'least'):
                            got = self.ask_json(prompt, [LIGHT], 3000)
                            if got:
                                if got.get('ok') is False and HEAVY in self.judges(limit):
                                    judges = [HEAVY]
                                else:
                                    judges = []
            if judges:
                if not self.no_credit:
                    held_292 = self.ask_json(prompt, judges, 3000)
                    if not held_292:
                        held_292 = got
                    got = held_292
            self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
            self.verdicts[key] = bool(got) and got.get('ok') is not False
            if not (not got or got.get('ok') is not False):
                self.leads = list((i for i in self.leads if i not in ids))
                problems = list((str(p) for p in _items(got.get('problems'))))[:5]
                better = list((str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))))[:4]
                text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
                if better:
                    text += '. Listings that look better: ' + ', '.join(better)
                held_293 = text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'
                return held_293
            else:
                return ''
        else:
            return ''

    def trail(self) -> tuple[str, bool]:
        live = set((need for _, need in self.needs_now()))
        pending = list((need for _, need in self.refs if need in live and need not in self.ref_values))
        held_294 = self.roles.get('offer')
        if not held_294:
            held_294 = self.roles.get('open')
        tool = held_294
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (SPEND_HARD > self.cost):
            self.trails = self.trails + 1
            records = []
            for ident in reversed(self.seen_order):
                if ident in self.opened:
                    record = self.full.get(ident) or self.rows.get(ident)
                else:
                    record = None
                if record:
                    if 45 > len(records):
                        records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if 60 <= len(records):
                    break
                elif ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            held_295 = self.ask_json(prompt, [LIGHT] + ([HEAVY] if SPEND_HARD >= self.cost + HEAVY_ROOM else []), 4500)
            if not held_295:
                held_295 = {}
            got = held_295
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if isinstance(chain, dict):
                    held_296 = next((p for p in pending if p == str(chain.get('need') or '').strip()), None)
                    if not held_296:
                        if 1 == len(pending):
                            held_296 = pending[0]
                        else:
                            held_296 = None
                    need = held_296
                    if need:
                        links = list((l for l in _items(chain.get('links')) if isinstance(l, dict)))
                        ids = list((str(l.get('id') or '').strip() for l in links))
                        if not (chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip()):
                            if links:
                                if all((i in self.opened for i in ids)):
                                    if not str(links[-1].get('find') or '').strip():
                                        self.ref_values[need] = str(chain.get('value') or '').strip() or 'as stated on {}'.format(ids[-1])
                                        self.version = self.version + 1
                                        continue
                            closed = list((i for i in ids if i in self.rows and i not in self.opened))
                            if not closed:
                                gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                                held_297 = str(gap.get('about') or '')
                                if not held_297:
                                    held_297 = 'the listing it points at'
                                about = held_297
                                words = str(gap.get('find') or '').strip()
                                lost.append('{} -> find {}'.format(need, about) + (' (search: {})'.format(words) if words else ''))
                                gaps.append((about, words))
                            else:
                                reach += list((i for i in closed if i not in reach))
                        else:
                            self.ref_values[need] = str(chain['value']).strip()
                            self.version = self.version + 1
                            continue
                    else:
                        continue
                else:
                    continue
            pending = list((p for p in pending if p not in self.ref_values))
            if not reach:
                if gaps:
                    if tool:
                        if target:
                            if self.left() > 12:
                                if 3 > self.link_searches:
                                    reach = self.find_link(*gaps[0])
                                    if self.trace:
                                        if self.trace[-1].get('kind') == 'link_known':
                                            held_298 = '{}: an opened page of the shop this chain names that shows the values its description gives; read the value from it'.format(self.trace[-1]['id'])
                                            seen.append(held_298)
                                            lost = lost[1:]
            if not (not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished):
                for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                    faults: list[str] = []
                    args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.act(tool, args)
                        if not (None is reply or self.finished):
                            held_299 = reply.get('observation')
                            self.observe(tool, held_299)
                            seen.append('{}: '.format(ident) + self.show(reply.get('observation'))[:1500])
                        else:
                            break
                    else:
                        break
            else:
                break
        text = ''
        if seen:
            text += '[Halo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join(('{} -> {}'.format(k, v) for k, v in self.ref_values.items())) + '\n'
        held_300 = bool(lost)
        if held_300:
            held_300 = self.trail_blocks < (3 if self.left() > 15 else 2)
        if held_300:
            held_300 = self.left() > 5
        hold = held_300
        if hold:
            self.trail_blocks = self.trail_blocks + 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        held_301 = (text, hold)
        return held_301

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            held_302 = self.rows.get(ident)
            if not held_302:
                held_302 = {}
            row = held_302
            names = list((str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)))
            if ident not in self.opened:
                if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                    out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            held_303 = self.rows.get(ident)
            if not held_303:
                held_303 = {}
            row = held_303
            names = list((str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)))
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = set((w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())))
        values = set((w for w in re.findall('[a-z0-9]+', (words or about).lower()) if 3 <= len(w) or w.isdigit())) - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for ident in self.opened_shop(about):
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if not isinstance(node, (dict, list)):
                    held_304 = str(node)
                    leaves.append(held_304)
                else:
                    stack += list(node.values()) if isinstance(node, dict) else node
            text = ' '.join(leaves).lower()
            if values:
                if all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                    self.trace.append({'kind': 'link_known', 'id': ident})
                    held_305 = []
                    return held_305
        plan = self.search_template()
        held_306 = words.lower().split()
        memo = ' '.join(held_306)
        if memo and memo in self.link_seen:
            self.link_searches = self.link_searches + 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan:
            if words:
                tool, base, field = plan
                faults: list[str] = []
                args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
                if not faults:
                    self.link_searches = self.link_searches + 1
                    reply = self.act(tool, args)
                    if None is not reply:
                        observation = reply.get('observation')
                        self.observe(tool, observation)
                        self.link_seen[memo] = list((i for i, _ in self.records_in(observation)))
                        known = self.shop_rows(about, self.link_seen[memo])
                        self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = set((w for w in re.findall('[a-z0-9]{3,}', (words or about).lower()))) - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            held_307 = list((i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            return held_307
        held_308 = (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]
        return held_308

    def offer_tool(self) -> str | None:
        held_309 = self.roles.get('offer') or self.roles.get('open')
        return held_309

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not (not tool or not target):
            since = max(self.step - 2, self.refused_at)
            stale = list((i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone))
            if not (not stale or self.left() < len(stale) + 3):
                lines = []
                for ident in stale:
                    faults: list[str] = []
                    args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        before = dict(self.rows.get(ident) or {})
                        reply = self.act(tool, args)
                        if not (None is reply or self.finished):
                            observation = reply.get('observation')
                            self.offer_seen[ident] = self.step
                            self.observe(tool, observation)
                            held_310 = self.rows.get(ident)
                            if not held_310:
                                held_310 = {}
                            after = held_310
                            moved = list((k for k, v in after.items() if k in before and v != before[k] and (not isinstance(v, (dict, list)))))
                            if moved:
                                held_312 = ', '.join(moved[:4])
                                held_311 = '{} changed ({}): {}'.format(ident, held_312, _dump(_clip(observation, 300))[:900])
                                lines.append(held_311)
                        else:
                            break
                    else:
                        break
                self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
                if lines:
                    self.version = self.version + 1
                held_313 = '\n'.join(lines)
                return held_313
            else:
                return ''
        else:
            return ''

    def sold_out(self, ident: str) -> bool:
        held_314 = ident in self.unavailable or ident in self.gone
        return held_314

    def swap_hint(self) -> str:
        lost = list((i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)))
        if lost:
            for ident in lost:
                self.gone[ident] = self.step
            self.leads = list((i for i in self.leads if i not in lost))
            self.version = self.version + 1
            picks = self.replacements(lost)
            held_315 = ', '.join(lost)
            text = '[Halo] Sold out now: {} (the store shows it cannot be bought). It will not come back, so do not open it again.'.format(held_315)
            if picks and self.left() <= 6:
                held_316 = text + ' Few steps are left: in this turn add {} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {}.'.format(picks[0], ', '.join(picks[1:]) or 'none')
                return held_316
            elif not (picks and self.brief.get('pick') == 'cheapest'):
                if picks:
                    held_318 = text + ' Listings already read that appear to fit that item instead: {}. In this turn take the sold-out line out of the cart and add {} at the requested quantity, then check out.'.format(', '.join(picks), picks[0])
                    return held_318
                held_319 = text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'
                return held_319
            else:
                held_317 = text + " Listings already read that appear to fit that item instead, cheapest first: {}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing.".format(', '.join(picks), picks[0])
                return held_317
        else:
            return ''

    def find_sold(self) -> str:
        ids = list((i for i in self.cart_ids() if i not in self.gone))
        if any((self.sold_out(i) for i in ids)):
            return ''
        elif not 1 == len(ids):
            if ids:
                held_320 = self.reread_offers()
                return held_320
            return ''
        else:
            self.unavailable.add(ids[0])
            return ''

    def replacements(self, lost: list[str]) -> list[str]:
        if self.judges():
            records = []
            for ident in reversed(self.seen_order):
                if not (ident in lost or ident in self.gone or self.sold_out(ident)):
                    records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
                    if 25 <= len(records):
                        break
                else:
                    continue
            if records:
                prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
                prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(('{}: {}'.format(i, self.record_of(i)) for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
                prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
                got = self.ask_json(prompt, self.judges()[::-1], 3500)
                fits: list[tuple[float, str]] = []
                for entry in _items((got or {}).get('fits')):
                    ident = str(entry.get('id') if isinstance(entry, dict) else entry)
                    if isinstance(entry, dict):
                        price = entry.get('price')
                    else:
                        price = None
                    if ident in self.rows:
                        if ident not in lost:
                            if ident not in self.gone:
                                if ident not in self.cart_ids():
                                    if ident not in list((i for _, i in fits)):
                                        fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
                if self.brief.get('pick') == 'cheapest':
                    ranked = list((i for _, i in sorted(fits, key=lambda f: f[0])))
                else:
                    ranked = list((i for _, i in fits))
                self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
                held_323 = ranked[:4]
                return held_323
            else:
                held_322 = []
                return held_322
        else:
            held_321 = []
            return held_321

    def audit(self) -> str:
        if not (self.audited or not self.judges()):
            self.audited = True
            got = self.audit_call()
            named = self.audit_fits(got)
            if not named:
                rows = list((str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))))
                if rows:
                    if self.open_rows(rows):
                        got = self.audit_call()
                        named = self.audit_fits(got)
            if named:
                self.lead_items = named
                self.leads = list((i for ids in named.values() for i in ids))
                held_324 = 'A careful review thinks these listings may fit: ' + '; '.join(('item {}: {}'.format(k, ', '.join(v)) for k, v in named.items())) + (' ({})'.format(got.get('reason')) if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'
                return held_324
            else:
                return ''
        else:
            return ''

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if not self.sold_out(ident):
                if ident in self.full:
                    record = self.full.get(ident)
                else:
                    record = None
                records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
                if 30 <= len(records):
                    break
            else:
                continue
        if records:
            prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
            prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
            judges = self.judges()
            got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
            if HEAVY in judges:
                if not self.no_credit:
                    if not self.audit_fits(got):
                        held_325 = self.ask_json(prompt, [HEAVY], 4000)
                        if not held_325:
                            held_325 = got
                        got = held_325
            self.trace.append({'kind': 'audit', 'got': got})
            return got
        else:
            return None

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if isinstance(fits, dict):
            named = dict(((str(k), list((str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))))[:4]) for k, v in fits.items()))
            held_327 = dict(((k, v) for k, v in named.items() if v))
            return held_327
        else:
            held_326 = {}
            return held_326

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif not (tool and target):
            return False
        else:
            plan = list(((tool, {target: i}) for i in rows[:2]))
        if not self.left() < len(plan) + 5:
            before = len(self.full)
            for name, args in plan:
                faults: list[str] = []
                held_328 = self.schemas.get(name)
                args = _shape(args, held_328, 'args', faults)
                if not faults:
                    reply = self.act(name, args)
                    if not (None is reply or self.finished):
                        held_329 = reply.get('observation')
                        self.observe(name, held_329)
                    else:
                        break
                else:
                    return False
            self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
            held_330 = before < len(self.full)
            return held_330
        else:
            return False

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        held_331 = self.roles.get('cart')
        if not held_331:
            held_331 = ''
        name = held_331
        held_332 = self.schemas.get(name)
        if not held_332:
            held_332 = {}
        schema = held_332
        held_333 = schema.get('properties')
        if not held_333:
            held_333 = {}
        props = held_333
        for key, spec in props.items():
            if isinstance(spec, dict) and 'array' in _kinds(spec):
                shape = spec.get('items')
            else:
                shape = None
            if isinstance(shape, dict):
                if isinstance(shape.get('properties'), dict):
                    inner = shape['properties']
                    id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                    count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                    if id_key:
                        held_334 = [{key: list(({id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()))}]
                        return held_334
                    else:
                        return None
        texts = list((k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)))
        held_335 = next((k for t, k in self.id_args if name == t and k in texts), None)
        if not held_335:
            held_335 = next((k for k in schema.get('required') or [] if k in texts), None)
        if not held_335:
            if 1 == len(texts):
                held_335 = texts[0]
            else:
                held_335 = None
        target = held_335
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (2 <= len(s['enum']))), None)
        if target:
            calls: list[dict[str, Any]] = []
            for ident in drop:
                have = self.cart_lines.get(ident, 1)
                if mode:
                    calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
                elif count and self.cart_sets(name):
                    calls.append({target: ident, count: 0})
                elif not (count and isinstance(props[count].get('minimum'), int) and (-have >= props[count]['minimum'])):
                    return None
                else:
                    calls.append({target: ident, count: -have})
            for ident, want in lines.items():
                if ident in self.cart_ids():
                    have = self.cart_lines.get(ident, 0)
                else:
                    have = 0
                if want == have:
                    continue
                elif not (have and (not self.cart_sets(name))):
                    calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
                else:
                    return None
            return calls
        else:
            return None

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if not (self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy)):
            self.completed = True
            cat_key = self.category_key()

            def kind(ident: str) -> str:
                if cat_key:
                    held_336 = str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower()
                    return held_336
                return ident
            plan: dict[str, int] = {}
            for number, ids in self.lead_items.items():
                held_337 = str(number)
                found = re.search('\\d+', held_337)
                if found:
                    index = int(found.group())
                else:
                    index = 1 if 1 == len(self.items) else None
                pick = next((i for i in ids if not self.sold_out(i)), None)
                if pick:
                    if index:
                        if 1 <= index <= len(self.items):
                            if kind(pick) not in set((kind(p) for p in plan)):
                                plan[pick] = self.items[index - 1].qty
            if not self.confirmed not in plan:
                now = self.cart_ids()
                keep = dict(((i, self.cart_lines.get(i, 1)) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in set((kind(p) for p in plan)))))
                drop = list((i for i in now if i not in plan and i not in keep))
                if not len(plan) + len(set((kind(i) for i in keep))) < len(self.items):
                    lines = {**keep, **plan}
                    calls = self.cart_calls(lines, drop)
                    if not calls or len(calls) + 1 > self.left():
                        return ''
                    held_338 = list(lines)
                    self.fill_facts(held_338, len(calls) + 3)
                    held_339 = list(lines)
                    self.show_links(held_339, len(calls) + 3)
                    held_340 = list(lines)
                    self.check_cart(held_340, SPEND_HARD)
                    if not self.verdicts.get(self.check_key(list(lines))):
                        return ''
                    done = []
                    for args in calls:
                        faults: list[str] = []
                        held_341 = self.schemas.get(cart)
                        args = _shape(args, held_341, 'args', faults)
                        if not faults:
                            reply = self.act(cart, args)
                            if not (None is reply or self.finished):
                                self.note_cart(cart, args)
                                held_342 = reply.get('observation')
                                self.observe(cart, held_342)
                                held_343 = _dump(args)
                                done.append(held_343)
                            else:
                                return ''
                        else:
                            return ''
                    faults = []
                    held_344 = dict(self.brief.get('buy_args') or {})
                    held_345 = self.schemas.get(buy)
                    args = _shape(held_344, held_345, 'args', faults)
                    if faults:
                        return ''
                    replies = []
                    for attempt in range(2):
                        if attempt:
                            self.reread_offers()
                            held_346 = list(lines)
                            self.check_cart(held_346, SPEND_HARD)
                            if not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)):
                                break
                        go, said = self.gate('order')
                        if said:
                            replies.append(said)
                        if go:
                            reply = self.act(buy, args)
                            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
                            if not None is reply:
                                self.buys = self.buys + 1
                                if not reply.get('finished'):
                                    self.refused_at = self.step
                                held_347 = reply.get('observation')
                                self.observe(buy, held_347)
                                held_349 = reply.get('observation')
                                held_348 = self.show(held_349)
                                replies.append(held_348)
                                if self.finished or reply.get('finished'):
                                    break
                            else:
                                break
                        else:
                            break
                    if not replies:
                        return ''
                    held_350 = 'Not run (Halo acted instead): the independent check confirmed ' + self.confirmed + ', so Halo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)
                    return held_350
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if not name != self.roles.get('buy'):
            if isinstance(self.brief.get('buy_args'), dict):
                fixed = self.brief.get('buy_args')
            else:
                fixed = {}
            held_351 = all((args.get(k) == v for k, v in fixed.items()))
            return held_351
        else:
            return False

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if not name != self.roles.get('refuse'):
            if isinstance(self.brief.get('refuse_args'), dict):
                fixed = self.brief.get('refuse_args')
            else:
                fixed = {}
            held_352 = all((args.get(k) == v for k, v in fixed.items()))
            return held_352
        else:
            return False

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif len(self.items) + 5 >= left:
            if 0 == self.buys:
                keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = list((t for t in self.tools if ask != t['function']['name']))
        if None is not keep:
            if ends:
                narrowed = list((t for t in tools if t['function']['name'] in keep))
                held_353 = narrowed
                if not held_353:
                    held_353 = tools
                tools = held_353
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        held_354 = self.schemas.get(name)
        if not held_354:
            held_354 = {}
        schema = held_354
        held_355 = schema.get('properties')
        if not held_355:
            held_355 = {}
        props = held_355
        out = dict(args)
        for key, spec in props.items():
            if not (not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec):
                value = out.get(key)
                if not (None is value and key not in (schema.get('required') or [])):
                    text = str(value or '')
                    held_356 = any((ident in text for ident in self.rows))
                    if not held_356:
                        held_356 = self.brief.get('quiet_decline') is not False
                    named = held_356
                    if named or not text:
                        out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
                else:
                    continue
            else:
                continue
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not (name not in set((self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare'))) - {None} or not self.id_key):
            held_358 = (self.schemas.get(name) or {}).get('properties')
            if not held_358:
                held_358 = {}
            props = held_358
            free = list((k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)))
            if not (1 != len(free) or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args)):

                def fix(value: Any) -> tuple[Any, str]:
                    if isinstance(value, str) and value.strip() in self.rows:
                        held_360 = (value.strip(), '')
                        return held_360
                    elif not (not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8):
                        low = value.strip().lower()
                        hits = list((i for i, row in self.rows.items() if any((self.id_key != k and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))))
                        if not 1 == len(hits):
                            if hits:
                                held_363 = (value, 'several listings carry the text {!r} ({}); pass one listing id'.format(value, ', '.join(hits[:6])))
                                return held_363
                            held_364 = (value, '{!r} is not a listing id; pass the id from the result rows (their {} field)'.format(value, self.id_key))
                            return held_364
                        else:
                            held_362 = (hits[0], '')
                            return held_362
                    else:
                        held_361 = (value, '')
                        return held_361
                out = dict(args)
                for key in free:
                    if not key not in args:
                        if isinstance(args[key], list):
                            fixed = list((fix(v) for v in args[key]))
                        else:
                            fixed = [fix(args[key])]
                        problem = next((p for _, p in fixed if p), '')
                        if not problem:
                            out[key] = list((v for v, _ in fixed)) if isinstance(args[key], list) else fixed[0][0]
                        else:
                            held_365 = (args, problem)
                            return held_365
                    else:
                        continue
                held_366 = (out, '')
                return held_366
            else:
                held_359 = (args, '')
                return held_359
        else:
            held_357 = (args, '')
            return held_357

    def cart_sets(self, name: str) -> bool:
        held_367 = self.schemas.get(name, {}).get('properties')
        if not held_367:
            held_367 = {}
        props = held_367
        if not any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (2 <= len(s['enum'])) for s in props.values())):
            held_368 = any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))
            return held_368
        else:
            return False

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        held_369 = self.schemas.get(name, {}).get('properties')
        if not held_369:
            held_369 = {}
        spec = held_369
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        if count and removal and ident and (count not in args):
            held_370 = {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))}
            return held_370
        return args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        held_371 = self.schemas.get(name, {}).get('properties')
        if not held_371:
            held_371 = {}
        spec = held_371
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list):
                if 2 <= len(options):
                    if value in options[1:]:
                        removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                held_372 = (spec.get(key) or {}).get('items')
                if not held_372:
                    held_372 = {}
                shape = held_372
                if not value:
                    if isinstance(shape, dict):
                        if shape.get('type') == 'object' or 'properties' in shape:
                            whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str):
                        if line in self.rows:
                            lines.append((line, loose_count))
            elif isinstance(value, str):
                if value in self.rows:
                    lines.append((value, loose_count))
        if not whole:
            sets = self.cart_sets(name)
            for ident, count in lines:
                have = self.cart_lines.get(ident, 0)
                if removal:
                    if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))):
                        have = have - (count or 1)
                    else:
                        have = have - count if count else 0
                elif not sets:
                    have = have + (count if None is not count else 1)
                elif None is not count:
                    have = count
                else:
                    have = 1
                if not 0 < have:
                    self.cart_lines.pop(ident, None)
                else:
                    self.cart_lines[ident] = have
        else:
            self.cart_lines = dict(((i, c) for i, c in lines if c and 0 < c))
            return

    def show(self, observation: Any) -> str:
        held_373 = _clip(observation, SHOW_CHARS)
        text = _dump(held_373)
        if OBS_LIMIT < len(text):
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        held_374 = self.schemas.get(self.roles.get('compare') or '')
        if not held_374:
            held_374 = {}
        schema = held_374
        held_375 = schema.get('properties')
        if not held_375:
            held_375 = {}
        props = held_375
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not (not target or any((target != k for k in schema.get('required') or []))):
            return target
        else:
            return None

    def compare_room(self) -> int:
        held_376 = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '')
        if not held_376:
            held_376 = {}
        spec = held_376
        if isinstance(spec.get('maxItems'), int):
            held_377 = spec['maxItems']
            return held_377
        elif not self.compare_bad:
            held_379 = max(4, self.compare_ok)
            return held_379
        else:
            held_378 = max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
            return held_378

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (joiner == opener):
            return calls
        elif not self.compare_bare():
            picks: list[tuple[int, str]] = []
            for index, call in enumerate(calls):
                held_380 = call.get('function')
                if not held_380:
                    held_380 = {}
                fn = held_380
                if isinstance(fn.get('arguments'), str):
                    args = _json_obj(fn.get('arguments'))
                else:
                    args = None
                ident = str((args or {}).get(target) or '').strip()
                if fn.get('name') == opener:
                    if args:
                        if set(args) == {target}:
                            if ident in self.rows:
                                if ident not in self.gone:
                                    picks.append((index, ident))
            picks = picks[:self.compare_room()]
            if not 2 > len(picks):
                first = picks[0][0]
                out = []
                for index, call in enumerate(calls):
                    if first == index:
                        out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: list((i for _, i in picks))})}})
                    elif not index in set((i for i, _ in picks)):
                        out.append(call)
                    else:
                        out.append({'id': call['id'], 'joined': True})
                self.trace.append({'kind': 'joined', 'ids': list((i for _, i in picks))})
                return out
            else:
                return calls
        else:
            return calls

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = set((t['function']['name'] for t in self.offered()))
            cid = call['id']
            if not call.get('joined'):
                held_381 = call.get('function')
                if not held_381:
                    held_381 = {}
                fn = held_381
                name = fn.get('name')
                raw = fn.get('arguments')
                if self.finished:
                    out.append((cid, 'Not run: the episode is over.'))
                    continue
                elif not name == self.roles.get('ask'):
                    if name not in self.schemas:
                        out.append((cid, 'Not run (no step used): unknown tool. Tools: {}.'.format(', '.join(sorted(allowed)))))
                        continue
                    if name not in allowed:
                        out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                        continue
                    if not isinstance(raw, dict):
                        if (raw or '').strip():
                            args = _json_obj(raw)
                        else:
                            args = {}
                        if None is args:
                            out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                            continue
                    else:
                        args = raw
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
                    if name == self.roles.get('compare'):
                        many = self.compare_target()
                    else:
                        many = None
                    if many:
                        if isinstance(args.get(many), list):
                            if len(args[many]) > self.compare_room():
                                if self.compare_bad:
                                    room = self.compare_room()
                                    held_383 = map(str, args[many][room:])
                                    held_382 = ', '.join(held_383)
                                    trimmed = '\n[Halo] Sent the first {} listings only: this store shows at most {} side by side. Not shown yet: {}.'.format(room, room, held_382)
                                    args = {**args, many: args[many][:room]}
                    target = self.single_target(name)
                    if target:
                        if name in (self.roles.get('open'), self.roles.get('offer')):
                            if str(args.get(target)) in self.gone:
                                out.append((cid, 'Not run (no step used): {} sold out at step {} and will not come back. Replace its cart line with another fitting listing.'.format(args[target], self.gone[str(args[target])])))
                                continue
                    if name == self.roles.get('cart'):
                        if any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                            if self.cart_at <= self.cart_seen_at:
                                view = _dump(self.cart_view)
                            else:
                                view = ''
                            absent = list((v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)))
                            if absent:
                                out.append((cid, 'Not run (no step used): {} is not in the cart, so there is nothing to take off. Cart now: '.format(absent[0]) + self.cart_text()))
                                continue
                    held_384 = self.is_buy(name, args)
                    if not held_384:
                        held_384 = self.is_refuse(name, args)
                    ending = held_384
                    if ending:
                        if 1 < len(calls):
                            out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                            continue
                    if not ending:
                        if self.left() <= self.tail():
                            if self.roles.get('buy') or self.roles.get('refuse'):
                                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                                continue
                    mark = name + _dump(args)
                    if not ending:
                        if name != self.roles.get('cart'):
                            if mark in self.recent:
                                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                                continue
                    readers = set((self.roles.get(r) for r in ('search', 'open', 'offer', 'compare'))) - {None}
                    cached = self.read_cache.get(mark)
                    if name in readers:
                        if cached:
                            if self.version == cached[0]:
                                out.append((cid, 'Not run (no step used): the same call at step {} returned this, and nothing has changed since:\n'.format(cached[1]) + cached[2][:3000]))
                                continue
                    if target:
                        if name in readers:
                            if name != self.roles.get('compare'):
                                if self.shown_already(name, str(args.get(target))):
                                    out.append((cid, 'Not run (no step used): {} has shown everything this view gives since the last change. Its record: '.format(args[target]) + self.record_of(str(args[target]))))
                                    continue
                    if many:
                        if isinstance(args.get(many), list):
                            if args[many]:
                                keep = list((i for i in args[many] if not self.shown_already(name, str(i))))
                                if not keep:
                                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(('{}: {}'.format(i, self.record_of(str(i))) for i in args[many]))[:4000]))
                                    continue
                                elif len(args[many]) > len(keep):
                                    trimmed += '\n[Halo] Left out {}: shown in full already, nothing changed since.'.format(', '.join((str(i) for i in args[many] if i not in keep)))
                                    args = {**args, many: keep}
                                    mark = name + _dump(args)
                    if ending:
                        if self.left() > 2:
                            if self.is_buy(name, args):
                                if not (self.refused_change and self.refused_change == self.cart_mark() and (self.left() > 3)):
                                    wider = self.wider_hold()
                                    if not wider:
                                        reread = self.reread_offers()
                                        if self.note_sold:
                                            self.find_sold()
                                        if self.cart_touched:
                                            if self.cart_at > self.cart_seen_at:
                                                if self.roles.get('cart_view'):
                                                    if self.left() > 3:
                                                        faults = []
                                                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                                                        if not faults:
                                                            view = self.act(self.roles['cart_view'], view_args)
                                                        else:
                                                            view = None
                                                        if None is not view:
                                                            self.cart_view = view.get('observation')
                                                            self.cart_seen_at = self.step
                                                            change = self.observe(self.roles['cart_view'], self.cart_view)
                                                            if change:
                                                                self.queued = []
                                                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                                                continue
                                        swap = self.swap_hint()
                                        if 2 > self.held:
                                            held = list((i for i in self.cart_ids() if i in self.gone))
                                        else:
                                            held = []
                                        if not (swap or held):
                                            chain, hold = self.trail()
                                            if hold:
                                                out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                                                continue
                                            if 2 > self.fit_holds and self.left() > self.tail() + 3:
                                                misfit = self.fit_faults()
                                            else:
                                                misfit = []
                                            if misfit:
                                                self.fit_holds = self.fit_holds + 1
                                                self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                                                out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                                                continue
                                            if 2 > self.line_holds and self.left() > self.tail() + 2:
                                                faults = self.line_faults()
                                            else:
                                                faults = []
                                            fixed = []
                                            if faults:
                                                if all((' is held at ' in f for f in faults)):
                                                    held_385 = self.count_fixes()
                                                    fixed = self.fix_counts(held_385)
                                                    if fixed:
                                                        faults = self.line_faults()
                                                    else:
                                                        faults = faults
                                                    if fixed:
                                                        if not faults:
                                                            self.line_holds = self.line_holds + 1
                                                            out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                                                            continue
                                            if faults:
                                                self.line_holds = self.line_holds + 1
                                                self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                                                out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                                                continue
                                            lines = self.cart_ids()
                                            read = self.fill_facts(lines, self.tail() + (1 if 2 <= len(lines) else 3))
                                            self.show_links(lines, self.tail() + 1)
                                            read += self.cheaper_rows()
                                            verdict = self.check_cart()
                                            if verdict:
                                                if read:
                                                    seen = "Halo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n'
                                                else:
                                                    seen = ''
                                                out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                                                continue
                                            if reread:
                                                if SPEND_SOFT < self.cost:
                                                    out.append((cid, "Not run yet (no step used). Halo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                                                    continue
                                        else:
                                            self.held += 0 if swap else 1
                                            out.append((cid, 'Not run yet (no step used). ' + (swap or '{} in the cart is sold out: replace that line first.'.format(', '.join(held))) + '\nCart now: ' + self.cart_text()))
                                            continue
                                    else:
                                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                                        continue
                                else:
                                    self.refused_change = ''
                                    self.trace.append({'kind': 'same_cart_blocked', 'cart': self.cart_mark()})
                                    out.append((cid, 'Not run yet (no step used). The store refused this exact cart and the refusal changed the task. Apply the change to the cart first - a line that no longer fits has to be replaced - then check out. Cart now: ' + self.cart_text()))
                                    continue
                            elif self.left() > 5:
                                review = self.audit()
                                if review:
                                    out.append((cid, 'Not run yet (no step used). ' + review))
                                    continue
                                elif not (self.leads and self.left() > 4 and (0 == self.refuse_blocks)):
                                    if self.leads:
                                        if self.left() > 4:
                                            if 1 == self.refuse_blocks:
                                                self.refuse_blocks = 2
                                                first = self.leads[0]
                                                self.check_cart([first], SPEND_HARD)
                                                if self.verdicts.get(self.check_key([first])):
                                                    if first in self.leads:
                                                        self.confirmed = first
                                                        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                                                        out.append((cid, "Not run yet (no step used). An independent check of {}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement.".format(first)))
                                                        continue
                                    if not self.ptr_block:
                                        if self.left() > 4:
                                            live = set((need for _, need in self.needs_now()))
                                            open_refs = list((need for _, need in self.refs if need in live and need not in self.ref_values))
                                            lines = list((i for i in self.built if not self.sold_out(i)))
                                            if open_refs:
                                                if len(self.items) <= len(lines):
                                                    self.ptr_block = True
                                                    self.trace.append({'kind': 'refuse_blocked_pointer', 'needs': open_refs[:3]})
                                                    out.append((cid, 'Not run yet (no step used). These needs take their value from another listing and that value was never read: ' + ' | '.join(open_refs[:3]) + '. A need with no value read cannot show that a listing fails, so it is no reason to end without a purchase.' + ' Find the listing each one describes and open its page in full (then every listing that page sends you to) if the steps allow;' + ' otherwise put the lines that meet every requirement you do hold in the cart and check out.\nCart now: ' + self.cart_text()))
                                                    continue
                                else:
                                    self.refuse_blocks = 1
                                    self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                                    out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                                    continue
                    if self.is_refuse(name, args):
                        if self.left() <= 5:
                            if not self.late_held:
                                lines = list((i for i in self.built if not self.sold_out(i)))
                                held_386 = None is not self.cart_view
                                if held_386:
                                    held_386 = self.cart_at <= self.cart_seen_at
                                fresh = held_386
                                missing = list((i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))))
                                if len(self.items) <= len(lines):
                                    if len(missing) + 1 <= self.left():
                                        self.late_held = True
                                        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                                        if missing:
                                            redo = 'Put {} back in the cart at the requested quantities, then check out.'.format(', '.join(missing))
                                        else:
                                            redo = 'Check out the cart as it is.'
                                        out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                                        continue
                    if self.is_refuse(name, args):
                        if not self.leads:
                            if not self.pushed:
                                if self.left() > self.max_steps // 3:
                                    self.pushed = True
                                    out.append((cid, "Not run yet (no step used). {} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that.".format(self.left())))
                                    continue
                    if self.is_refuse(name, args):
                        done = self.complete_cart()
                        if not done:
                            args = self.quiet(name, args)
                        else:
                            out.append((cid, done + (' [episode finished]' if self.finished else '')))
                            continue
                    if self.is_buy(name, args):
                        late = self.late_question()
                        if late:
                            out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                            continue
                    if ending:
                        if self.left() >= 2:
                            go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                            if not go:
                                out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                                continue
                            elif said:
                                trimmed = '\n[Halo] ' + said + trimmed
                    reply = self.act(name, args)
                    if None is reply:
                        out.append((cid, 'The call failed (network or server). You may try again.'))
                        continue
                    observation = reply.get('observation')
                    if many:
                        if isinstance(args.get(many), list):
                            if 1 < len(args[many]):
                                shown = _dump(observation)
                                if any((str(i) in shown for i in args[many])):
                                    self.compare_ok = max(self.compare_ok, len(args[many]))
                                else:
                                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                                    trimmed += '\n[Halo] The side-by-side view showed nothing for {} listings: send at most {} per call.'.format(len(args[many]), self.compare_room())
                    self.recent = (self.recent + [mark])[-3:]
                    self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
                    self.absorb(observation)
                    if name == self.roles.get('compare'):
                        rows = self.records_in(observation)
                        if rows:
                            self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
                    if name in readers - {self.roles.get('search')}:
                        self.note_view(name, observation)
                    plain = set((tool for job, tool in self.roles.items() if 'search' != job))
                    if name not in plain and (not ending):
                        paged = self.more_pages(name, args, observation)
                    else:
                        paged = ''
                    if self.single_target(name):
                        if name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                            if isinstance(observation, dict):
                                listing = self.ident(observation)
                            else:
                                listing = None
                            if listing:
                                self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                                if name == self.roles.get('open'):
                                    if not self.facts_from[name]:
                                        other = next((t for t in self.schemas if name != t and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                                        if other:
                                            self.roles['open'] = other
                                if not self.facts_from[name]:
                                    if name != self.roles.get('open'):
                                        if not self.roles.get('offer'):
                                            self.roles['offer'] = name
                                if name == self.offer_tool():
                                    self.offer_seen[listing] = self.step
                    if name == self.roles.get('cart'):
                        self.note_cart(name, args)
                        self.cart_said = _dump(observation)
                        self.recent = []
                        counts = list((_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None))
                        for value in args.values():
                            if counts:
                                if isinstance(value, str):
                                    if value in self.rows:
                                        if 0 >= counts[0] and (not _in_rows(observation, value)):
                                            self.dropped.add(value)
                                        elif 0 < counts[0]:
                                            self.dropped.discard(value)
                        lines = list((i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped))
                        if len(self.built) <= len(lines):
                            self.built = lines
                    if name == self.roles.get('search'):
                        if self.records_in(observation):
                            self.searched = (self.searched + [dict(args)])[-8:]
                    if name == self.roles.get('cart_view'):
                        self.cart_view = observation
                        self.cart_seen_at = self.step
                    if self.is_buy(name, args):
                        self.buys = self.buys + 1
                        if not reply.get('finished'):
                            self.refused_at = self.step
                    text = self.show(observation) + paged + trimmed
                    if name not in plain:
                        if not ending:
                            text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
                    base = text
                    proof = self.evidence(observation)
                    if proof:
                        text += '\n' + proof
                    if name == self.roles.get('cart'):
                        if any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                            if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                                text += '\n[Halo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
                    self.note_sold = False
                    lines = self.loose_lines(observation)
                    meaning = ''
                    if lines:
                        meaning = self.take_note(lines)
                        if meaning:
                            text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
                    held_387 = self.is_buy(name, args)
                    if held_387:
                        held_387 = not reply.get('finished')
                    refused = held_387
                    if refused:
                        if meaning:
                            self.refused_change = self.cart_mark()
                            self.trace.append({'kind': 'refused_change', 'cart': self.cart_mark(), 'change': meaning[:200]})
                            text += '\n[Halo] The store refused this cart and the refusal changes the task. Re-pick for the change and set the cart again: the same cart cannot pass now.'
                    held_388 = refused
                    if held_388:
                        held_388 = any((line in self.spoken(observation) for line in self.sold_lines))
                    again = held_388
                    held_389 = self.note_sold
                    if held_389:
                        held_390 = refused
                        if not held_390:
                            held_391 = not self.note_named
                            if held_391:
                                held_391 = not self.is_buy(name, args)
                            held_390 = held_391
                        held_389 = held_390
                    loose = held_389
                    if loose or again:
                        if self.cart_ids():
                            if not self.finished:
                                seen = self.find_sold()
                                if seen:
                                    text += "\n[Halo re-read the cart's lines after the refusal]\n" + seen
                    self.note_sold = False
                    swap = self.swap_hint()
                    if swap:
                        text += '\n' + swap
                    held_392 = self.fact_ids(observation)
                    self.told_ids.update(held_392)
                    own = list((i for i in self.full if i not in self.told_ids and _fact_block(self.full.get(i) or {})))
                    if own:
                        self.told_ids.update(own)
                        said = self.evidence({'records': list((self.full[i] for i in own[:10]))})
                        if said:
                            self.trace.append({'kind': 'own_reads_told', 'ids': own[:10]})
                            text += '\n[Halo opened these listings itself; you have not seen their records: ' + ', '.join(own[:10]) + '] ' + said
                    if self.queued:
                        text += "\n[notes seen on Halo's own calls] " + ' | '.join(self.queued)
                        self.queued = []
                    if name in readers:
                        self.read_cache[mark] = (self.version, self.step, base)
                    if reply.get('finished'):
                        text += ' [episode finished]'
                    out.append((cid, text))
                else:
                    out.append((cid, "Not run (no step used): Halo's interviewer handles every message to the shopper."))
                    continue
            else:
                out.append((cid, 'Shown in the side-by-side result above: Halo opened these listings together in one step.'))
                continue
        return out

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if refuse:
            faults: list[str] = []
            held_395 = dict(self.brief.get('refuse_args') or {})
            held_393 = self.quiet(refuse, held_395)
            held_394 = self.schemas.get(refuse)
            _shape(held_393, held_394, 'args', faults)
            held_396 = not faults
            return held_396
        else:
            return False

    def finish(self, again: bool=False) -> str:
        if not (self.finished or self.broke or self.left() <= 0):
            buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
            held_397 = None is not self.cart_view
            if held_397:
                held_397 = self.cart_at <= self.cart_seen_at
            fresh = held_397
            if fresh:
                holding = bool(self.cart_ids())
            else:
                holding = bool(self.cart_lines)
            whole = True
            if holding:
                if 2 <= len(self.items):
                    held_398 = self.fill_short_cart()
                    if not held_398:
                        held_398 = not self.can_refuse()
                    whole = held_398
                    if whole:
                        if len(self.cart_ids()) >= 2:
                            held_399 = self.cart_ids()
                            held_400 = self.tail()
                            self.fill_facts(held_399, held_400)
                            if self.left() >= 3:
                                if self.model_room() > 5:
                                    held_401 = self.cart_ids()
                                    held_402 = self.tail()
                                    self.show_links(held_401, held_402)
                    if not whole:
                        self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
            if not (self.finished or self.left() <= 0):
                if whole:
                    if buy:
                        if holding or (self.cart_touched and (not fresh) and (self.left() >= 2)):
                            faults: list[str] = []
                            held_403 = dict(self.brief.get('buy_args') or {})
                            held_404 = self.schemas.get(buy)
                            args = _shape(held_403, held_404, 'args', faults)
                            if not faults:
                                if self.left() >= 2:
                                    self.gate('order')
                                if not (self.finished or self.left() <= 0):
                                    reply = self.act(buy, args)
                                    self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                                    if self.finished or self.left() <= 0:
                                        return ''
                                    elif None is not reply:
                                        if again:
                                            self.buys = self.buys + 1
                                            self.refused_at = self.step
                                            held_405 = reply.get('observation')
                                            self.observe(buy, held_405)
                                            held_406 = 'the checkout was refused: ' + self.show(reply.get('observation'))
                                            return held_406
                                else:
                                    return ''
                if refuse:
                    faults = []
                    held_409 = dict(self.brief.get('refuse_args') or {})
                    held_407 = self.quiet(refuse, held_409)
                    held_408 = self.schemas.get(refuse)
                    args = _shape(held_407, held_408, 'args', faults)
                    if not faults:
                        if self.left() >= 2:
                            self.gate('end')
                        if not (self.finished or self.left() <= 0):
                            reply = self.act(refuse, args)
                            self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                            if None is not reply:
                                if not self.finished:
                                    held_410 = reply.get('observation')
                                    self.observe(refuse, held_410)
                                    held_411 = 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
                                    return held_411
                        else:
                            return ''
                return ''
            else:
                return ''
        else:
            return ''

    def trim(self) -> None:
        size = sum((len(_dump(m.get('content'))) for m in self.chat))
        if not HISTORY_LIMIT >= size:
            tools = list((i for i, m in enumerate(self.chat) if m.get('role') == 'tool'))
            for index in tools[:max(1, len(tools) // 2)]:
                held_412 = self.chat[index].get('content')
                if not held_412:
                    held_412 = ''
                content = held_412
                if 600 < len(content):
                    ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                    ids = list((i for i in dict.fromkeys(ids) if i in self.rows))[:20]
                    self.chat[index]['content'] = content[:400] + '...(older result shortened; listings in it: {})'.format(', '.join(ids))
        else:
            return

    def run(self) -> list[dict[str, Any]]:
        if not self.finished:
            self.read_request()
            self.trace.append({'kind': 'brief', 'brief': self.brief})
            opening = self.view.get('opening')
            if None is not opening:
                self.absorb(opening)
            intro = "SHOPPER'S REQUEST:\n" + self.request
            if None is not opening:
                intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
            self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
            self.work(LOOP_CALLS, SPEND_HARD)
            if not self.finished:
                turned = self.finish(again=True)
                if turned:
                    if not self.finished:
                        if not self.broke:
                            if not self.no_credit:
                                if self.left() > 1:
                                    self.chat.append({'role': 'user', 'content': '[Halo] Halo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                                    self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
                if not self.finished:
                    self.finish()
            self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
            held_414 = self.trace
            return held_414
        else:
            held_413 = self.trace
            return held_413

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (loop_cap > self.loop_calls) and (spend_cap > self.cost) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if not self.ready_to_ask():
                self.trim()
                self.loop_calls = self.loop_calls + 1
                message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
                if not None is message:
                    calls = []
                    for number, call in enumerate(message.get('tool_calls') or []):
                        if isinstance(call, dict):
                            if isinstance(call.get('function'), dict):
                                fn = call.get('function')
                            else:
                                fn = {}
                            raw = fn.get('arguments')
                            if isinstance(raw, dict):
                                parsed = raw
                            else:
                                parsed = _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                            calls.append({'id': str(call.get('id') or 'k{}_{}'.format(self.loop_calls, number)), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
                        else:
                            continue
                    content = message.get('content')
                    if not isinstance(content, str):
                        if content:
                            content = _dump(content)
                        else:
                            content = ''
                    if calls:
                        idle = 0
                        committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
                        if not self.asked:
                            if self.roles.get('ask'):
                                if self.msgs_free() > 0:
                                    if any((c['function']['name'] in committing for c in calls)):
                                        if self.left() > 8:
                                            if self.clock() <= ASK_BY:
                                                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                                                for call in calls:
                                                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                                                summary = self.interview()
                                                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                                                continue
                        self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                        self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': list((c['function'] for c in calls))})
                        replies = self.run_calls(calls)
                        for index, (cid, text) in enumerate(replies):
                            if len(replies) - 1 == index:
                                text += '\n' + self.status()
                            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})
                    else:
                        idle = idle + 1
                        self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                        self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                        if not 3 <= idle:
                            continue
                        else:
                            break
                else:
                    self.misses = self.misses + 1
                    if not (self.no_credit or ALL_CALLS <= self.calls or 2 < self.misses):
                        continue
                    else:
                        break
            else:
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Halo(problem_data)
    except Exception as exc:
        held_415 = [{'kind': 'crash', 'error': '{}: {}'.format(type(exc).__name__, exc)[:400]}]
        return held_415
    try:
        held_416 = episode.run()
        return held_416
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': '{}: {}'.format(type(exc).__name__, exc)[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': '{}: {}'.format(type(again).__name__, again)[:400]})
        held_417 = episode.trace
        return held_417
__all__ = ['agent_main']
