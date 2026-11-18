from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

class _Branch(Exception):
    pass
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
# With this few messages the first product question names one property. In race 16 the shopper refused list questions
# ("smaller bites", "split them up") on most tasks, and a refusal spends one of the three messages; in race 15 no list
# was refused. Asking the top-ranked property alone first keeps that message whatever the shopper's habit.
FIRST_ALONE = 3
SWEEP_QUERIES = 4
SWEEP_OPENS = 2
# The cheaper-listing sweep runs at most once per episode, before the first checkout. (The note handler resets
# wider_held after a sold-out note, so without this cap the sweep could run again at a later checkout.)
SWEEP_RUNS = 1
TITLE_STOP = '^(?:amazon|brand|basics|with|from|pack|piece|pieces|inch|inches|size|sizes|large|small|medium|women|womens|mens|unisex|kids|gift|gifts|new|best|premium|quality|style|design|home|perfect|ideal|great|cute|set|sets|color|colors|colour|colours)$'
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
GUIDE = 'You are Halo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Halo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Halo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Halo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Halo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Halo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When it asks for whatever \"goes with\" that value, the store\'s own pairing rule decides (a listing\'s comments say, for example, \"Our color rule: black goes together with white\"): the wanted value is the partner the rule gives, not the pointed-at value itself; Halo works it out and shows it in the ledger.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Halo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page for its own listing (never a value the page says goes together with it), or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
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
    _seq1 = list(re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I))
    _ix2 = 0
    while _ix2 < len(_seq1):
        hit = _seq1[_ix2]
        _ix2 = _ix2 + 1
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
    try:
        True if depth > 7 else (_ for _ in ()).throw(_Branch())
        return '...'
    except _Branch:
        pass
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    try:
        True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    except _Branch:
        pass
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        return [_clip(v, limit, depth + 1) for v in value]
    except _Branch:
        pass
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
    try:
        True if 0 <= left < right else (_ for _ in ()).throw(_Branch())
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    except _Branch:
        pass
    return None

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _drive_scope(key: str) -> str:
    return _words(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _snap_delta(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        try:
            True if len(w) >= 3 and w not in ('and', 'for', 'the', 'with') else (_ for _ in ()).throw(_Branch())
            out.add(w)
        except _Branch:
            pass
    return out

def _flip_ring94(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

def _drive_delta50(prop: frozenset[str], key: frozenset[str]) -> bool:
    try:
        True if not prop or not key else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    try:
        True if kind == 'string' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, str)
    except _Branch:
        pass
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    try:
        True if kind == 'number' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    except _Branch:
        pass
    try:
        True if kind == 'boolean' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, bool)
    except _Branch:
        pass
    if kind == 'array':
        return isinstance(value, list)
    try:
        True if kind == 'object' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, dict)
    except _Branch:
        pass
    try:
        True if kind == 'null' else (_ for _ in ()).throw(_Branch())
        return value is None
    except _Branch:
        pass
    return True

def _flip_base(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    try:
        True if not isinstance(schema, dict) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
    kinds = _kinds(schema)
    try:
        True if kinds and (not any((_fits_kind(value, k) for k in kinds))) else (_ for _ in ()).throw(_Branch())
        _seq3 = list(kinds)
        _ix4 = 0
        while _ix4 < len(_seq3):
            kind = _seq3[_ix4]
            _ix4 = _ix4 + 1
            try:
                True if kind == 'integer' and isinstance(value, float) and value.is_integer() else (_ for _ in ()).throw(_Branch())
                value = int(value)
                break
            except _Branch:
                pass
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            try:
                True if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)) else (_ for _ in ()).throw(_Branch())
                value = str(value)
                break
            except _Branch:
                pass
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            try:
                True if kind == 'array' and (not isinstance(value, list)) and (value is not None) else (_ for _ in ()).throw(_Branch())
                value = [value]
                break
            except _Branch:
                pass
        try:
            True if not any((_fits_kind(value, k) for k in kinds)) else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
        except _Branch:
            pass
    except _Branch:
        pass
    options = schema.get('enum')
    try:
        True if isinstance(options, list) and value not in options else (_ for _ in ()).throw(_Branch())
        try:
            True if isinstance(value, str) else (_ for _ in ()).throw(_Branch())
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        except _Branch:
            pass
        try:
            True if value not in options else (_ for _ in ()).throw(_Branch())
            faults.append(f'{where}: allowed values are {options}')
        except _Branch:
            pass
    except _Branch:
        pass
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    try:
        True if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']) else (_ for _ in ()).throw(_Branch())
        faults.append(f'{where} is too short')
    except _Branch:
        pass
    try:
        True if isinstance(value, (int, float)) and (not isinstance(value, bool)) else (_ for _ in ()).throw(_Branch())
        try:
            True if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum'] else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} is below {schema['minimum']}")
        except _Branch:
            pass
        try:
            True if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum'] else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} is above {schema['maximum']}")
        except _Branch:
            pass
    except _Branch:
        pass
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        try:
            True if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems'] else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        except _Branch:
            pass
        try:
            True if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems'] else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        except _Branch:
            pass
        try:
            True if isinstance(schema.get('items'), dict) else (_ for _ in ()).throw(_Branch())
            value = [_flip_base(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
        except _Branch:
            pass
    except _Branch:
        pass
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        _seq5 = list(schema.get('required') or [])
        _ix6 = 0
        while _ix6 < len(_seq5):
            name = _seq5[_ix6]
            _ix6 = _ix6 + 1
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _flip_base(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _snap_cell(value: Any) -> int | None:
    try:
        True if isinstance(value, bool) else (_ for _ in ()).throw(_Branch())
        return None
    except _Branch:
        pass
    if isinstance(value, int):
        return value
    try:
        True if isinstance(value, float) and value.is_integer() else (_ for _ in ()).throw(_Branch())
        return int(value)
    except _Branch:
        pass
    try:
        True if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value) else (_ for _ in ()).throw(_Branch())
        return int(float(value))
    except _Branch:
        pass
    return None

def _draw_flag(value: Any, named: bool=False) -> bool:
    try:
        True if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip() else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return named or bool(re.search('\\d', value))

def _flip_mark35(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_draw_flag(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _flip_sigma21(node: Any, ident: str, depth: int=0) -> bool:
    try:
        True if depth > 5 else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    try:
        True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_flip_sigma21(v, ident, depth + 1) for v in node))
    except _Branch:
        pass
    try:
        True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
        return any((_flip_sigma21(v, ident, depth + 1) for v in node.values()))
    except _Branch:
        pass
    return False

def _llm_within(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    try:
        True if seconds >= 50 else (_ for _ in ()).throw(_Branch())
        return _llm
    except _Branch:
        pass
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _step_core(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            try:
                True if len(scalars) >= 2 and (best is None or len(scalars) > len(best)) else (_ for _ in ()).throw(_Branch())
                best = scalars
            except _Branch:
                pass
    return best

# Store pairing rules. Race 15 shoppers answered with pointers such as "colour ... anything that goes with just what
# the listing for <shop>'s <kind>, the one with ..., gives", and the pages along that chain carry the store's rule for
# the property ("Our color rule: black goes together with white, and white goes together with black") or a statement
# about the listing itself ("Its component type is camera, which goes together with camera"). The wanted value is the
# partner the rule gives, not the pointed-at value: Halo v2 and Big passed 0 of 20 such episodes (same-as pointers 6 of
# 8). Races 14 and 15 shop comments held 49 distinct rule sentences.
PAIRS_WITH = r'goes (?:together )?with|pairs? with|is paired with'
_RULE_HEAD = re.compile(
    r"(?:\brule for (?P<p1>[a-z][\w &/'-]*?)(?=\s*(?:,|:|\bpairs\b|\bis that\b|\bruns\b|\bacross\b))"
    r"|\b(?:our|the store's|store|this store's|shop) (?P<p2>[a-z][\w &/'-]*?) rule\b)", re.I)
_PAIR = re.compile(r"^\s*(?:and\s+)?(?:that\s+)?(?P<a>[\w][\w &/.+'-]*?)\s+(?:goes (?:together )?with|with|pairs with)\s+(?P<b>[\w][\w &/.+'-]*?)\s*$", re.I)
_OWN = re.compile(r"\b(?:its|that|this listing's|this item's)\s+(?P<prop>[a-z][\w &/'-]*?)(?:\s+value)?(?:,\s*(?P<val>[\w &/.+'-]+?),|\s+is\s+(?P<val2>[\w &/.+'-]+?),\s*which)?\s+goes (?:together )?with\s+(?P<w>[\w &/.+'-]+)", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?;])\s+|\s*\|\s*|\n', text) if s.strip()]


def store_pairings(text: str) -> list[tuple[str, str, str]]:
    """Store-wide pairing rules a page states, as (property, value, partner) in the rule's own direction:
    "Our rule for material: acrylic goes with hydrogel, and tempered glass goes with plastic" and "The store's rule for
    product form pairs emulsion with serum, and liquid with cream" both give one triple per pair. A rule that names
    no property gives property ''."""
    found = []
    for sentence in _sentences(str(text)):
        store_wide = 'rule' in sentence.lower() or re.match(r'\s*in this (?:store|shop)\b', sentence, re.I)
        if not store_wide or not re.search(PAIRS_WITH + r'|\bwith\b', sentence, re.I):
            continue
        sentence = re.sub(r'^\s*in this (?:store|shop),?\s*', '', sentence, flags=re.I)
        head = _RULE_HEAD.search(sentence)
        prop = ((head.group('p1') or head.group('p2')) if head else '') or ''
        body = sentence[head.end():] if head else sentence
        body = re.sub(r"^\s*(?:,\s*across the range)?\s*(?::|,|\bpairs\b|\bis that\b|\bruns:?)\s*", '', body, flags=re.I)
        body = re.sub(r'^(?:pairs\s+)', '', body.strip(), flags=re.I).rstrip('.')
        for part in re.split(r',\s*(?:and\s+)?|\s+and\s+(?=[\w &/.+\'-]+?\s+(?:goes|with)\b)', body):
            pair = _PAIR.match(part)
            if pair:
                found.append((prop.strip().lower(), pair.group('a').strip().lower(), pair.group('b').strip().lower()))
    return found


def listing_pairings(text: str) -> list[tuple[str, str, str]]:
    """Pairings a page states about its own listing, as (property, own value or '', partner): "Its cooking style value,
    chinese, goes together with western", "Its component type is camera, which goes together with camera", "Its
    material goes together with tempered glass", and with no property named, "A single camera goes together with
    triple" (value "single camera")."""
    found = []
    for sentence in _sentences(str(text)):
        if 'rule' in sentence.lower():
            continue
        own = list(_OWN.finditer(sentence))
        for m in own:
            found.append((m.group('prop').strip().lower(), (m.group('val') or m.group('val2') or '').strip().lower(), m.group('w').strip().rstrip('.').lower()))
        bare = None if own else re.search(r"\b(?:a|an|the|this)\s+(?P<val>[\w&/.+'-]+(?:\s+[\w&/.+'-]+)?)\s+goes (?:together )?with\s+(?P<w>[\w &/.+'-]+)", sentence, re.I)
        if bare:
            found.append(('', bare.group('val').strip().lower(), bare.group('w').strip().rstrip('.').lower()))
    return found


def wants_partner(need: str) -> bool:
    """Whether a requirement asks for the value the store pairs with another value ("anything that goes with just
    what the listing for ... gives") rather than that value itself."""
    return bool(re.search(r'\bgoes (?:together )?with\b', need, re.I))


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
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _snap_cell(self.view.get('max_steps'))
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
        self.paired: dict[str, tuple[str, str]] = {}
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
        self.swept: set[str] = set()
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
        return self.max_steps - self.step

    def wrap_sigma14(self) -> float:
        return time.monotonic() - self.started

    def open_index(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.wrap_sigma14()

    def fetch_grid87(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def drive_grid94(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def drive_pool(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def wrap_base(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.push_unit34(t)), None)
        try:
            True if known else (_ for _ in ()).throw(_Branch())
            return known
        except _Branch:
            pass
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.push_unit34(tool) else None

    def fold_node(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.fold_node()
        return seen is False or (seen is None and (self.facts_from.get(self.wrap_base() or '') is True or bool(self.roles.get('offer'))))

    def fold_base83(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.drive_grid94() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.fold_base83('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        try:
            True if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD else (_ for _ in ()).throw(_Branch())
            return [HEAVY, LIGHT]
        except _Branch:
            pass
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        _seq7 = list(models)
        _ix8 = 0
        while _ix8 < len(_seq7):
            model = _seq7[_ix8]
            _ix8 = _ix8 + 1
            try:
                True if self.calls >= ALL_CALLS else (_ for _ in ()).throw(_Branch())
                return None
            except _Branch:
                pass
            if self.open_index() < CALL_FLOOR:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.wrap_sigma14(), 1)})
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
                if self.open_index() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            try:
                True if status == 402 else (_ for _ in ()).throw(_Branch())
                self.no_credit = True
                return None
            except _Branch:
                pass
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            _seq9 = list(marks)
            _ix10 = 0
            while _ix10 < len(_seq9):
                index = _seq9[_ix10]
                _ix10 = _ix10 + 1
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        try:
            True if tools else (_ for _ in ()).throw(_Branch())
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        except _Branch:
            pass
        result = _llm_within(self.open_index()).post_verbose('/inference/chat/completions', body)
        data = result.data
        try:
            True if isinstance(data, dict) and data.get('choices') else (_ for _ in ()).throw(_Branch())
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            try:
                True if message.get('content') or message.get('tool_calls') else (_ for _ in ()).throw(_Branch())
                return (message, 200)
            except _Branch:
                pass
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        except _Branch:
            pass
        error = result.error or {}
        status = error.get('status')
        try:
            True if isinstance(data, dict) and isinstance(data.get('error'), dict) else (_ for _ in ()).throw(_Branch())
            status = _snap_cell(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        except _Branch:
            pass
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def slot_cell(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            try:
                True if got is not None else (_ for _ in ()).throw(_Branch())
                return got
            except _Branch:
                pass
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def draw_port25(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            try:
                True if isinstance(result.data, dict) else (_ for _ in ()).throw(_Branch())
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                try:
                    True if data.get('finished') else (_ for _ in ()).throw(_Branch())
                    self.finished = True
                except _Branch:
                    pass
                seen = data.get('observation')
                if name in self.drive_pool() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            except _Branch:
                pass
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            try:
                True if status == 409 else (_ for _ in ()).throw(_Branch())
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
            except _Branch:
                pass
            try:
                True if status in (400, 401, 402, 403, 404, 422) else (_ for _ in ()).throw(_Branch())
                if status == 402:
                    self.broke = True
                return None
            except _Branch:
                pass
        return None

    def draw_cell(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.slot_cell(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.slot_cell(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.no_credit and self.lift_rail70(brief):
            again = self.slot_cell(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            try:
                True if not isinstance(brief.get(key), dict) else (_ for _ in ()).throw(_Branch())
                brief[key] = {}
            except _Branch:
                pass
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        try:
            True if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']) else (_ for _ in ()).throw(_Branch())
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.slot_cell(ask, [LIGHT, HEAVY], 1500) or {}
            try:
                True if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']) else (_ for _ in ()).throw(_Branch())
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
            except _Branch:
                pass
        except _Branch:
            pass
        _seq11 = list(_items(brief.get('items')))
        _ix12 = 0
        while _ix12 < len(_seq11):
            entry = _seq11[_ix12]
            _ix12 = _ix12 + 1
            try:
                True if not isinstance(entry, dict) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            quantity = _snap_cell(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        try:
            True if not self.items else (_ for _ in ()).throw(_Branch())
            self.items = [Item('the requested product', '', 1, [])]
        except _Branch:
            pass
        _seq13 = list(_items(brief.get('fits')))
        _ix14 = 0
        while _ix14 < len(_seq13):
            pair = _seq13[_ix14]
            _ix14 = _ix14 + 1
            try:
                True if isinstance(pair, dict) else (_ for _ in ()).throw(_Branch())
                part, main = (became.get(_snap_cell(pair.get('item')) or 0), became.get(_snap_cell(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
            except _Branch:
                pass
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
        try:
            True if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget) else (_ for _ in ()).throw(_Branch())
            budget = float(budget)
        except _Branch:
            pass
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _snap_cell(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _snap_cell(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _snap_cell(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def lift_rail70(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            try:
                True if not pointers or not rest else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.slot_cell(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            try:
                True if not moved else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def bind_line(self, number: int, need: str) -> bool:
        try:
            True if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))] else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        try:
            True if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        cat_key = self.push_port()
        keys: set[str] = set()
        read = 0
        _seq15 = list(self.full.values())
        _ix16 = 0
        while _ix16 < len(_seq15):
            record = _seq15[_ix16]
            _ix16 = _ix16 + 1
            try:
                True if cat_key and (not self.owns(item, str(record.get(cat_key) or ''))) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            block = _step_core(record) or {}
            if block:
                read += 1
                keys |= {_words(k) for k in block}
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def step_pool(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.bind_line(number, m)]
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            try:
                True if item.shelf else (_ for _ in ()).throw(_Branch())
                head += f' (store label: {item.shelf})'
            except _Branch:
                pass
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            parts.append(text)
        brief = self.brief
        try:
            True if self.budget is not None else (_ for _ in ()).throw(_Branch())
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        except _Branch:
            pass
        pick = brief.get('pick')
        try:
            True if pick == 'cheapest' else (_ for _ in ()).throw(_Branch())
            parts.append('pick: the cheapest listing that fits (lowest price)')
        except _Branch:
            try:
                True if pick in ('most', 'least') else (_ for _ in ()).throw(_Branch())
                parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
            except _Branch:
                parts.append('pick: any listing that fits everything')
        try:
            True if brief.get('in_stock_only', True) is not False else (_ for _ in ()).throw(_Branch())
            parts.append('in stock only')
        except _Branch:
            pass
        _seq17 = list(_items(brief.get('wishes')))
        _ix18 = 0
        while _ix18 < len(_seq17):
            wish = _seq17[_ix18]
            _ix18 = _ix18 + 1
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _items(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        _seq19 = list(self.process)
        _ix20 = 0
        while _ix20 < len(_seq19):
            entry = _seq19[_ix20]
            _ix20 = _ix20 + 1
            when = {'choose': 'before choosing', 'order': 'at the order (Halo does it when you check out)', 'end': 'at an ending without a purchase (Halo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.wrap_base()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.drive_pool() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.drive_pool() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def shut_token40(self) -> str:
        left = self.left()
        text = f'[Halo] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        else:
            try:
                True if self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2) else (_ for _ in ()).throw(_Branch())
                text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
            except _Branch:
                pass
        try:
            True if self.wrap_sigma14() >= WIND_DOWN and left > self.tail() + 1 else (_ for _ in ()).throw(_Branch())
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        except _Branch:
            pass
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fold_flag9() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.step_pool()
        return text

    def bind_edge(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        try:
            True if self.id_key and _draw_flag(node.get(self.id_key), True) else (_ for _ in ()).throw(_Branch())
            return node[self.id_key].strip()
        except _Branch:
            pass
        nested = any((isinstance(v, dict) for v in node.values()))
        try:
            True if not (in_list or depth == 0 or nested) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        found = _flip_mark35([node])
        try:
            True if found and ('id' in str(found).lower() or node[found].strip() in self.rows) else (_ for _ in ()).throw(_Branch())
            return node[found].strip()
        except _Branch:
            pass
        return None

    def build_span(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        try:
            True if not rows or len(rows) < len(node) / 2 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        if self.id_key and all((_draw_flag(r.get(self.id_key), True) for r in rows)):
            return True
        found = _flip_mark35(rows)
        try:
            True if found else (_ for _ in ()).throw(_Branch())
            self.id_key = found
            return True
        except _Branch:
            pass
        return False

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.bind_edge(node, in_list, depth)
                try:
                    True if ident else (_ for _ in ()).throw(_Branch())
                    try:
                        True if ident not in self.rows else (_ for _ in ()).throw(_Branch())
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    except _Branch:
                        pass
                    old = self.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _step_core(node):
                        self.full[ident] = dict(self.rows[ident])
                    else:
                        try:
                            True if ident in self.full else (_ for _ in ()).throw(_Branch())
                            self.full[ident].update(node)
                        except _Branch:
                            pass
                    return
                except _Branch:
                    pass
                _seq21 = list(node.values())
                _ix22 = 0
                while _ix22 < len(_seq21):
                    value = _seq21[_ix22]
                    _ix22 = _ix22 + 1
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.build_span(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            try:
                True if depth > 6 or not isinstance(schema, dict) else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            for option in schema.get('enum') or []:
                try:
                    True if isinstance(option, str) else (_ for _ in ()).throw(_Branch())
                    found.add(option.strip().lower())
                except _Branch:
                    pass
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        _seq23 = list(self.schemas.values())
        _ix24 = 0
        while _ix24 < len(_seq23):
            schema = _seq23[_ix24]
            _ix24 = _ix24 + 1
            walk(schema, 0)
        return found

    def push_port(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        _seq25 = list(self.rows.values())
        _ix26 = 0
        while _ix26 < len(_seq25):
            row = _seq25[_ix26]
            _ix26 = _ix26 + 1
            for key, value in row.items():
                try:
                    True if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)) else (_ for _ in ()).throw(_Branch())
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
                except _Branch:
                    pass
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            try:
                True if best is None or score < best[0] else (_ for _ in ()).throw(_Branch())
                best = (score, key)
            except _Branch:
                pass
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.push_port()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _step_core(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                try:
                    True if not name or name in done else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        try:
            True if matched else (_ for _ in ()).throw(_Branch())
            groups = {label: group for label, group in groups.items() if label in matched}
        except _Branch:
            pass
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                try:
                    True if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                try:
                    True if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                twin = next((spec for other, spec in kept if self.bind_mark(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        _seq27 = list(self.items[number - 1].musts)
        _ix28 = 0
        while _ix28 < len(_seq27):
            must = _seq27[_ix28]
            _ix28 = _ix28 + 1
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            try:
                True if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown) else (_ for _ in ()).throw(_Branch())
                return f' [not the stated {prop.strip()}: another property]'
            except _Branch:
                pass
        return ''

    @staticmethod
    def bind_mark(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        try:
            True if not (a <= b or b <= a) else (_ for _ in ()).throw(_Branch())
            return len(both) >= 2 and same == len(both)
        except _Branch:
            pass
        try:
            True if both else (_ for _ in ()).throw(_Branch())
            return same * 2 > len(both)
        except _Branch:
            pass
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def slot_scope(self) -> list[list[str]]:
        cat_key = self.push_port()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def push_unit34(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        try:
            True if not tool or not target or any((k != target for k in schema.get('required') or [])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return target

    def tune_stash(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        try:
            True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        except _Branch:
            pass
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def step_port94(self, ident: str) -> bool:
        try:
            True if self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def walk_mark69(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        try:
            True if isinstance(observation, dict) else (_ for _ in ()).throw(_Branch())
            ident = self.bind_edge(observation)
            try:
                True if ident else (_ for _ in ()).throw(_Branch())
                return [(ident, observation)]
            except _Branch:
                pass
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                try:
                    True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.bind_edge(r, True)] if i]
                except _Branch:
                    try:
                        True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
                        found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.bind_edge(r, True)] if i]
                    except _Branch:
                        pass
            return found
        except _Branch:
            pass
        try:
            True if isinstance(observation, list) else (_ for _ in ()).throw(_Branch())
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.bind_edge(r, True)] if i]
        except _Branch:
            pass
        return []

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.walk_mark69(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def fetch_hook(self, name: str, ident: str) -> bool:
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
        got = self.slot_cell(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        try:
            True if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)) else (_ for _ in ()).throw(_Branch())
            shape['count'] = None
        except _Branch:
            pass
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        try:
            True if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        try:
            True if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.tune_stash(i) for i, _ in self.walk_mark69(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        try:
            True if not plan else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        try:
            True if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12) else (_ for _ in ()).throw(_Branch())
            pages = 2
        except _Branch:
            pass
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return f"\n[Halo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _flip_base({**args, param: value}, self.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.draw_port25(name, call)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
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
        rest = f"\n[Halo] {count} listings matched and Halo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Halo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def snap_orbit84(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.walk_mark69(observation) if (self.tune_stash(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Halo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def slot_flag16(self, observations: list[Any]) -> str:
        try:
            True if self.brief.get('pick') != 'cheapest' or not self.price_key else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        found = {i for seen in observations for i, _ in self.walk_mark69(seen)}
        rows = [(self.tune_stash(i), i) for i in found if self.step_port94(i) and self.tune_stash(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Halo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def title_words(self, number: int, label: str) -> list[str]:
        """Words that recur in the titles of this shelf's rows seen so far, most common first, leaving out the item's
        and the shelf's own words, brands and shops, numbers, and words any search already used. A words search
        matches titles, so each such word reaches another slice of the shelf than the searches made so far."""
        cat_key = self.push_port()
        item = self.items[number - 1]
        rows = [r for r in self.rows.values() if cat_key and str(r.get(cat_key) or '').strip().lower() == label.lower()]
        skip = _snap_delta(item.name) | _snap_delta(label)
        for args in self.searched:
            for v in args.values():
                if isinstance(v, str):
                    skip |= _snap_delta(v)
        count: dict[str, int] = {}
        for r in rows:
            title = next((str(v) for k, v in r.items() if re.search('title|name', k, re.I) and (not re.search('brand|shop|seller|sold|store|vendor|categor', k, re.I)) and isinstance(v, str)), '')
            own = set()
            for k, v in r.items():
                if isinstance(v, str) and re.search('brand|shop|seller|sold|store|vendor|maker', k, re.I):
                    own |= _snap_delta(v)
            for w in _snap_delta(title) - own - skip:
                if re.fullmatch('[a-z]{4,14}', w) and (not re.search(TITLE_STOP, w)):
                    count[w] = count.get(w, 0) + 1
        return [w for w, n in sorted(count.items(), key=lambda x: -x[1]) if n >= 2][:SWEEP_QUERIES]

    def undercut(self) -> bool:
        """For a cheapest pick, before the first checkout, when the store cannot list a shelf under a price: search the
        shelf of each cart line again with words its titles share (and once with the first required value) to bring up
        rows priced below the line, then read the two that look most like a fit, cheapest first. The cart check that
        follows compares them with the cart. At most SWEEP_RUNS sweeps run per episode. Returns whether it searched."""
        plan = self.value_template()
        cat_key = self.push_port()
        room = self.left() - self.tail() - len(self.items) - 3
        mark = self.drive_grid94()
        if not plan or not cat_key or self.cap_template() or room < 4 or self.wrap_sigma14() > WIND_DOWN - 20 or mark in self.swept or len(self.swept) >= SWEEP_RUNS:
            return False
        self.swept.add(mark)
        tool, base, field = plan
        stop = self.step + min(SWEEP_QUERIES, room - SWEEP_OPENS)
        tried = {frozenset(_snap_delta(str(v))) for a in self.searched for v in a.values() if isinstance(v, str)}
        cheaper: list[tuple[int, float, str]] = []
        ran = False
        for line in self.walk_pool():
            price = self.tune_stash(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip()
            number = next((n for n, it in enumerate(self.items, 1) if label and self.owns(it, label)), 0)
            if price is None or not number:
                continue
            head = (_words(self.items[number - 1].name if re.search(r'&|\band\b|,', label) else label).split() or [''])[-1]
            head = head[:-2] if re.search('(?:ch|sh|ss|x|z)es$', head) else head[:-1] if len(head) > 3 and head.endswith('s') and (not head.endswith('ss')) else head
            values = [v for v, _ in self.build_cell(number) if len(v.split()) <= 2]
            queries = [head] + ([f'{values[0]} {head}'] if values else []) + [f'{w} {head}' for w in self.title_words(number, label)]
            for words in queries:
                if self.step >= stop or self.finished:
                    break
                if frozenset(_snap_delta(words)) in tried:
                    continue
                tried.add(frozenset(_snap_delta(words)))
                faults: list[str] = []
                args = _flip_base({**base, field: words}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.draw_port25(tool, args)
                if reply is None:
                    break
                ran = True
                seen = reply.get('observation')
                self.observe(tool, seen)
                found = list(self.walk_mark69(seen))
                if found:
                    self.searched = (self.searched + [dict(args)])[-8:]
                page = self.paging(tool, seen) if words == head else None
                if page and page[3] is not None and page[0] > page[1] and self.step < stop and (not self.finished):
                    more = _flip_base({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                    later = None if faults else self.draw_port25(tool, more)
                    if later is not None:
                        self.observe(tool, later.get('observation'))
                        found += list(self.walk_mark69(later.get('observation')))
                wanted = {w for v, _ in self.build_cell(number) for w in _snap_delta(v)}
                for ident, row in found:
                    p = self.tune_stash(ident)
                    if ident == line or ident in self.full or p is None or p >= price or (not self.step_port94(ident)) or str(row.get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() != label.lower():
                        continue
                    if ident not in [i for _, _, i in cheaper]:
                        cheaper.append((-len(wanted & _snap_delta(_dump(self.rows.get(ident) or row))), p, ident))
        cheaper.sort()
        opened = self.build_index([i for _, _, i in cheaper][:SWEEP_OPENS], self.tail() + len(self.items) + 2) if cheaper else []
        self.trace.append({'kind': 'undercut', 'ran': ran, 'cheaper': [i for _, _, i in cheaper][:6], 'opened': len(opened)})
        return ran

    def wider_hold(self) -> str:
        try:
            True if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.wider_held = True
        if self.undercut():
            return ''
        lines = [f'{i} at {self.tune_stash(i):g}' for i in self.walk_pool() if self.tune_stash(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def build_index(self, ids: list[str], keep: int) -> list[str]:
        tool = self.wrap_base()
        target = self.push_unit34(tool)
        if not tool or not target:
            return []
        texts = []
        _seq29 = list([i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))])
        _ix30 = 0
        while _ix30 < len(_seq29):
            ident = _seq29[_ix30]
            _ix30 = _ix30 + 1
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _flip_base({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.draw_port25(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.bind_edge(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_step_core(seen))
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
            try:
                True if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            if self.left() <= keep or self.finished:
                return
            self.link_tried.add(code)
            faults: list[str] = []
            args = _flip_base({**base, field: code}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            reply = self.draw_port25(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.fit_index()['marks']
            self.build_index([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.tail() + 6))

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and re.search(CAP_ARG, _drive_scope(k), re.I) and (not re.search(CAP_NOT, k, re.I))]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(SHELF_ARG, _drive_scope(k), re.I)]
            try:
                True if len(caps) != 1 or len(shelves) != 1 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                try:
                    True if not isinstance(options, list) or len(options) < 2 else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and (not any((k != shelves[0] and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _drive_scope(k), re.I) for k, v in a.items())))), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                try:
                    True if pick is None else (_ for _ in ()).throw(_Branch())
                    return None
                except _Branch:
                    pass
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def flip_flag5(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.push_port()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = self.tune_stash(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            try:
                True if price is None or price < 2 or (not label) or ((line, price) in self.capped) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            try:
                True if self.left() <= self.tail() + 5 or self.finished else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.capped.add((line, price))
            faults: list[str] = []
            args = _flip_base({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            reply = self.draw_port25(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.paging(tool, reply.get('observation'))
            if page and page[3] is not None and (page[1] < page[0] <= 2 * page[1]) and (self.left() > self.tail() + 5) and (not self.price_sorted(reply.get('observation'))):
                call = _flip_base({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                more = None if faults else self.draw_port25(tool, call)
                try:
                    True if more is not None else (_ for _ in ()).throw(_Branch())
                    self.observe(tool, more.get('observation'))
                except _Branch:
                    pass

    def draw_sigma(self) -> list[str]:
        try:
            True if self.brief.get('pick') != 'cheapest' else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        mark = self.drive_grid94() + f'#{self.version}'
        cat_key = self.push_port()
        try:
            True if mark in self.cheaper_marks or not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        self.cheaper_marks.add(mark)
        cart = self.walk_pool()
        self.flip_flag5(cart)
        picks: list[str] = []
        _seq31 = list(cart)
        _ix32 = 0
        while _ix32 < len(_seq31):
            line = _seq31[_ix32]
            _ix32 = _ix32 + 1
            price = self.tune_stash(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.step_port94(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.tune_stash(i) if self.tune_stash(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.tune_stash(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.build_index(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        meaning = self.build_grid81(lines)
        try:
            True if meaning else (_ for _ in ()).throw(_Branch())
            self.queued.append(f'(seen on {name}) {meaning}')
        except _Branch:
            pass
        return meaning

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare():
            self.fold_scope61()
            return
        try:
            True if self.left() <= 12 else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.slot_scope()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        _seq33 = list(ranked)
        _ix34 = 0
        while _ix34 < len(_seq33):
            ids = _seq33[_ix34]
            _ix34 = _ix34 + 1
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _flip_base({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.full)
        reply = self.draw_port25(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.walk_mark69(seen)
            try:
                True if rows else (_ for _ in ()).throw(_Branch())
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_step_core(r) for _, r in rows))
            except _Branch:
                pass
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            try:
                True if rows and self.facts_from.get(tool) is False else (_ for _ in ()).throw(_Branch())
                self.fold_scope61()
                return
            except _Branch:
                pass
            if len(self.full) == before and len(picks) > 2 and (self.left() > 12):
                faults = []
                args = _flip_base({target: picks[:2]}, schema, 'args', faults)
                try:
                    True if not faults else (_ for _ in ()).throw(_Branch())
                    reply = self.draw_port25(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))
                except _Branch:
                    pass

    def fold_scope61(self) -> None:
        tool = self.wrap_base()
        target = self.push_unit34(tool)
        try:
            True if not tool or not target else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        _seq35 = list(self.slot_scope())
        _ix36 = 0
        while _ix36 < len(_seq35):
            ids = _seq35[_ix36]
            _ix36 = _ix36 + 1
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.step_port94(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.tune_stash(i) if self.tune_stash(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.build_index(first[:cap][:8], 22)
        self.build_index((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def shut_token38(self) -> bool:
        try:
            True if self.asked or not self.roles.get('ask') or self.fetch_grid87() <= 0 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        try:
            True if self.left() <= 8 or self.wrap_sigma14() > ASK_BY else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def draw_cell69(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        try:
            True if not target else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            try:
                True if key not in args else (_ for _ in ()).throw(_Branch())
                spec = props.get(key) or {}
                try:
                    True if isinstance(spec.get('enum'), list) and spec['enum'] else (_ for _ in ()).throw(_Branch())
                    args[key] = spec['enum'][0]
                except _Branch:
                    return None
            except _Branch:
                pass
        faults: list[str] = []
        args = _flip_base(args, schema, 'args', faults)
        return None if faults else args

    def build_sigma85(self) -> str:
        self.asked = True
        if self.fetch_grid87() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.fetch_heap(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.fetch_grid87() > 0 and (not self.finished) and (self.left() > 6) and (self.wrap_sigma14() < QUESTION_BY):
            batch = self.open_delta(told)
            try:
                True if not batch else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        try:
            True if told else (_ for _ in ()).throw(_Branch())
            self.version += 1
        except _Branch:
            pass
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Halo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        try:
            True if searched else (_ for _ in ()).throw(_Branch())
            summary += '\n' + searched
        except _Branch:
            pass
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Halo is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        try:
            True if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        try:
            True if self.left() < LATE_STEPS or self.wrap_sigma14() > LATE_BY or (not self.judges()) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.late_asked = True
        try:
            True if not self.lanes else (_ for _ in ()).throw(_Branch())
            self.lanes = self.fetch_heap(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        except _Branch:
            pass
        told: list[str] = []
        budget = self.budget
        batch = self.open_delta(told, LATE_STEPS - 2)
        try:
            True if not batch else (_ for _ in ()).throw(_Branch())
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        except _Branch:
            pass
        try:
            True if self.budget != budget else (_ for _ in ()).throw(_Branch())
            told.append(f'budget now {self.budget}')
        except _Branch:
            pass
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Halo asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        try:
            True if pointed else (_ for _ in ()).throw(_Branch())
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        except _Branch:
            pass
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def walk_line(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _drive_scope(k), re.I)]
        _seq39 = list(reversed(self.searched))
        _ix40 = 0
        while _ix40 < len(_seq39):
            args = _seq39[_ix40]
            _ix40 = _ix40 + 1
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.push_port()) or '').strip().lower() for r in self.rows.values()} - {''} if self.push_port() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            try:
                True if fields and tool else (_ for _ in ()).throw(_Branch())
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.push_port()
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                _seq37 = list(list(keep))
                _ix38 = 0
                while _ix38 < len(_seq37):
                    k = _seq37[_ix38]
                    _ix38 = _ix38 + 1
                    try:
                        True if isinstance((props.get(k) or {}).get('enum'), list) else (_ for _ in ()).throw(_Branch())
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                    except _Branch:
                        pass
                return (tool, keep, field)
            except _Branch:
                pass
        return None

    def build_cell(self, number: int) -> list[tuple[str, bool]]:
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
            try:
                True if edge else (_ for _ in ()).throw(_Branch())
                if need in told:
                    out.append((edge.group(1).strip(), True))
                continue
            except _Branch:
                pass
            try:
                True if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            out.append((value, need in told))
        return out

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.walk_line()
        tool = self.roles.get('search')
        if plan or not tool or (not self.searched):
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(QUERY_WORDS, _drive_scope(k), re.I)]
        last = self.searched[-1]
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and (len(others) == 1):
                modes.append((key, others[0]))
        try:
            True if len(named) != 1 or len(modes) > 1 or (enums and (not modes)) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        try:
            True if not plan or self.left() < 22 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        _seq41 = list(numbers)
        _ix42 = 0
        while _ix42 < len(_seq41):
            number = _seq41[_ix42]
            _ix42 = _ix42 + 1
            pairs = self.build_cell(number)
            item = self.items[number - 1]
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told]
            try:
                True if not fresh or budget <= 2 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [[values[0], item.name], values] if lead else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _flip_base({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                try:
                    True if faults or self.finished else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                reply = self.draw_port25(tool, args)
                budget -= 1
                try:
                    True if reply is None else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                observation = reply.get('observation')
                self.observe(tool, observation)
                cat_key = self.push_port()
                found = [i for i, _ in self.walk_mark69(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.tune_stash(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.step_port94(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.tune_stash(i) or 0) <= cap)]
                home = self.bind_zone94(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                try:
                    True if rows else (_ for _ in ()).throw(_Branch())
                    opened = self.build_index([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    try:
                        True if opened else (_ for _ in ()).throw(_Branch())
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    except _Branch:
                        pass
                    break
                except _Branch:
                    pass
        try:
            True if not seen else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Halo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def bind_zone94(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.push_port()
        try:
            True if not cat_key else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        item = self.items[number - 1]
        _seq43 = list((self.full, self.rows))
        _ix44 = 0
        while _ix44 < len(_seq43):
            pool = _seq43[_ix44]
            _ix44 = _ix44 + 1
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.owns(item, label):
                    counts[_words(label)] = counts.get(_words(label), 0) + 1
            for name in (item.shelf, item.name):
                try:
                    True if _words(name) in counts else (_ for _ in ()).throw(_Branch())
                    return _words(name)
                except _Branch:
                    pass
            try:
                True if counts else (_ for _ in ()).throw(_Branch())
                return max(counts, key=lambda label: counts[label])
            except _Branch:
                pass
        return ''

    def push_heap(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        _seq45 = list(sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))))
        _ix46 = 0
        while _ix46 < len(_seq45):
            number = _seq45[_ix46]
            _ix46 = _ix46 + 1
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            try:
                True if left else (_ for _ in ()).throw(_Branch())
                alone = self.msg_limit <= FIRST_ALONE and (not any(self.lane_spent.values()))
                return (number, left[:LIST_SIZE if self.listing and (not alone) else 1])
            except _Branch:
                pass
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = _flip_ring94(label)
        return any((_drive_delta50(mine, _flip_ring94(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        try:
            True if len(labels) >= 2 else (_ for _ in ()).throw(_Branch())
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.cast_rail(number)}?"
        except _Branch:
            pass
        return self.question(number, labels[0], self.step_rail(number, labels[0]) if shown else None)

    def open_link56(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.draw_cell69(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def open_delta(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            try:
                True if self.follow_up else (_ for _ in ()).throw(_Branch())
                number, labels, shown = self.follow_up.pop(0)
            except _Branch:
                number, labels = self.push_heap(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                shown = False
            if not labels:
                return []
            fitted, text = self.open_link56(number, labels, shown)
            try:
                True if not fitted and shown else (_ for _ in ()).throw(_Branch())
                fitted, text = self.open_link56(number, labels, False)
            except _Branch:
                pass
            try:
                True if fitted else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.wrap_node([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        try:
            True if len(labels) > 1 else (_ for _ in ()).throw(_Branch())
            heard.pop('free', None)
        except _Branch:
            pass
        self.take_heard(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            try:
                True if len(item.told) > size else (_ for _ in ()).throw(_Branch())
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
            except _Branch:
                pass
        crowded = any((_snap_cell(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
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
            if new or not crowded:
                return
            if number not in self.one_product and self.open_unit(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.cast_rail(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            else:
                try:
                    True if not shown and len(self.step_rail(number, labels[0])) >= 2 else (_ for _ in ()).throw(_Branch())
                    self.follow_up.insert(0, (number, labels, True))
                except _Branch:
                    pass
            return
        try:
            True if crowded and (not new) else (_ for _ in ()).throw(_Branch())
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        except _Branch:
            pass
        hit = self.answered_label(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = rest + ahead if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            seen.setdefault(_words(key), set()).update((w for v in values for w in _snap_delta(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(_flip_ring94(label) & _flip_ring94(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(_words(label), set()) & _snap_delta(value)) for label in labels]
            try:
                True if max(scores, default=0) > 0 else (_ for _ in ()).throw(_Branch())
                hits.append(scores.index(max(scores)))
            except _Branch:
                pass
        return min(hits) if hits else -1

    def open_token73(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        try:
            True if len(self.items) == 1 and (not rows) else (_ for _ in ()).throw(_Branch())
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        except _Branch:
            pass
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_drive_delta50(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def check_scope15(self) -> None:
        cat_key = self.push_port()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        try:
            True if strays and len(orphans) == 1 else (_ for _ in ()).throw(_Branch())
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})
        except _Branch:
            pass

    def owner_of(self, label: str) -> int:
        try:
            True if not label else (_ for _ in ()).throw(_Branch())
            return 0
        except _Branch:
            pass
        try:
            True if label in self.adopted else (_ for _ in ()).throw(_Branch())
            return self.adopted[label]
        except _Branch:
            pass
        words = _words(label)
        long = _snap_delta(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            try:
                True if words == _words(item.shelf or '') else (_ for _ in ()).throw(_Branch())
                score = 3.0
            except _Branch:
                try:
                    True if words == _words(item.name) else (_ for _ in ()).throw(_Branch())
                    score = 2.0
                except _Branch:
                    mine = _snap_delta(item.shelf or '') | _snap_delta(item.name)
                    score = len(long & mine) / max(1, len(long | mine))
            try:
                True if score > top else (_ for _ in ()).throw(_Branch())
                best, top = (number, score)
            except _Branch:
                pass
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.push_port()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _step_core(record)))

    def fetch_heap(self, count: int) -> dict[int, list[str]]:
        self.check_scope15()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            try:
                True if label != last else (_ for _ in ()).throw(_Branch())
                listing.append(f'category {label!r}:')
                last = label
            except _Branch:
                pass
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.slot_cell(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        _seq47 = list(_items(plan.get('ask')))
        _ix48 = 0
        while _ix48 < len(_seq47):
            pick = _seq47[_ix48]
            _ix48 = _ix48 + 1
            try:
                True if not isinstance(pick, dict) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            number = min(max(1, _snap_cell(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            try:
                True if not label or len(label.split()) > 5 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            try:
                True if all((_words(label) != _words(other) for other in lanes[number])) else (_ for _ in ()).throw(_Branch())
                lanes[number].append(label)
            except _Branch:
                pass
            try:
                True if str(pick.get('hint') or '').strip() else (_ for _ in ()).throw(_Branch())
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
            except _Branch:
                pass
        for number in lanes:
            keys = self.open_token73(number, table)
            read = max(1, self.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                try:
                    True if _words(label) in own else (_ for _ in ()).throw(_Branch())
                    return own[_words(label)]
                except _Branch:
                    pass
                near = [s for k, s in own.items() if _drive_delta50(_stem(label), _stem(k)) or _drive_delta50(_stem(k), _stem(label))]
                return max(near) if near else 0.0
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                try:
                    True if all((_words(spelled(pick)) != _words(other) for other in lane)) else (_ for _ in ()).throw(_Branch())
                    lane.append(spelled(pick))
                except _Branch:
                    pass
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
                try:
                    True if any((_words(key) == _words(other) for other in lane[:3])) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                try:
                    True if weak is None and len(lane) >= 3 else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                placed = [other for other in lane if _words(other) != _words(key)]
                try:
                    True if weak is not None and len(self.items) > 1 else (_ for _ in ()).throw(_Branch())
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                except _Branch:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                try:
                    True if any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                lane = placed
            lanes[number] = lane
            _seq49 = list(keys)
            _ix50 = 0
            while _ix50 < len(_seq49):
                key = _seq49[_ix50]
                _ix50 = _ix50 + 1
                try:
                    True if all((_words(key) != _words(other) for other in lanes[number])) else (_ for _ in ()).throw(_Branch())
                    lanes[number].append(key)
                except _Branch:
                    pass
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def open_unit(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def cast_rail(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first)."""
        item = self.items[number - 1]
        parts = self.open_unit(number)
        try:
            True if not parts else (_ for _ in ()).throw(_Branch())
            return item.name
        except _Branch:
            pass
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.push_port()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))]
        named = [_snap_delta(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _snap_delta(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.cast_rail(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def step_rail(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            try:
                True if name == want else (_ for _ in ()).throw(_Branch())
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
            except _Branch:
                pass
        return []

    def wrap_node(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.draw_cell69(text)
            try:
                True if not args else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.draw_port25(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.drive_grid94()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            try:
                True if isinstance(answer, dict) and answer.get('error') else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        _seq51 = list(_items(heard.get('needs')))
        _ix52 = 0
        while _ix52 < len(_seq51):
            entry = _seq51[_ix52]
            _ix52 = _ix52 + 1
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _snap_cell(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            try:
                True if need not in self.items[number - 1].told else (_ for _ in ()).throw(_Branch())
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            except _Branch:
                pass
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _snap_cell(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _snap_cell(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.slot_cell(prompt, list(models), 2500) or {}

    def fold_base48(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        try:
            True if not steps or self.finished or (not self.fold_base83(stage)) else (_ for _ in ()).throw(_Branch())
            return (True, '')
        except _Branch:
            pass
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.step_pool() + '\n\nCART NOW: ' + self.lift_cell() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.slot_cell(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            try:
                True if told or hold else (_ for _ in ()).throw(_Branch())
                self.version += 1
                return (False, ('Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            except _Branch:
                pass
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_snap_cell(n) or 0) in proof for n in _items(e.get('lines'))))]
            try:
                True if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2) else (_ for _ in ()).throw(_Branch())
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Halo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            except _Branch:
                pass
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.drive_grid94() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _flip_base(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.wrap_node([(1, '', text)], 1)
                try:
                    True if not batch else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.draw_port25(name, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.drive_grid94()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        try:
            True if isinstance(observation, str) else (_ for _ in ()).throw(_Branch())
            return observation
        except _Branch:
            pass
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            try:
                True if depth > 4 else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                _seq53 = list(node.values())
                _ix54 = 0
                while _ix54 < len(_seq53):
                    value = _seq53[_ix54]
                    _ix54 = _ix54 + 1
                    walk(value, depth + 1)
            elif isinstance(node, list):
                _seq55 = list(node)
                _ix56 = 0
                while _ix56 < len(_seq55):
                    value = _seq55[_ix56]
                    _ix56 = _ix56 + 1
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.bind_edge(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                else:
                    try:
                        True if isinstance(value, dict) and (not self.bind_edge(value, False, 1)) else (_ for _ in ()).throw(_Branch())
                        _seq57 = list(value.values())
                        _ix58 = 0
                        while _ix58 < len(_seq57):
                            inner = _seq57[_ix58]
                            _ix58 = _ix58 + 1
                            try:
                                True if isinstance(inner, str) and len(inner.split()) >= 6 else (_ for _ in ()).throw(_Branch())
                                lines.append(inner)
                            except _Branch:
                                pass
                    except _Branch:
                        pass
        return [line for line in lines if line not in self.heard]

    def build_grid81(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.step_pool() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.slot_cell(prompt, [LIGHT, SPARE], 2500) or {}
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
            item = self.items[min(max(1, _snap_cell(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            try:
                True if not hit else (_ for _ in ()).throw(_Branch())
                hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])][:1]
            except _Branch:
                pass
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        _seq59 = list(adds)
        _ix60 = 0
        while _ix60 < len(_seq59):
            entry = _seq59[_ix60]
            _ix60 = _ix60 + 1
            item = self.items[min(max(1, _snap_cell(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            try:
                True if need not in item.musts + item.told else (_ for _ in ()).throw(_Branch())
                item.told.append(need)
            except _Branch:
                pass
            added.add(need)
            changed.append(f'added {need}')
        try:
            True if gone else (_ for _ in ()).throw(_Branch())
            changed.append('unavailable: ' + ', '.join(gone))
        except _Branch:
            pass
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
            _seq61 = list(item.musts + item.told)
            _ix62 = 0
            while _ix62 < len(_seq61):
                need = _seq61[_ix62]
                _ix62 = _ix62 + 1
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def wrap_port(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            try:
                True if depth > 6 else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            if isinstance(node, dict):
                ident = self.bind_edge(node, False, depth)
                try:
                    True if ident and _step_core(node) and (ident not in ids) else (_ for _ in ()).throw(_Branch())
                    ids.append(ident)
                    return
                except _Branch:
                    pass
                _seq63 = list(node.values())
                _ix64 = 0
                while _ix64 < len(_seq63):
                    value = _seq63[_ix64]
                    _ix64 = _ix64 + 1
                    walk(value, depth + 1)
            else:
                try:
                    True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                    for value in node:
                        walk(value, depth + 1)
                except _Branch:
                    pass
        walk(observation, 0)
        needs = self.needs_now()
        try:
            True if not ids or not needs else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        many = len(self.items) > 1
        lines = []
        _seq65 = list(ids[:12])
        _ix66 = 0
        while _ix66 < len(_seq65):
            ident = _seq65[_ix66]
            _ix66 = _ix66 + 1
            block = _step_core(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _drive_delta50(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            try:
                True if parts else (_ for _ in ()).throw(_Branch())
                lines.append(f'{ident}: ' + '; '.join(parts))
            except _Branch:
                pass
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def lift_cell(self) -> str:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            return _dump(_clip(self.cart_view, 300))[:3000]
        except _Branch:
            pass
        return "lines by Halo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_snap_cell(v) for k, v in node.items() if not isinstance(v, str) and _snap_cell(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        except _Branch:
            pass
        return self.cart_lines.get(ident, 1)

    def walk_pool(self) -> list[str]:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        except _Branch:
            pass
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        try:
            True if self.brief.get('pick') == 'cheapest' else (_ for _ in ()).throw(_Branch())
            order.sort(key=lambda i: (i not in self.full, self.tune_stash(i) if self.tune_stash(i) is not None else float('inf')))
        except _Branch:
            pass
        for ident in order:
            try:
                True if ident in skip or self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            try:
                True if len(lines) >= limit else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
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

    def tune_edge(self, number: int) -> list[str]:
        cat_key = self.push_port()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def step_grid(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.walk_pool()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.tune_edge(main)]
            parts = [i for i in cart if i in self.tune_edge(part)]
            try:
                True if len(devices) != 1 or not parts else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.fit_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.tune_edge(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fold_flag9(short=True)))
        return faults

    def fold_flag9(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        _seq71 = list(sorted({m for _, m in self.bound}))
        _ix72 = 0
        while _ix72 < len(_seq71):
            main = _seq71[_ix72]
            _ix72 = _ix72 + 1
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            _seq69 = list(parts)
            _ix70 = 0
            while _ix70 < len(_seq69):
                p = _seq69[_ix70]
                _ix70 = _ix70 + 1
                for i in self.tune_edge(p):
                    try:
                        True if self.sold_out(i) else (_ for _ in ()).throw(_Branch())
                        continue
                    except _Branch:
                        pass
                    _seq67 = list(self.fit_set(i, index))
                    _ix68 = 0
                    while _ix68 < len(_seq67):
                        code = _seq67[_ix68]
                        _ix68 = _ix68 + 1
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.tune_edge(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def build_grid64(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.push_port()
        ids = [i for i in self.walk_pool() if (self.held_units(i) or 0) > 0]
        try:
            True if not cat_key or len(ids) >= len(self.items) else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        block = _step_core(self.full.get(ident) or {}) or {}
        for owner, need in self.needs_now():
            try:
                True if owner != number else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            prop, _, value = need.partition(':')
            split = lambda text: set(re.sub('[,/;|]', ' ', _words(text)).split())
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            if not any((_drive_delta50(_flip_ring94(prop), _flip_ring94(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        _seq75 = list(self.build_grid64())
        _ix76 = 0
        while _ix76 < len(_seq75):
            number = _seq75[_ix76]
            _ix76 = _ix76 + 1
            item = self.items[number - 1]
            fits = [i for i in self.tune_edge(number) if i in self.full and i not in self.cart_lines and self.step_port94(i) and (self.tune_stash(i) is not None) and self.states_needs(i, number)]
            fits.sort(key=lambda i: self.tune_stash(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.budget, (int, float)):
                held = sum(((self.tune_stash(i) or 0.0) * (self.held_units(i) or 0) for i in self.walk_pool()))
                cost = (self.tune_stash(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                    pick = None
            calls = self.fetch_grid({**{i: self.cart_lines[i] for i in self.walk_pool() if i in self.cart_lines}, pick: item.qty}, []) if pick else None
            try:
                True if not calls or len(calls) + 1 > self.left() else (_ for _ in ()).throw(_Branch())
                return False
            except _Branch:
                pass
            _seq73 = list(calls)
            _ix74 = 0
            while _ix74 < len(_seq73):
                args = _seq73[_ix74]
                _ix74 = _ix74 + 1
                faults: list[str] = []
                args = _flip_base(args, self.schemas.get(cart), 'args', faults)
                if faults or self.draw_port25(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.push_port()
        mine = _snap_delta(self.items[number - 1].name)
        try:
            True if len(mine) < 2 or mine <= _snap_delta(label) or (not cat_key) or (self.items[number - 1].shelf and _snap_delta(label) == _snap_delta(self.items[number - 1].shelf)) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _snap_delta(seen)), '')

    def build_port(self) -> list[str]:
        self.check_scope15()
        cat_key = self.push_port()
        ids = [i for i in self.walk_pool() if (self.held_units(i) or 0) > 0]
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
            else:
                try:
                    True if len([i for i in lines if owner[i] == [n]]) > 1 else (_ for _ in ()).throw(_Branch())
                    faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
                except _Branch:
                    pass
            _seq77 = list(lines)
            _ix78 = 0
            while _ix78 < len(_seq77):
                i = _seq77[_ix78]
                _ix78 = _ix78 + 1
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
        try:
            True if len(known) == len(self.items) else (_ for _ in ()).throw(_Branch())
            for i in ids:
                try:
                    True if not owner[i] else (_ for _ in ()).throw(_Branch())
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
                except _Branch:
                    pass
        except _Branch:
            pass
        _seq79 = list(ids)
        _ix80 = 0
        while _ix80 < len(_seq79):
            i = _seq79[_ix80]
            _ix80 = _ix80 + 1
            if self.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.tune_stash(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                try:
                    True if self.budget + 1e-09 < total <= 10 * self.budget else (_ for _ in ()).throw(_Branch())
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
                except _Branch:
                    pass
        return faults

    def step_stash(self) -> list[tuple[str, int, int]]:
        cat_key = self.push_port()
        try:
            True if not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        out = []
        for ident in self.walk_pool():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.walk_pool() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
            try:
                True if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.items[owners[0] - 1].qty) and (not self.sold_out(ident)) else (_ for _ in ()).throw(_Branch())
                out.append((ident, have, self.items[owners[0] - 1].qty))
            except _Branch:
                pass
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        try:
            True if not name or not target or (not count) else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cast_rail65(name):
                calls.append({target: ident, count: want})
            else:
                try:
                    True if mode else (_ for _ in ()).throw(_Branch())
                    options = [str(o) for o in props[mode]['enum']]
                    off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                    on = next((o for o in options if o != off), options[0])
                    calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
                except _Branch:
                    return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        _seq81 = list(calls)
        _ix82 = 0
        while _ix82 < len(_seq81):
            args = _seq81[_ix82]
            _ix82 = _ix82 + 1
            faults: list[str] = []
            args = _flip_base(args, schema, 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.draw_port25(name, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.note_cart(name, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def draw_edge(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.walk_pool()
        key = self.check_key(ids)
        try:
            True if key in self.checked or not self.judges(limit) or (not ids) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.step_pool()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.lift_cell()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        try:
            True if self.brief.get('pick') in ('cheapest', 'most', 'least') else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        except _Branch:
            pass
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and (not lead) and (not self.audited) and (self.brief.get('pick') not in ('cheapest', 'most', 'least')):
            got = self.slot_cell(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        try:
            True if judges and (not self.no_credit) else (_ for _ in ()).throw(_Branch())
            got = self.slot_cell(prompt, judges, 3000) or got
        except _Branch:
            pass
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        try:
            True if not got or got.get('ok') is not False else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        try:
            True if better else (_ for _ in ()).throw(_Branch())
            text += '. Listings that look better: ' + ', '.join(better)
        except _Branch:
            pass
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def page_text(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        parts: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for inner in value.values():
                    walk(inner)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner)
        walk(record)
        return ' | '.join(parts)

    def store_rules(self) -> list[tuple[str, str, str]]:
        """Every store-wide pairing rule stated on a page or row seen so far (see store_pairings)."""
        rules: list[tuple[str, str, str]] = []
        for ident in set(self.full) | set(self.rows):
            for rule in store_pairings(self.page_text(ident)):
                if rule not in rules:
                    rules.append(rule)
        return rules

    def partner_of(self, prop: str, value: str, page: str) -> tuple[str, str]:
        """The value the store pairs with `value` for `prop`, and where that came from; ('', '') when nothing seen
        says. The last page's statement about its own listing comes first (one naming the property, else its only
        statement), then a store-wide rule for the property (or one naming no property) whose left side is the value,
        then one whose right side is; rules stated on the last page itself come before rules seen elsewhere, since two
        categories can each have a rule for a property of the same name."""
        wanted = _snap_delta(prop)
        same = lambda a, b: _words(a) == _words(b) or bool(_snap_delta(a)) and (_snap_delta(a) <= _snap_delta(b) or _snap_delta(b) <= _snap_delta(a))
        own = listing_pairings(self.page_text(page)) if page else []
        named = [w for p, _, w in own if _snap_delta(p) & wanted]
        if named or len(own) == 1:
            return ((named or [own[0][2]])[0], f'the page {page} says its {prop} goes with it')
        local = store_pairings(self.page_text(page)) if page else []
        rules = [r for r in local + [r for r in self.store_rules() if r not in local] if not r[0] or _snap_delta(r[0]) & wanted]
        forward = [b for _, a, b in rules if same(a, value)]
        if forward:
            return (forward[0], f'the store rule pairs {value} with {forward[0]}')
        backward = [a for _, a, b in rules if same(b, value)]
        if backward:
            return (backward[0], f'the store rule pairs {backward[0]} with {value}')
        return ('', '')

    def apply_pairing(self, need: str, page: str) -> None:
        """For a requirement asking for whatever goes with the pointed-at value, replace the value read from the chain's
        last page by its partner. Applied only when the read value appears on that page, so a value the reader already
        converted is not paired a second time."""
        value = self.ref_values.get(need, '')
        if not wants_partner(need) or not value or need in self.paired:
            return
        if page:
            own_words = ' '.join((s for s in _sentences(self.page_text(page)) if not re.search(PAIRS_WITH + r'|\brule\b', s, re.I)))
            if _words(value) not in _words(own_words):
                return
        prop = need.split(':', 1)[0]
        partner, why = self.partner_of(prop, value, page)
        if not partner:
            self.trace.append({'kind': 'pairing_unknown', 'need': need[:120], 'value': value})
            return
        self.paired[need] = (value, partner)
        self.ref_values[need] = f'{partner} ({why}; the requirement asks for what goes with {value}, so the wanted {prop} is {partner})'
        self.version += 1
        self.trace.append({'kind': 'paired', 'need': need[:120], 'read': value, 'wanted': partner, 'why': why})

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.push_unit34(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            _seq83 = list(reversed(self.seen_order))
            _ix84 = 0
            while _ix84 < len(_seq83):
                ident = _seq83[_ix84]
                _ix84 = _ix84 + 1
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            _seq85 = list(reversed(self.seen_order))
            _ix86 = 0
            while _ix86 < len(_seq85):
                ident = _seq85[_ix86]
                _ix86 = _ix86 + 1
                if len(records) >= 60:
                    break
                try:
                    True if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)) else (_ for _ in ()).throw(_Branch())
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
                except _Branch:
                    pass
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.slot_cell(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                try:
                    True if not isinstance(chain, dict) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                try:
                    True if not need else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                try:
                    True if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip() else (_ for _ in ()).throw(_Branch())
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    self.apply_pairing(need, ids[-1])
                    continue
                except _Branch:
                    pass
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    self.apply_pairing(need, ids[-1])
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                try:
                    True if closed else (_ for _ in ()).throw(_Branch())
                    reach += [i for i in closed if i not in reach]
                except _Branch:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                try:
                    True if self.trace and self.trace[-1].get('kind') == 'link_known' else (_ for _ in ()).throw(_Branch())
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
                except _Branch:
                    pass
            try:
                True if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            _seq87 = list(reach[:max(0, min(3, 6 - len(seen)))])
            _ix88 = 0
            while _ix88 < len(_seq87):
                ident = _seq87[_ix88]
                _ix88 = _ix88 + 1
                faults: list[str] = []
                args = _flip_base({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.draw_port25(tool, args)
                try:
                    True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Halo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        try:
            True if hold else (_ for _ in ()).throw(_Branch())
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        except _Branch:
            pass
        return (text, hold)

    def check_ring(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            try:
                True if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)) else (_ for _ in ()).throw(_Branch())
                out.append(ident)
            except _Branch:
                pass
        return out

    def open_token(self, about: str) -> list[str]:
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
        named = {w for ident in self.check_ring(about, list(self.rows)) + self.open_token(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        _seq89 = list(self.open_token(about))
        _ix90 = 0
        while _ix90 < len(_seq89):
            ident = _seq89[_ix90]
            _ix90 = _ix90 + 1
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
        plan = self.walk_line()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.check_ring(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _flip_base({**base, field: words}, self.schemas.get(tool), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                self.link_searches += 1
                reply = self.draw_port25(tool, args)
                try:
                    True if reply is not None else (_ for _ in ()).throw(_Branch())
                    observation = reply.get('observation')
                    self.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.walk_mark69(observation)]
                    known = self.check_ring(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
                except _Branch:
                    pass
            except _Branch:
                pass
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.check_ring(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.push_unit34(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.walk_pool() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _flip_base({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            before = dict(self.rows.get(ident) or {})
            reply = self.draw_port25(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            try:
                True if moved else (_ for _ in ()).throw(_Branch())
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
            except _Branch:
                pass
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        try:
            True if lines else (_ for _ in ()).throw(_Branch())
            self.version += 1
        except _Branch:
            pass
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def cast_stash(self) -> str:
        lost = [i for i in self.walk_pool() if i not in self.gone and self.sold_out(i)]
        try:
            True if not lost else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        _seq91 = list(lost)
        _ix92 = 0
        while _ix92 < len(_seq91):
            ident = _seq91[_ix92]
            _ix92 = _ix92 + 1
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.open_index86(lost)
        text = f"[Halo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def drive_core81(self) -> str:
        ids = [i for i in self.walk_pool() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        try:
            True if len(ids) == 1 else (_ for _ in ()).throw(_Branch())
            self.unavailable.add(ids[0])
            return ''
        except _Branch:
            pass
        return self.reread_offers() if ids else ''

    def open_index86(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        records = []
        for ident in reversed(self.seen_order):
            try:
                True if ident in lost or ident in self.gone or self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.step_pool()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.lift_cell()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.slot_cell(prompt, self.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.walk_pool()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def snap_zone(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.cast_phase23()
        named = self.snap_index(got)
        try:
            True if not named else (_ for _ in ()).throw(_Branch())
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.shut_delta(rows):
                got = self.cast_phase23()
                named = self.snap_index(got)
        except _Branch:
            pass
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def cast_phase23(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            try:
                True if self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            try:
                True if len(records) >= 30 else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        try:
            True if not records else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.step_pool()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        judges = self.judges()
        got = self.slot_cell(prompt, [LIGHT] if judges else [], 4000)
        try:
            True if HEAVY in judges and (not self.no_credit) and (not self.snap_index(got)) else (_ for _ in ()).throw(_Branch())
            got = self.slot_cell(prompt, [HEAVY], 4000) or got
        except _Branch:
            pass
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def snap_index(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def shut_delta(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.wrap_base()
        target = self.push_unit34(tool)
        try:
            True if many and joiner and (not self.compare_bare()) else (_ for _ in ()).throw(_Branch())
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        except _Branch:
            try:
                True if tool and target else (_ for _ in ()).throw(_Branch())
                plan = [(tool, {target: i}) for i in rows[:2]]
            except _Branch:
                return False
        try:
            True if self.left() < len(plan) + 5 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _flip_base(args, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.draw_port25(name, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def fetch_grid(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                try:
                    True if not id_key else (_ for _ in ()).throw(_Branch())
                    return None
                except _Branch:
                    pass
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        try:
            True if not target else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cast_rail65(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.walk_pool() else 0
            try:
                True if have == want else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            try:
                True if have and (not self.cast_rail65(name)) else (_ for _ in ()).throw(_Branch())
                return None
            except _Branch:
                pass
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        try:
            True if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.completed = True
        cat_key = self.push_port()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else 1 if len(self.items) == 1 else None
            pick = next((i for i in ids if not self.sold_out(i)), None)
            try:
                True if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}) else (_ for _ in ()).throw(_Branch())
                plan[pick] = self.items[index - 1].qty
            except _Branch:
                pass
        try:
            True if self.confirmed not in plan else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        now = self.walk_pool()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        try:
            True if len(plan) + len({kind(i) for i in keep}) < len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        lines = {**keep, **plan}
        calls = self.fetch_grid(lines, drop)
        try:
            True if not calls or len(calls) + 1 > self.left() else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.build_index(list(lines), len(calls) + 3)
        self.show_links(list(lines), len(calls) + 3)
        self.draw_edge(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ''
        done = []
        _seq93 = list(calls)
        _ix94 = 0
        while _ix94 < len(_seq93):
            args = _seq93[_ix94]
            _ix94 = _ix94 + 1
            faults: list[str] = []
            args = _flip_base(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.draw_port25(cart, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                return ''
            except _Branch:
                pass
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        faults = []
        args = _flip_base(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        _ri95 = 0
        while _ri95 < 2:
            attempt = _ri95
            _ri95 = _ri95 + 1
            if attempt:
                self.reread_offers()
                self.draw_edge(list(lines), SPEND_HARD)
                try:
                    True if not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)) else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
            go, said = self.fold_base48('order')
            try:
                True if said else (_ for _ in ()).throw(_Branch())
                replies.append(said)
            except _Branch:
                pass
            try:
                True if not go else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.draw_port25(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            try:
                True if self.finished or reply.get('finished') else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        if not replies:
            return ''
        return 'Not run (Halo acted instead): the independent check confirmed ' + self.confirmed + ', so Halo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def step_mark5(self, name: str, args: dict[str, Any]) -> bool:
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
        try:
            True if keep is not None and ends else (_ for _ in ()).throw(_Branch())
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        except _Branch:
            pass
        return tools

    def step_port(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            try:
                True if value is None and key not in (schema.get('required') or []) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def step_index(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            True if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key else (_ for _ in ()).throw(_Branch())
            return (args, '')
        except _Branch:
            pass
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            try:
                True if not isinstance(value, str) or _draw_flag(value, True) or len(value.strip()) < 8 else (_ for _ in ()).throw(_Branch())
                return (value, '')
            except _Branch:
                pass
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            try:
                True if len(hits) == 1 else (_ for _ in ()).throw(_Branch())
                return (hits[0], '')
            except _Branch:
                pass
            try:
                True if hits else (_ for _ in ()).throw(_Branch())
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            except _Branch:
                pass
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        for key in free:
            try:
                True if key not in args else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            try:
                True if problem else (_ for _ in ()).throw(_Branch())
                return (args, problem)
            except _Branch:
                pass
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cast_rail65(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def slot_base(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and (count not in args) else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_snap_cell(v) for v in args.values() if _snap_cell(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            try:
                True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
                shape = (spec.get(key) or {}).get('items') or {}
                try:
                    True if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape) else (_ for _ in ()).throw(_Branch())
                    whole = True
                except _Branch:
                    pass
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_snap_cell(v) for v in line.values() if _snap_cell(v) is not None and (not isinstance(v, str))), 1)
                        try:
                            True if ident else (_ for _ in ()).throw(_Branch())
                            lines.append((ident, count))
                        except _Branch:
                            pass
                    else:
                        try:
                            True if isinstance(line, str) and line in self.rows else (_ for _ in ()).throw(_Branch())
                            lines.append((line, loose_count))
                        except _Branch:
                            pass
            except _Branch:
                try:
                    True if isinstance(value, str) and value in self.rows else (_ for _ in ()).throw(_Branch())
                    lines.append((value, loose_count))
                except _Branch:
                    pass
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cast_rail65(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - (count or 1) if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
            else:
                try:
                    True if sets else (_ for _ in ()).throw(_Branch())
                    have = count if count is not None else 1
                except _Branch:
                    have = have + (count if count is not None else 1)
            try:
                True if have > 0 else (_ for _ in ()).throw(_Branch())
                self.cart_lines[ident] = have
            except _Branch:
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
        try:
            True if not target or any((k != target for k in schema.get('required') or [])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return target

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        try:
            True if isinstance(spec.get('maxItems'), int) else (_ for _ in ()).throw(_Branch())
            return spec['maxItems']
        except _Branch:
            pass
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def shut_unit(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.push_unit34(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        try:
            True if self.compare_bare() else (_ for _ in ()).throw(_Branch())
            return calls
        except _Branch:
            pass
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            try:
                True if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone) else (_ for _ in ()).throw(_Branch())
                picks.append((index, ident))
            except _Branch:
                pass
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            try:
                True if index == first else (_ for _ in ()).throw(_Branch())
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: [i for _, i in picks]})}})
            except _Branch:
                try:
                    True if index in {i for i, _ in picks} else (_ for _ in ()).throw(_Branch())
                    out.append({'id': call['id'], 'joined': True})
                except _Branch:
                    out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def drive_line82(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.shut_unit(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            try:
                True if call.get('joined') else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Shown in the side-by-side result above: Halo opened these listings together in one step.'))
                continue
            except _Branch:
                pass
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Halo's interviewer handles every message to the shopper."))
                continue
            try:
                True if name not in self.schemas else (_ for _ in ()).throw(_Branch())
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            except _Branch:
                pass
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or '').strip() else {}
                try:
                    True if args is None else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
                except _Branch:
                    pass
            faults: list[str] = []
            args = _flip_base(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.step_index(name, args)
            try:
                True if wrong else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            except _Branch:
                pass
            try:
                True if name == self.roles.get('cart') else (_ for _ in ()).throw(_Branch())
                args = self.slot_base(name, args)
            except _Branch:
                pass
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Halo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.push_unit34(name)
            try:
                True if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            except _Branch:
                pass
            try:
                True if name == self.roles.get('cart') and any(((_snap_cell(v) or 0) < 0 for v in args.values() if not isinstance(v, str))) else (_ for _ in ()).throw(_Branch())
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                try:
                    True if absent else (_ for _ in ()).throw(_Branch())
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.lift_cell()))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            ending = self.step_mark5(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            try:
                True if name in readers and cached and (cached[0] == self.version) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            except _Branch:
                pass
            try:
                True if target and name in readers and (name != self.roles.get('compare')) and self.fetch_hook(name, str(args.get(target))) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            except _Branch:
                pass
            try:
                True if many and isinstance(args.get(many), list) and args[many] else (_ for _ in ()).throw(_Branch())
                keep = [i for i in args[many] if not self.fetch_hook(name, str(i))]
                try:
                    True if not keep else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                except _Branch:
                    pass
                try:
                    True if len(keep) < len(args[many]) else (_ for _ in ()).throw(_Branch())
                    trimmed += f"\n[Halo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if ending and self.left() > 2 else (_ for _ in ()).throw(_Branch())
                if self.step_mark5(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.lift_cell()))
                        continue
                    reread = self.reread_offers()
                    try:
                        True if self.note_sold else (_ for _ in ()).throw(_Branch())
                        self.drive_core81()
                    except _Branch:
                        pass
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
                        faults = []
                        view_args = _flip_base({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.draw_port25(self.roles['cart_view'], view_args) if not faults else None
                        try:
                            True if view is not None else (_ for _ in ()).throw(_Branch())
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.lift_cell() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                        except _Branch:
                            pass
                    swap = self.cast_stash()
                    held = [i for i in self.walk_pool() if i in self.gone] if self.held < 2 else []
                    try:
                        True if swap or held else (_ for _ in ()).throw(_Branch())
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.lift_cell()))
                        continue
                    except _Branch:
                        pass
                    chain, hold = self.trail()
                    try:
                        True if hold else (_ for _ in ()).throw(_Branch())
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.lift_cell()))
                        continue
                    except _Branch:
                        pass
                    misfit = self.step_grid() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    try:
                        True if misfit else (_ for _ in ()).throw(_Branch())
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.lift_cell()))
                        continue
                    except _Branch:
                        pass
                    faults = self.build_port() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.step_stash())
                        faults = self.build_port() if fixed else faults
                        try:
                            True if fixed and (not faults) else (_ for _ in ()).throw(_Branch())
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.lift_cell()))
                            continue
                        except _Branch:
                            pass
                    try:
                        True if faults else (_ for _ in ()).throw(_Branch())
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.lift_cell()))
                        continue
                    except _Branch:
                        pass
                    lines = self.walk_pool()
                    read = self.build_index(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.draw_sigma()
                    verdict = self.draw_edge()
                    try:
                        True if verdict else (_ for _ in ()).throw(_Branch())
                        seen = "Halo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.lift_cell()))
                        continue
                    except _Branch:
                        pass
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Halo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                else:
                    try:
                        True if self.left() > 5 else (_ for _ in ()).throw(_Branch())
                        review = self.snap_zone()
                        if review:
                            out.append((cid, 'Not run yet (no step used). ' + review))
                            continue
                        try:
                            True if self.leads and self.left() > 4 and (self.refuse_blocks == 0) else (_ for _ in ()).throw(_Branch())
                            self.refuse_blocks = 1
                            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                            out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                            continue
                        except _Branch:
                            pass
                        if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.draw_edge([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
                    except _Branch:
                        pass
            except _Branch:
                pass
            try:
                True if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held) else (_ for _ in ()).throw(_Branch())
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.walk_pool() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            except _Branch:
                pass
            try:
                True if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) else (_ for _ in ()).throw(_Branch())
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            except _Branch:
                pass
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.step_port(name, args)
            try:
                True if self.step_mark5(name, args) else (_ for _ in ()).throw(_Branch())
                late = self.late_question()
                try:
                    True if late else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.lift_cell()))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if ending and self.left() >= 2 else (_ for _ in ()).throw(_Branch())
                go, said = self.fold_base48('order' if self.step_mark5(name, args) else 'end')
                try:
                    True if not go else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.lift_cell()))
                    continue
                except _Branch:
                    pass
                if said:
                    trimmed = '\n[Halo] ' + said + trimmed
            except _Branch:
                pass
            reply = self.draw_port25(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            try:
                True if many and isinstance(args.get(many), list) and (len(args[many]) > 1) else (_ for _ in ()).throw(_Branch())
                shown = _dump(observation)
                try:
                    True if not any((str(i) in shown for i in args[many])) else (_ for _ in ()).throw(_Branch())
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Halo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                except _Branch:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            except _Branch:
                pass
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.absorb(observation)
            try:
                True if name == self.roles.get('compare') else (_ for _ in ()).throw(_Branch())
                rows = self.walk_mark69(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_step_core(r) for _, r in rows))
            except _Branch:
                pass
            try:
                True if name in readers - {self.roles.get('search')} else (_ for _ in ()).throw(_Branch())
                self.note_view(name, observation)
            except _Branch:
                pass
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.push_unit34(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.bind_edge(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_step_core(observation))
                    try:
                        True if name == self.roles.get('open') and (not self.facts_from[name]) else (_ for _ in ()).throw(_Branch())
                        other = next((t for t in self.schemas if t != name and self.push_unit34(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        try:
                            True if other else (_ for _ in ()).throw(_Branch())
                            self.roles['open'] = other
                        except _Branch:
                            pass
                    except _Branch:
                        pass
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    try:
                        True if name == self.offer_tool() else (_ for _ in ()).throw(_Branch())
                        self.offer_seen[listing] = self.step
                    except _Branch:
                        pass
            try:
                True if name == self.roles.get('cart') else (_ for _ in ()).throw(_Branch())
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
                counts = [_snap_cell(v) for v in args.values() if not isinstance(v, str) and _snap_cell(v) is not None]
                _seq96 = list(args.values())
                _ix97 = 0
                while _ix97 < len(_seq96):
                    value = _seq96[_ix97]
                    _ix97 = _ix97 + 1
                    try:
                        True if counts and isinstance(value, str) and (value in self.rows) else (_ for _ in ()).throw(_Branch())
                        try:
                            True if counts[0] <= 0 and (not _flip_sigma21(observation, value)) else (_ for _ in ()).throw(_Branch())
                            self.dropped.add(value)
                        except _Branch:
                            try:
                                True if counts[0] > 0 else (_ for _ in ()).throw(_Branch())
                                self.dropped.discard(value)
                            except _Branch:
                                pass
                    except _Branch:
                        pass
                lines = [i for i in self.walk_pool() if not self.sold_out(i) and i not in self.dropped]
                try:
                    True if len(lines) >= len(self.built) else (_ for _ in ()).throw(_Branch())
                    self.built = lines
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if name == self.roles.get('search') and self.walk_mark69(observation) else (_ for _ in ()).throw(_Branch())
                self.searched = (self.searched + [dict(args)])[-8:]
            except _Branch:
                pass
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            try:
                True if self.step_mark5(name, args) else (_ for _ in ()).throw(_Branch())
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            except _Branch:
                pass
            text = self.show(observation) + paged + trimmed
            try:
                True if name not in plain and (not ending) else (_ for _ in ()).throw(_Branch())
                text += self.slot_flag16([observation] + self.paged_obs) + self.snap_orbit84(name, args, observation)
            except _Branch:
                pass
            base = text
            proof = self.wrap_port(observation)
            try:
                True if proof else (_ for _ in ()).throw(_Branch())
                text += '\n' + proof
            except _Branch:
                pass
            if name == self.roles.get('cart') and any((_snap_cell(v) == 0 for v in args.values() if not isinstance(v, str))):
                try:
                    True if any((isinstance(v, str) and v in self.rows and _flip_sigma21(observation, v) for v in args.values())) else (_ for _ in ()).throw(_Branch())
                    text += '\n[Halo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
                except _Branch:
                    pass
            self.note_sold = False
            lines = self.loose_lines(observation)
            try:
                True if lines else (_ for _ in ()).throw(_Branch())
                meaning = self.build_grid81(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            except _Branch:
                pass
            refused = self.step_mark5(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.step_mark5(name, args))))
            try:
                True if (loose or again) and self.walk_pool() and (not self.finished) else (_ for _ in ()).throw(_Branch())
                seen = self.drive_core81()
                try:
                    True if seen else (_ for _ in ()).throw(_Branch())
                    text += "\n[Halo re-read the cart's lines after the refusal]\n" + seen
                except _Branch:
                    pass
            except _Branch:
                pass
            self.note_sold = False
            swap = self.cast_stash()
            if swap:
                text += '\n' + swap
            try:
                True if self.queued else (_ for _ in ()).throw(_Branch())
                text += "\n[notes seen on Halo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            except _Branch:
                pass
            try:
                True if name in readers else (_ for _ in ()).throw(_Branch())
                self.read_cache[mark] = (self.version, self.step, base)
            except _Branch:
                pass
            try:
                True if reply.get('finished') else (_ for _ in ()).throw(_Branch())
                text += ' [episode finished]'
            except _Branch:
                pass
            out.append((cid, text))
        return out

    def shut_link14(self) -> bool:
        refuse = self.roles.get('refuse')
        try:
            True if not refuse else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        faults: list[str] = []
        _flip_base(self.step_port(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def flip_ring(self, again: bool=False) -> str:
        try:
            True if self.finished or self.broke or self.left() <= 0 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.walk_pool()) if fresh else bool(self.cart_lines)
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.shut_link14()
            try:
                True if whole and len(self.walk_pool()) >= 2 else (_ for _ in ()).throw(_Branch())
                self.build_index(self.walk_pool(), self.tail())
                if self.left() >= 3 and self.open_index() > 5:
                    self.show_links(self.walk_pool(), self.tail())
            except _Branch:
                pass
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.walk_pool()})
        try:
            True if self.finished or self.left() <= 0 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))) else (_ for _ in ()).throw(_Branch())
            faults: list[str] = []
            args = _flip_base(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                if self.left() >= 2:
                    self.fold_base48('order')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.draw_port25(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ''
                try:
                    True if reply is not None and again else (_ for _ in ()).throw(_Branch())
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.show(reply.get('observation'))
                except _Branch:
                    pass
            except _Branch:
                pass
        except _Branch:
            pass
        if refuse:
            faults = []
            args = _flip_base(self.step_port(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                if self.left() >= 2:
                    self.fold_base48('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.draw_port25(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
            except _Branch:
                pass
        return ''

    def trim(self) -> None:
        size = sum((len(_dump(m.get('content'))) for m in self.chat))
        try:
            True if size <= HISTORY_LIMIT else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        _seq98 = list(tools[:max(1, len(tools) // 2)])
        _ix99 = 0
        while _ix99 < len(_seq98):
            index = _seq98[_ix99]
            _ix99 = _ix99 + 1
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        try:
            True if self.finished else (_ for _ in ()).throw(_Branch())
            return self.trace
        except _Branch:
            pass
        self.draw_cell()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        try:
            True if opening is not None else (_ for _ in ()).throw(_Branch())
            self.absorb(opening)
        except _Branch:
            pass
        intro = "SHOPPER'S REQUEST:\n" + self.request
        try:
            True if opening is not None else (_ for _ in ()).throw(_Branch())
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        except _Branch:
            pass
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.shut_token40()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.shut_token(LOOP_CALLS, SPEND_HARD)
        try:
            True if not self.finished else (_ for _ in ()).throw(_Branch())
            turned = self.flip_ring(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Halo] Halo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.shut_token40()})
                self.shut_token(self.loop_calls + 10, SPEND_HARD + 0.01)
            try:
                True if not self.finished else (_ for _ in ()).throw(_Branch())
                self.flip_ring()
            except _Branch:
                pass
        except _Branch:
            pass
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def shut_token(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.wrap_sigma14() < LAST_TURN or (not self.walk_pool() and self.open_index() > EMPTY_ROOM)):
            try:
                True if self.shut_token38() else (_ for _ in ()).throw(_Branch())
                summary = self.build_sigma85()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.shut_token40()})
                continue
            except _Branch:
                pass
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
                try:
                    True if not isinstance(call, dict) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            try:
                True if not calls else (_ for _ in ()).throw(_Branch())
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.shut_token40()})
                if idle >= 3:
                    break
                continue
            except _Branch:
                pass
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            try:
                True if not self.asked and self.roles.get('ask') and (self.fetch_grid87() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.wrap_sigma14() <= ASK_BY) else (_ for _ in ()).throw(_Branch())
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                _seq100 = list(calls)
                _ix101 = 0
                while _ix101 < len(_seq100):
                    call = _seq100[_ix101]
                    _ix101 = _ix101 + 1
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.build_sigma85()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.shut_token40()})
                continue
            except _Branch:
                pass
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.drive_line82(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.shut_token40()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Halo(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.flip_ring()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']