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
PREFIX_KEYS = 3
JOINT_LABELS = 3
FIRST_READ_BY = 100.0
FIRST_READ_WAIT = 10.0
ASK_WORDS = 'message|ask|question|shopper|customer|buyer|user|clarif|chat|talk'
DEFAULT_MESSAGES = 3
LATE_BY_UNASKED = 225.0
REPLANS = 2
SWEEP_PAGES = 6
SWEEP_ROOM = 8
SWEEP_SEARCHES = 4
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
GUIDE = 'You are Halo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Halo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Halo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Halo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Halo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Halo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Halo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "unclear": [numbers of the questions whose answer says the shopper did not follow or understand it, or asks for it again or in other words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.'
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

def _trim_signal(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _take_piece(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _trim_signal(hit.group(1))
    _seq1 = list(re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I))
    _ix2 = 0
    while _ix2 < len(_seq1):
        hit = _seq1[_ix2]
        _ix2 = _ix2 + 1
        codes += _trim_signal(hit.group(1))
    works: list[str] = []
    _seq3 = list(re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I))
    _ix4 = 0
    while _ix4 < len(_seq3):
        hit = _seq3[_ix4]
        _ix4 = _ix4 + 1
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _trim_signal(hit.group(1))
    also = [(hit.group(1).upper(), _trim_signal(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

def _trim_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _do_slot(value: Any, limit: int, depth: int=0) -> Any:
    try:
        True if depth > 7 else (_ for _ in ()).throw(_Branch())
        return '...'
    except _Branch:
        pass
    try:
        True if isinstance(value, str) else (_ for _ in ()).throw(_Branch())
        return value if len(value) <= limit else value[:limit] + '...'
    except _Branch:
        pass
    if isinstance(value, dict):
        return {k: _do_slot(v, limit, depth + 1) for k, v in value.items()}
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        return [_do_slot(v, limit, depth + 1) for v in value]
    except _Branch:
        pass
    return value

def _seek_sketch94(text: Any) -> dict[str, Any] | None:
    try:
        True if not isinstance(text, str) else (_ for _ in ()).throw(_Branch())
        return None
    except _Branch:
        pass
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    try:
        True if fence else (_ for _ in ()).throw(_Branch())
        text = fence.group(1)
    except _Branch:
        pass
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

def _note_trace58(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    return _note_trace58(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _note_trace58(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _note_trace58(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        try:
            True if len(w) >= 3 and w not in ('and', 'for', 'the', 'with') else (_ for _ in ()).throw(_Branch())
            out.add(w)
        except _Branch:
            pass
    return out

def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _note_trace58(text).split()))

def _pack_bundle13(prop: frozenset[str], key: frozenset[str]) -> bool:
    try:
        True if not prop or not key else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _read_sketch(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    try:
        True if isinstance(kind, str) else (_ for _ in ()).throw(_Branch())
        return [kind]
    except _Branch:
        pass
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    try:
        True if kind == 'string' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, str)
    except _Branch:
        pass
    try:
        True if kind == 'integer' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, int) and (not isinstance(value, bool))
    except _Branch:
        pass
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
    try:
        True if kind == 'array' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, list)
    except _Branch:
        pass
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

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    try:
        True if not isinstance(schema, dict) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
    kinds = _read_sketch(schema)
    try:
        True if kinds and (not any((_fits_kind(value, k) for k in kinds))) else (_ for _ in ()).throw(_Branch())
        for kind in kinds:
            try:
                True if kind == 'integer' and isinstance(value, float) and value.is_integer() else (_ for _ in ()).throw(_Branch())
                value = int(value)
                break
            except _Branch:
                pass
            try:
                True if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value) else (_ for _ in ()).throw(_Branch())
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            except _Branch:
                pass
            try:
                True if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)) else (_ for _ in ()).throw(_Branch())
                value = str(value)
                break
            except _Branch:
                pass
            try:
                True if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')) else (_ for _ in ()).throw(_Branch())
                value = value.strip().lower() == 'true'
                break
            except _Branch:
                pass
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
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    except _Branch:
        pass
    if isinstance(value, list):
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
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
        except _Branch:
            pass
    try:
        True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        _seq5 = list(schema.get('required') or [])
        _ix6 = 0
        while _ix6 < len(_seq5):
            name = _seq5[_ix6]
            _ix6 = _ix6 + 1
            try:
                True if name not in value else (_ for _ in ()).throw(_Branch())
                faults.append(f'{where}.{name} is required')
            except _Branch:
                pass
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    except _Branch:
        pass
    return value

def _seek_gate59(value: Any) -> list[Any]:
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _note_piece(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        True if isinstance(value, int) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
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

def _send_row(value: Any, named: bool=False) -> bool:
    try:
        True if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip() else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    _seq7 = list(rows[0])
    _ix8 = 0
    while _ix8 < len(_seq7):
        key = _seq7[_ix8]
        _ix8 = _ix8 + 1
        values = [row.get(key) for row in rows]
        try:
            True if not all((_send_row(v, 'id' in str(key).lower()) for v in values)) else (_ for _ in ()).throw(_Branch())
            continue
        except _Branch:
            pass
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        try:
            True if best is None or score > best[0] else (_ for _ in ()).throw(_Branch())
            best = (score, key)
        except _Branch:
            pass
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    try:
        True if depth > 5 else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    try:
        True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    except _Branch:
        pass
    return False

def _pull_key95(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    try:
        True if seconds >= 50 else (_ for _ in ()).throw(_Branch())
        return _llm
    except _Branch:
        pass
    limit = max(5, int(seconds) // 5 * 5)
    try:
        True if limit not in _llm_short else (_ for _ in ()).throw(_Branch())
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    except _Branch:
        pass
    return _llm_short[limit]

def _make_probe(node: Any) -> dict[str, Any] | None:
    try:
        True if not isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
        return None
    except _Branch:
        pass
    best = None
    for value in node.values():
        try:
            True if isinstance(value, dict) and len(value) >= 2 else (_ for _ in ()).throw(_Branch())
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
        except _Branch:
            pass
    return best

TEXT_SEPARATORS = (' | ', ' || ', '\n', '\t', ' ; ', '; ')
TEXT_PAIRS = (': ', ' = ', '=')
TEXT_NAME = "[A-Za-z_][A-Za-z0-9_ ./&'()-]{0,38}"
TEXT_MIN_FIELDS = 3
BLOCK_NAME_MAX = 60
BLOCK_PART_MAX = 200
FLAT_MIN_KEYS = 5
FLAT_VALUE_MAX = 60

def _plain_scalar(text: str) -> Any:
    """A field's text as the value a structured result would hold: true, false, none or a number; else the text."""
    bare = text.strip()
    low = bare.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null'):
        return None
    if re.fullmatch('-?\\d{1,15}', bare):
        return int(bare)
    if re.fullmatch('-?\\d{1,12}\\.\\d{1,6}', bare):
        return float(bare)
    return bare

def _named_part(part: str, pair: str) -> tuple[str, str] | None:
    name, found, value = part.partition(pair)
    name = name.strip()
    if not found or not re.fullmatch(TEXT_NAME, name) or len(name.split()) > 4:
        return None
    return (name, value.strip())

def _cut_fields(text: str, between: str, pair: str) -> tuple[list[tuple[str, str]], int]:
    """The named fields of a text cut at `between`, with how many parts carried a name. A part without a name belongs
    to the value before it (a title may hold the separator). A text that does not open with a name gives no fields."""
    fields: list[tuple[str, str]] = []
    named = 0
    for part in text.split(between):
        if not part.strip():
            continue
        found = _named_part(part, pair)
        if found and found[0] not in (name for name, _ in fields):
            fields.append(found)
            named += 1
        elif fields:
            fields[-1] = (fields[-1][0], fields[-1][1] + between + part.strip())
        else:
            return ([], 0)
    return (fields, named)

def _block_of_pairs(text: str) -> dict[str, str] | None:
    """A listing's attributes written inside one field: a JSON object, "a=b; c=d" or short "a: b; c: d" parts. Prose stays
    prose, whatever colons or equal signs it holds: in the colon form every part must be short and named, and in the
    equals form most parts must be, the others continuing the value before them."""
    bare = text.strip()
    if len(bare) < 5:
        return None
    if bare[0] == '{' and bare[-1] == '}':
        try:
            parsed = json.loads(bare)
        except (ValueError, RecursionError):
            return None
        return parsed if isinstance(parsed, dict) else None
    if '=' in bare:
        out: dict[str, str] = {}
        last = ''
        parts = [p.strip() for p in re.split(';\\s+|;$|,\\s+(?=[^=,;]{1,60}=)', bare) if p.strip()]
        for part in parts:
            name, found, value = part.partition('=')
            name = name.strip()
            if found and 0 < len(name) <= BLOCK_NAME_MAX and name not in out:
                last = name
                out[name] = value.strip()
            elif last:
                out[last] = f'{out[last]}; {part}'
            else:
                return None
        return out if len(out) >= 2 and len(out) * 2 >= len(parts) else None
    parts = [p.strip() for p in bare.split('; ') if p.strip()]
    if len(parts) >= 2 and ': ' in bare:
        pairs = [_named_part(p, ': ') for p in parts]
        if all(pairs) and all(len(pair[1]) <= FLAT_VALUE_MAX for pair in pairs):
            return {pair[0]: pair[1] for pair in pairs}
    return None

def _record_of_text(text: str) -> dict[str, Any] | None:
    """The record a text spells field by field, whichever separator and pair mark this store uses, or None. It counts as a
    record only when one field holds a listing code: a store's note written with colons stays a note."""
    if text[-1:] in (':', '='):
        text += ' '  # the last field is empty ("... | description:"): its name still ends the field before it
    for between in TEXT_SEPARATORS:
        if between not in text:
            continue
        best: list[tuple[str, str]] = []
        for pair in TEXT_PAIRS:
            fields, named = _cut_fields(text, between, pair)
            if named >= TEXT_MIN_FIELDS and named > len(best):
                best = fields
        if best:
            record: dict[str, Any] = {}
            for name, value in best:
                block = _block_of_pairs(value)
                record[name] = block if block else value if 'id' in name.lower() else _plain_scalar(value)
            if any(_send_row(value, 'id' in name.lower()) for name, value in best if ' ' not in value.strip()):
                return record
            return None
    return None

def _results_of_text(text: str) -> Any:
    """The structured form of a text a store tool returned: one record, several records a line each or in blocks, or the
    JSON it holds; None when it holds none of these."""
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in '{[':
        try:
            return _unspell(json.loads(bare))
        except (ValueError, RecursionError):
            pass
    for parts in (re.split('\\n\\s*\\n', bare), bare.split('\n')):
        if len(parts) >= 2:
            rows = [_record_of_text(part) for part in parts if part.strip()]
            good = [row for row in rows if row]
            if len(good) >= 2 and len(good) * 5 >= len(rows) * 4:
                return good
    return _record_of_text(bare)

def _pair_list(rows: list[Any]) -> dict[str, Any] | None:
    """Attributes given as a list: objects of a name and a value, or [name, value] lists."""
    if len(rows) < 2:
        return None
    out: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, dict) and len(row) == 2:
            name, value = row.values()
            if not re.search('name|key|label|attr|prop|field|spec|title', str(next(iter(row))), re.I):
                return None
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            name, value = row
        else:
            return None
        if not isinstance(name, str) or not 0 < len(name) <= BLOCK_NAME_MAX or isinstance(value, (dict, list)):
            return None
        out[name.strip()] = value
    return out

def _unspell(node: Any, depth: int=0) -> Any:
    """A store result in the one form the agent reads: rows as objects and a listing's attributes as a nested object, when
    the store spells them as text. Race 18's store returned every listing as "id: p-1 | title: ... | attributes_map: a=b; c=d";
    the agent read no row from it, and its question planner had no attribute key to ask about (Mars v2 asked labels it
    made up, and on 56 of 180 episodes had too few of them to spend its messages). A structured result passes unchanged."""
    if depth > 6:
        return node
    if isinstance(node, str):
        found = _results_of_text(node)
        return found if found is not None else node
    if isinstance(node, list):
        return [_unspell(value, depth + 1) for value in node]
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if isinstance(value, list):
                pairs = _pair_list(value)
                out[key] = pairs if pairs and len(node) >= TEXT_MIN_FIELDS else _unspell(value, depth + 1)
            elif isinstance(value, str) and len(node) >= TEXT_MIN_FIELDS:
                block = _block_of_pairs(value) if '=' in value and len(value) <= 3000 and re.fullmatch('[^=:]{1,60}=[^=]*(?:;\\s+[^=:]{1,60}=[^=]*)+', value.strip()) else None
                out[key] = block if block else value
            else:
                out[key] = _unspell(value, depth + 1)
        if len(out) == 1 and isinstance(next(iter(out.values())), dict) and isinstance(next(iter(node.values())), str):
            return next(iter(out.values()))
        return out
    return node

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
        steps = _note_piece(self.view.get('max_steps'))
        try:
            True if steps is None and isinstance(self.view.get('max_steps'), float) else (_ for _ in ()).throw(_Branch())
            steps = int(self.view['max_steps'])
        except _Branch:
            pass
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
        self.list_unclear: set[str] = set()
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
        self.plan_full = -1
        self.replans = 0
        self.puzzled: set[int] = set()
        self.search_log: list[tuple[dict[str, Any], Any]] = []
        self.swept = False

    def make_log28(self) -> int:
        return self.max_steps - self.step

    def seek_frame(self) -> float:
        return time.monotonic() - self.started

    def pick_layer(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.seek_frame()

    def view_chunk(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _trim_log(sorted(self.cart_lines.items()))

    def mark_anchor(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.pack_gauge(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.pack_gauge(tool) else None

    def view_key(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.view_key()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def scan_set(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.scan_set('order') else 1

    def sort_slot(self, limit: float=SPEND_SOFT) -> list[str]:
        try:
            True if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD else (_ for _ in ()).throw(_Branch())
            return [HEAVY, LIGHT]
        except _Branch:
            pass
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def run_tag(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        _seq9 = list(models)
        _ix10 = 0
        while _ix10 < len(_seq9):
            model = _seq9[_ix10]
            _ix10 = _ix10 + 1
            try:
                True if self.calls >= ALL_CALLS else (_ for _ in ()).throw(_Branch())
                return None
            except _Branch:
                pass
            try:
                True if self.pick_layer() < CALL_FLOOR else (_ for _ in ()).throw(_Branch())
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.seek_frame(), 1)})
                return None
            except _Branch:
                pass
            self.calls += 1
            tries = 0
            while True:
                message, status = self.get_frame(model, messages, max_tokens, tools)
                try:
                    True if message is not None else (_ for _ in ()).throw(_Branch())
                    return message
                except _Branch:
                    pass
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                try:
                    True if self.pick_layer() - WAITS[tries] * 1.2 < CALL_FLOOR else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
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

    def get_frame(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            _seq11 = list(marks)
            _ix12 = 0
            while _ix12 < len(_seq11):
                index = _seq11[_ix12]
                _ix12 = _ix12 + 1
                content = sent[index].get('content')
                try:
                    True if isinstance(content, str) and content else (_ for _ in ()).throw(_Branch())
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
                except _Branch:
                    pass
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        try:
            True if tools else (_ for _ in ()).throw(_Branch())
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        except _Branch:
            pass
        result = _pull_key95(self.pick_layer()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
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
        error = result.error or {}
        status = error.get('status')
        try:
            True if isinstance(data, dict) and isinstance(data.get('error'), dict) else (_ for _ in ()).throw(_Branch())
            status = _note_piece(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        except _Branch:
            pass
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def sort_gauge(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        _seq13 = list(models)
        _ix14 = 0
        while _ix14 < len(_seq13):
            model = _seq13[_ix14]
            _ix14 = _ix14 + 1
            message = self.run_tag([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _seek_sketch94((message or {}).get('content'))
            try:
                True if got is not None else (_ for _ in ()).throw(_Branch())
                return got
            except _Branch:
                pass
            try:
                True if self.broke or self.no_credit or self.calls >= ALL_CALLS else (_ for _ in ()).throw(_Branch())
                return None
            except _Branch:
                pass
        return None

    def fill_row80(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        _ri17 = 0
        while _ri17 < 4:
            _ = _ri17
            _ri17 = _ri17 + 1
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
                if name != self.roles.get('ask'):
                    data['observation'] = self.gather_flat(name, _unspell(data.get('observation')))
                seen = data.get('observation')
                if name == self.roles.get('search'):
                    self.search_log = (self.search_log + [(dict(args), seen)])[-16:]
                try:
                    True if name in self.mark_anchor() and (not (isinstance(seen, dict) and seen.get('error'))) else (_ for _ in ()).throw(_Branch())
                    _seq15 = list(args.values())
                    _ix16 = 0
                    while _ix16 < len(_seq15):
                        value = _seq15[_ix16]
                        _ix16 = _ix16 + 1
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                except _Branch:
                    pass
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
                try:
                    True if found else (_ for _ in ()).throw(_Branch())
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                except _Branch:
                    pass
                try:
                    True if 'reuse' in low or 'different action' in low else (_ for _ in ()).throw(_Branch())
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                except _Branch:
                    pass
                return None
            except _Branch:
                pass
            if status in (400, 401, 402, 403, 404, 422):
                try:
                    True if status == 402 else (_ for _ in ()).throw(_Branch())
                    self.broke = True
                except _Branch:
                    pass
                return None
        return None

    def find_ask_tool(self) -> None:
        """The message tool when the brief named none: a tool no other job took whose one free-text parameter carries the
        message, by its own name and description. A message the store does not allow is refused at the cost of one step."""
        if self.roles.get('ask'):
            return
        taken = set(self.roles.values())
        for tool in self.tools:
            fn = tool['function']
            name = fn['name']
            props = (self.schemas.get(name) or {}).get('properties') or {}
            texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and 'enum' not in v]
            if name in taken or len(texts) != 1 or len(props) > 2 or re.search(QUERY_WORDS, _snake(texts[0]), re.I):
                continue
            if re.search(ASK_WORDS, _snake(name) + ' ' + str(fn.get('description') or ''), re.I):
                self.roles['ask'] = name
                self.trace.append({'kind': 'ask_tool_found', 'tool': name})
                return

    def limit_in_request(self) -> int | None:
        """The number of messages the request itself allows, when it says so in figures or small number words."""
        counted = '(\\d{1,2}|one|two|three|four|five|six)\\s+(?:\\w+\\s+){0,2}?(?:messages?|questions?)\\b'
        found = re.search(counted, self.request, re.I)
        if not found:
            return None
        word = found.group(1).lower()
        return int(word) if word.isdigit() else ('one', 'two', 'three', 'four', 'five', 'six').index(word) + 1

    def gather_flat(self, name: str, observation: Any) -> Any:
        """On a listing's page, attributes spread among the record's own keys are moved into one object: the short plain
        values under keys the search rows do not have, when there are enough of them to be attributes and not a note."""
        if name not in (self.roles.get('open'), self.roles.get('compare')):
            return observation
        row_keys: set[str] = set()
        for ident, row in self.rows.items():
            if ident not in self.opened:
                row_keys.update(row)
        if not row_keys:
            return observation
        if isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, list):
                    observation[key] = [self.fold_record(entry, row_keys) for entry in value]
                else:
                    observation[key] = self.fold_record(value, row_keys)
            return observation
        return self.fold_record(observation, row_keys)

    def fold_record(self, record: Any, row_keys: set[str]) -> Any:
        if not isinstance(record, dict) or _make_probe(record) or not self.ident(record, True):
            return record
        folded = {}
        for key, value in record.items():
            if key not in row_keys and not isinstance(value, bool) and isinstance(value, (str, int, float)) and len(str(value)) <= FLAT_VALUE_MAX:
                folded[key] = value
        if len(folded) < FLAT_MIN_KEYS:
            return record
        kept = {key: value for key, value in record.items() if key not in folded}
        kept['attributes'] = folded
        return kept

    def keep_row61(self) -> None:
        lines = []
        _seq18 = list(self.tools)
        _ix19 = 0
        while _ix19 < len(_seq18):
            tool = _seq18[_ix19]
            _ix19 = _ix19 + 1
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_trim_log(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        try:
            True if self.view.get('opening') is not None else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _trim_log(_do_slot(self.view.get('opening'), 1500))
        except _Branch:
            pass
        brief = self.sort_gauge(prompt, [LIGHT], 4000) or {}
        while not brief and not self.no_credit and self.calls < ALL_CALLS and self.seek_frame() < FIRST_READ_BY:
            time.sleep(FIRST_READ_WAIT)
            brief = self.sort_gauge(prompt, [LIGHT, HEAVY], 4000) or {}
        try:
            True if not self.no_credit and (not _seek_gate59(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))) else (_ for _ in ()).throw(_Branch())
            brief = self.sort_gauge(prompt, [HEAVY, LIGHT], 2500) or brief
        except _Branch:
            try:
                True if not self.no_credit and self.pack_layer(brief) else (_ for _ in ()).throw(_Branch())
                again = self.sort_gauge(prompt, [HEAVY], 2500) or {}
                try:
                    True if _seek_gate59(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy') else (_ for _ in ()).throw(_Branch())
                    brief = again
                except _Branch:
                    pass
            except _Branch:
                pass
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        _seq20 = list(('buy_args', 'refuse_args'))
        _ix21 = 0
        while _ix21 < len(_seq20):
            key = _seq20[_ix21]
            _ix21 = _ix21 + 1
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        try:
            True if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']) else (_ for _ in ()).throw(_Branch())
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _trim_log(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.sort_gauge(ask, [LIGHT, HEAVY], 1500) or {}
            try:
                True if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']) else (_ for _ in ()).throw(_Branch())
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
            except _Branch:
                pass
        except _Branch:
            pass
        _seq22 = list(_seek_gate59(brief.get('items')))
        _ix23 = 0
        while _ix23 < len(_seq22):
            entry = _seq22[_ix23]
            _ix23 = _ix23 + 1
            if not isinstance(entry, dict):
                continue
            quantity = _note_piece(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _seek_gate59(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _note_trace58(m.shelf) == _note_trace58(item.shelf) and (not any((_note_trace58(a.split(':')[0]) == _note_trace58(b.split(':')[0]) and _note_trace58(a.partition(':')[2]) != _note_trace58(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            try:
                True if twin is None else (_ for _ in ()).throw(_Branch())
                merged.append(item)
                became[old] = len(merged)
            except _Branch:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        try:
            True if not self.items else (_ for _ in ()).throw(_Branch())
            self.items = [Item('the requested product', '', 1, [])]
        except _Branch:
            pass
        _seq24 = list(_seek_gate59(brief.get('fits')))
        _ix25 = 0
        while _ix25 < len(_seq24):
            pair = _seq24[_ix25]
            _ix25 = _ix25 + 1
            if isinstance(pair, dict):
                part, main = (became.get(_note_piece(pair.get('item')) or 0), became.get(_note_piece(pair.get('with')) or 0))
                try:
                    True if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound) else (_ for _ in ()).throw(_Branch())
                    self.bound.append((part, main))
                except _Branch:
                    pass
        _seq26 = list(_seek_gate59(brief.get('lookups')))
        _ix27 = 0
        while _ix27 < len(_seq26):
            lookup = _seq26[_ix27]
            _ix27 = _ix27 + 1
            text = str(lookup).strip()
            try:
                True if not any((text in item.musts for item in self.items)) else (_ for _ in ()).throw(_Branch())
                key = _note_trace58(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _note_trace58(m.split(':')[0]) == key), text)
            except _Branch:
                pass
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            try:
                True if text and (number, text) not in self.refs else (_ for _ in ()).throw(_Branch())
                self.refs.append((number, text))
            except _Branch:
                pass
        self.make_piece()
        budget = brief.get('budget')
        try:
            True if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget) else (_ for _ in ()).throw(_Branch())
            budget = float(budget)
        except _Branch:
            pass
        try:
            True if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) else (_ for _ in ()).throw(_Branch())
            self.budget = budget
        except _Branch:
            pass
        self.find_ask_tool()
        limit = _note_piece(brief.get('messages'))
        said_limit = self.limit_in_request()
        if limit is not None and limit > 0:
            self.msg_limit = limit
        elif said_limit is not None:
            self.msg_limit = said_limit
        elif limit == 0 and (not self.roles.get('ask') or re.search("\\b(?:do not|don't|never|no|without)\\s+(?:\\w+\\s+){0,2}?(?:messag|ask|question|contact)", self.request, re.I)):
            self.msg_limit = 0
        elif self.roles.get('ask'):
            self.msg_limit = DEFAULT_MESSAGES
        steps = _note_piece(brief.get('steps'))
        try:
            True if steps is not None and 10 <= steps < self.max_steps else (_ for _ in ()).throw(_Branch())
            self.max_steps = steps
        except _Branch:
            pass
        _seq28 = list(_seek_gate59(brief.get('process')))
        _ix29 = 0
        while _ix29 < len(_seq28):
            entry = _seq28[_ix29]
            _ix29 = _ix29 + 1
            try:
                True if isinstance(entry, dict) and str(entry.get('step') or '').strip() else (_ for _ in ()).throw(_Branch())
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _note_piece(entry.get('messages')) or 0)})
            except _Branch:
                pass
        try:
            True if self.roles.get('ask') else (_ for _ in ()).throw(_Branch())
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0
        except _Branch:
            pass

    @staticmethod
    def pack_layer(brief: dict[str, Any]) -> bool:
        shelves = [_note_trace58(str(e.get('shelf') or '')) for e in _seek_gate59(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def make_piece(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            try:
                True if not pointers or not rest else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.sort_gauge(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _seek_gate59(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def do_frame(self, number: int, need: str) -> bool:
        try:
            True if need in [m for _, m in self.refs] or need in [str(x) for x in _seek_gate59(self.brief.get('lookups'))] else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        prop, _, value = need.partition(':')
        try:
            True if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        try:
            True if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        item = self.items[number - 1]
        try:
            True if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        cat_key = self.sort_chunk89()
        keys: set[str] = set()
        read = 0
        _seq30 = list(self.full.values())
        _ix31 = 0
        while _ix31 < len(_seq30):
            record = _seq30[_ix31]
            _ix31 = _ix31 + 1
            try:
                True if cat_key and (not self.send_frame(item, str(record.get(cat_key) or ''))) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            block = _make_probe(record) or {}
            try:
                True if block else (_ for _ in ()).throw(_Branch())
                read += 1
                keys |= {_note_trace58(k) for k in block}
            except _Branch:
                pass
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def fill_signal(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.do_frame(number, m)]
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            try:
                True if item.shelf else (_ for _ in ()).throw(_Branch())
                head += f' (store label: {item.shelf})'
            except _Branch:
                pass
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            try:
                True if told else (_ for _ in ()).throw(_Branch())
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            except _Branch:
                pass
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
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
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        _seq32 = list(_seek_gate59(brief.get('wishes')))
        _ix33 = 0
        while _ix33 < len(_seq32):
            wish = _seq32[_ix33]
            _ix33 = _ix33 + 1
            parts.append(f'wish (tie-breaker only): {wish}')
        _seq34 = list(_seek_gate59(brief.get('rules')))
        _ix35 = 0
        while _ix35 < len(_seq34):
            rule = _seq34[_ix35]
            _ix35 = _ix35 + 1
            parts.append(f'rule from the request: {rule}')
        try:
            True if brief.get('quiet_decline') is True else (_ for _ in ()).throw(_Branch())
            parts.append('if nothing fits: end without naming or suggesting any listing')
        except _Branch:
            pass
        _seq36 = list(self.process)
        _ix37 = 0
        while _ix37 < len(_seq36):
            entry = _seq36[_ix37]
            _ix37 = _ix37 + 1
            when = {'choose': 'before choosing', 'order': 'at the order (Halo does it when you check out)', 'end': 'at an ending without a purchase (Halo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        try:
            True if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')) else (_ for _ in ()).throw(_Branch())
            bare = sorted((t for t in self.mark_anchor() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.mark_anchor() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        except _Branch:
            pass
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.make_log28()
        text = f'[Halo] step {self.step} of {self.max_steps} used, {left} left.'
        try:
            True if left <= self.tail() + 1 else (_ for _ in ()).throw(_Branch())
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        except _Branch:
            try:
                True if left <= 6 or (left <= len(self.items) + 5 and self.buys == 0) else (_ for _ in ()).throw(_Branch())
                text += ' Few steps left: set the cart to the chosen lines and check out now.'
            except _Branch:
                try:
                    True if self.buys == 0 and left <= 12 else (_ for _ in ()).throw(_Branch())
                    text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
                except _Branch:
                    try:
                        True if self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2) else (_ for _ in ()).throw(_Branch())
                        text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
                    except _Branch:
                        pass
        try:
            True if self.seek_frame() >= WIND_DOWN and left > self.tail() + 1 else (_ for _ in ()).throw(_Branch())
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        except _Branch:
            pass
        live = {need for _, need in self.get_slot81()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        try:
            True if lookups and self.buys == 0 else (_ for _ in ()).throw(_Branch())
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        except _Branch:
            pass
        try:
            True if self.leads else (_ for _ in ()).throw(_Branch())
            text += '\n[review candidates] ' + ', '.join(self.leads)
        except _Branch:
            pass
        plan = self.keep_set() if self.bound and self.buys == 0 else ''
        try:
            True if plan else (_ for _ in ()).throw(_Branch())
            text += '\n' + plan
        except _Branch:
            pass
        try:
            True if self.notes else (_ for _ in ()).throw(_Branch())
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        except _Branch:
            pass
        text += '\n[ledger]\n' + self.fill_signal()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        try:
            True if not isinstance(node, dict) or len(node) < 3 else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        try:
            True if self.id_key and _send_row(node.get(self.id_key), True) else (_ for _ in ()).throw(_Branch())
            return node[self.id_key].strip()
        except _Branch:
            pass
        nested = any((isinstance(v, dict) for v in node.values()))
        try:
            True if not (in_list or depth == 0 or nested) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        found = _learn_id([node])
        try:
            True if found and ('id' in str(found).lower() or node[found].strip() in self.rows) else (_ for _ in ()).throw(_Branch())
            return node[found].strip()
        except _Branch:
            pass
        return None

    def do_piece(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        try:
            True if not rows or len(rows) < len(node) / 2 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        try:
            True if self.id_key and all((_send_row(r.get(self.id_key), True) for r in rows)) else (_ for _ in ()).throw(_Branch())
            return True
        except _Branch:
            pass
        found = _learn_id(rows)
        try:
            True if found else (_ for _ in ()).throw(_Branch())
            self.id_key = found
            return True
        except _Branch:
            pass
        return False

    def get_probe(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            try:
                True if depth > 6 else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            try:
                True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
                ident = self.ident(node, in_list, depth)
                try:
                    True if ident else (_ for _ in ()).throw(_Branch())
                    try:
                        True if ident not in self.rows else (_ for _ in ()).throw(_Branch())
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    except _Branch:
                        pass
                    old = self.rows[ident]
                    try:
                        True if any((old.get(k) is True and v is False for k, v in node.items())) else (_ for _ in ()).throw(_Branch())
                        self.unavailable.add(ident)
                    except _Branch:
                        try:
                            True if any((old.get(k) is False and v is True for k, v in node.items())) else (_ for _ in ()).throw(_Branch())
                            self.unavailable.discard(ident)
                        except _Branch:
                            pass
                    self.rows[ident].update(node)
                    try:
                        True if _make_probe(node) else (_ for _ in ()).throw(_Branch())
                        self.full[ident] = dict(self.rows[ident])
                    except _Branch:
                        try:
                            True if ident in self.full else (_ for _ in ()).throw(_Branch())
                            self.full[ident].update(node)
                        except _Branch:
                            pass
                    return
                except _Branch:
                    pass
                for value in node.values():
                    walk(value, depth + 1)
            except _Branch:
                try:
                    True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                    rows = self.do_piece(node)
                    _seq38 = list(node)
                    _ix39 = 0
                    while _ix39 < len(_seq38):
                        value = _seq38[_ix39]
                        _ix39 = _ix39 + 1
                        walk(value, depth + 1, rows)
                except _Branch:
                    pass
        walk(observation, 0)

    def note_tip(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            try:
                True if depth > 6 or not isinstance(schema, dict) else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            _seq40 = list(schema.get('enum') or [])
            _ix41 = 0
            while _ix41 < len(_seq40):
                option = _seq40[_ix41]
                _ix41 = _ix41 + 1
                try:
                    True if isinstance(option, str) else (_ for _ in ()).throw(_Branch())
                    found.add(option.strip().lower())
                except _Branch:
                    pass
            _seq42 = list(schema.values())
            _ix43 = 0
            while _ix43 < len(_seq42):
                value = _seq42[_ix43]
                _ix43 = _ix43 + 1
                try:
                    True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
                    walk(value, depth + 1)
                except _Branch:
                    pass
        _seq44 = list(self.schemas.values())
        _ix45 = 0
        while _ix45 < len(_seq44):
            schema = _seq44[_ix45]
            _ix45 = _ix45 + 1
            walk(schema, 0)
        return found

    def sort_chunk89(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        _seq46 = list(self.rows.values())
        _ix47 = 0
        while _ix47 < len(_seq46):
            row = _seq46[_ix47]
            _ix47 = _ix47 + 1
            for key, value in row.items():
                try:
                    True if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)) else (_ for _ in ()).throw(_Branch())
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
                except _Branch:
                    pass
        total = max(1, len(self.rows))
        known = self.note_tip()
        best = None
        for key, values in counts.items():
            try:
                True if present[key] < 0.8 * total or len(values) > 40 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def send_trace(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.sort_chunk89()
        groups: dict[str, dict[str, Any]] = {}
        _seq48 = list(self.seen_order)
        _ix49 = 0
        while _ix49 < len(_seq48):
            ident = _seq48[_ix49]
            _ix49 = _ix49 + 1
            record = self.full.get(ident)
            try:
                True if not record else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            block = _make_probe(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _note_trace58(key)
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
        wanted = [(_note_trace58(item.shelf or ''), _note_trace58(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_note_trace58(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
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
                twin = next((spec for other, spec in kept if self.pack_roster13(other, spec, name, slot)), None)
                try:
                    True if twin is not None else (_ for _ in ()).throw(_Branch())
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                except _Branch:
                    pass
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def pack_gate(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        try:
            True if not number else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        _seq50 = list(self.items[number - 1].musts)
        _ix51 = 0
        while _ix51 < len(_seq50):
            must = _seq50[_ix51]
            _ix51 = _ix51 + 1
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _note_trace58(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def pack_roster13(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        try:
            True if not a or not b or (not a & b) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass

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

    def get_row90(self) -> list[list[str]]:
        cat_key = self.sort_chunk89()
        groups: dict[str, list[str]] = {}
        _seq52 = list(self.seen_order)
        _ix53 = 0
        while _ix53 < len(_seq52):
            ident = _seq52[_ix53]
            _ix53 = _ix53 + 1
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_note_trace58(item.shelf or ''), _note_trace58(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_note_trace58(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def pack_gauge(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        try:
            True if not tool or not target or any((k != target for k in schema.get('required') or [])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return target

    def send_log(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        try:
            True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        except _Branch:
            pass
        try:
            True if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value) else (_ for _ in ()).throw(_Branch())
            value = float(value)
        except _Branch:
            pass
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def take_ledger(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def note_ref(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        try:
            True if isinstance(observation, dict) else (_ for _ in ()).throw(_Branch())
            ident = self.ident(observation)
            try:
                True if ident else (_ for _ in ()).throw(_Branch())
                return [(ident, observation)]
            except _Branch:
                pass
            found: list[tuple[str, dict[str, Any]]] = []
            _seq54 = list(observation.values())
            _ix55 = 0
            while _ix55 < len(_seq54):
                value = _seq54[_ix55]
                _ix55 = _ix55 + 1
                try:
                    True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                except _Branch:
                    try:
                        True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
                        found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                    except _Branch:
                        pass
            return found
        except _Branch:
            pass
        try:
            True if isinstance(observation, list) else (_ for _ in ()).throw(_Branch())
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        except _Branch:
            pass
        return []

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.note_ref(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def make_bundle(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def hold_slot85(self, name: str, observation: Any) -> dict[str, Any] | None:
        try:
            True if name in self.shapes else (_ for _ in ()).throw(_Branch())
            return self.shapes[name]
        except _Branch:
            pass
        try:
            True if not isinstance(observation, dict) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        try:
            True if not lists else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _trim_log(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _trim_log(_do_slot(sample, 120))[:4000]
        got = self.sort_gauge(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        try:
            True if not rows else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
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

    def pull_chunk87(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.hold_slot85(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        try:
            True if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def trim_cue32(self, observation: Any) -> bool:
        prices = [self.send_log(i) for i, _ in self.note_ref(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.pull_chunk87(name, observation) if pick in ('cheapest', 'most', 'least') else None
        try:
            True if not plan else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        try:
            True if value is None or later or count <= shown else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if pick == 'cheapest' and self.trim_cue32(observation) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        pages = -(-(count - shown) // shown)
        try:
            True if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.make_log28() - self.tail() - 2 >= 12) else (_ for _ in ()).throw(_Branch())
            pages = 2
        except _Branch:
            pass
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.make_log28() - self.tail() - pages < 12:
            return f"\n[Halo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_trim_log(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.fill_row80(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.pull_tip(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.seek_probe36(seen)
            self.read_cache[name + _trim_log(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_trim_log(value)}: {shown_text}')
            after = self.pull_chunk87(name, seen)
            value = after[3] if after else None
            try:
                True if value is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Halo] {count} listings matched and Halo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Halo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def run_trace(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        try:
            True if not caps or not self.price_key else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.note_ref(observation) if (self.send_log(i) or 0) > cap]
        try:
            True if not over else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_trim_log(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Halo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def mark_sketch10(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.note_ref(seen)}
        rows = [(self.send_log(i), i) for i in found if self.take_ledger(i) and self.send_log(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Halo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def do_bundle(self) -> str:
        try:
            True if self.wider_held or self.brief.get('pick') != 'cheapest' or self.make_log28() < 14 + 2 * len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.wider_held = True
        lines = [f'{i} at {self.send_log(i):g}' for i in self.trim_bit() if self.send_log(i) is not None]
        return f"Not run yet (no step used). {self.make_log28()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def make_log73(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.pack_gauge(tool)
        try:
            True if not tool or not target else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        texts = []
        _seq56 = list([i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))])
        _ix57 = 0
        while _ix57 < len(_seq56):
            ident = _seq56[_ix57]
            _ix57 = _ix57 + 1
            try:
                True if self.make_log28() <= keep or self.finished or self.facts_from.get(tool) is False else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.fill_row80(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.pull_tip(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_make_probe(seen))
            texts.append(f'{ident}: ' + self.seek_probe36(seen)[:1500])
        try:
            True if texts else (_ for _ in ()).throw(_Branch())
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        except _Branch:
            pass
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.value_template()
        try:
            True if not plan or (len(self.items) < 2 and len(ids) < 2) else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        tool, base, field = plan
        _seq58 = list(ids)
        _ix59 = 0
        while _ix59 < len(_seq58):
            ident = _seq58[_ix59]
            _ix59 = _ix59 + 1
            marks = self.fill_pad()['marks']
            code = (marks.get(ident) or {}).get('more')
            try:
                True if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            try:
                True if self.make_log28() <= keep or self.finished else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.link_tried.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            reply = self.fill_row80(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.pull_tip(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.fill_pad()['marks']
            self.make_log73([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.tail() + 6))

    def view_signal(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_read_sketch(v)) and re.search(CAP_ARG, _snake(k), re.I) and (not re.search(CAP_NOT, k, re.I))]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v) and re.search(SHELF_ARG, _snake(k), re.I)]
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
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and (not any((k != shelves[0] and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())))), None)
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

    def pick_gauge13(self, cart: list[str]) -> None:
        plan = self.view_signal()
        cat_key = self.sort_chunk89()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        _seq60 = list(cart)
        _ix61 = 0
        while _ix61 < len(_seq60):
            line = _seq60[_ix61]
            _ix61 = _ix61 + 1
            price = self.send_log(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or (not label) or ((line, price) in self.capped):
                continue
            try:
                True if self.make_log28() <= self.tail() + 5 or self.finished else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.capped.add((line, price))
            faults: list[str] = []
            args = _shape({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            reply = self.fill_row80(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            self.pull_tip(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.pull_chunk87(tool, reply.get('observation'))
            if page and page[3] is not None and (page[1] < page[0] <= 2 * page[1]) and (self.make_log28() > self.tail() + 5) and (not self.trim_cue32(reply.get('observation'))):
                call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                more = None if faults else self.fill_row80(tool, call)
                try:
                    True if more is not None else (_ for _ in ()).throw(_Branch())
                    self.pull_tip(tool, more.get('observation'))
                except _Branch:
                    pass

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.sort_chunk89()
        try:
            True if mark in self.cheaper_marks or not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        self.cheaper_marks.add(mark)
        cart = self.trim_bit()
        self.pick_gauge13(cart)
        picks: list[str] = []
        _seq62 = list(cart)
        _ix63 = 0
        while _ix63 < len(_seq62):
            line = _seq62[_ix63]
            _ix63 = _ix63 + 1
            price = self.send_log(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.take_ledger(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.send_log(i) if self.send_log(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.send_log(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.make_log73(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)

    def pull_tip(self, name: str, observation: Any) -> str:
        self.get_probe(observation)
        lines = self.scan_cap(observation)
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        meaning = self.do_log(lines)
        try:
            True if meaning else (_ for _ in ()).throw(_Branch())
            self.queued.append(f'(seen on {name}) {meaning}')
        except _Branch:
            pass
        return meaning

    def scan_filter(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _read_sketch(v)), None)
        try:
            True if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare() else (_ for _ in ()).throw(_Branch())
            self.read_cap()
            return
        except _Branch:
            pass
        try:
            True if self.make_log28() <= 12 else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.get_row90()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        _seq64 = list(ranked)
        _ix65 = 0
        while _ix65 < len(_seq64):
            ids = _seq64[_ix65]
            _ix65 = _ix65 + 1
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        try:
            True if len(picks) < 2 else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        faults: list[str] = []
        args = _shape({target: picks}, schema, 'args', faults)
        try:
            True if faults else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        before = len(self.full)
        reply = self.fill_row80(tool, args)
        try:
            True if reply is not None else (_ for _ in ()).throw(_Branch())
            seen = reply.get('observation')
            self.pull_tip(tool, seen)
            self.note_view(tool, seen)
            rows = self.note_ref(seen)
            try:
                True if rows else (_ for _ in ()).throw(_Branch())
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_make_probe(r) for _, r in rows))
            except _Branch:
                pass
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            try:
                True if rows and self.facts_from.get(tool) is False else (_ for _ in ()).throw(_Branch())
                self.read_cap()
                return
            except _Branch:
                pass
            try:
                True if len(self.full) == before and len(picks) > 2 and (self.make_log28() > 12) else (_ for _ in ()).throw(_Branch())
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                try:
                    True if not faults else (_ for _ in ()).throw(_Branch())
                    reply = self.fill_row80(tool, args)
                    try:
                        True if reply is not None else (_ for _ in ()).throw(_Branch())
                        self.pull_tip(tool, reply.get('observation'))
                    except _Branch:
                        pass
                except _Branch:
                    pass
            except _Branch:
                pass
        except _Branch:
            pass

    def read_cap(self) -> None:
        tool = self.facts_tool()
        target = self.pack_gauge(tool)
        try:
            True if not tool or not target else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        wanted = {w for _, need in self.get_slot81() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        _seq66 = list(self.get_row90())
        _ix67 = 0
        while _ix67 < len(_seq66):
            ids = _seq66[_ix67]
            _ix67 = _ix67 + 1
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.take_ledger(i)]
            try:
                True if self.brief.get('pick') == 'cheapest' else (_ for _ in ()).throw(_Branch())
                pool.sort(key=lambda i: self.send_log(i) if self.send_log(i) is not None else float('inf'))
            except _Branch:
                pass
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _trim_log(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.make_log73(first[:cap][:8], 22)
        self.make_log73((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def send_cue79(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.view_chunk() <= 0:
            return False
        if self.make_log28() <= 8 or self.seek_frame() > ASK_BY:
            return False
        opened = len(self.full)
        try:
            True if opened >= 3 * len(self.items) + 1 else (_ for _ in ()).throw(_Branch())
            return True
        except _Branch:
            pass
        return self.step >= max(8, self.max_steps // 4)

    def trim_ref(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v)]
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
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
            except _Branch:
                pass
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def more_to_ask(self) -> bool:
        """Whether an interview that ended with free messages can now do better: it ran out of labels (or never had any)
        and more listings have been read since. At most REPLANS times, so it cannot loop; the budget stays with `send`."""
        if not self.asked or self.replans >= REPLANS or not self.roles.get('ask') or self.view_chunk() <= 0:
            return False
        if len(self.full) <= self.plan_full or self.make_log28() <= 8 or self.seek_frame() > ASK_BY:
            return False
        return not any(self.labels_left(n) for n in self.lanes)

    def labels_left(self, number: int) -> list[str]:
        used = self.lane_used.get(number, set())
        return [label for label in self.lanes.get(number) or [] if _note_trace58(label) not in used and not self.note_filter(number, label)]

    def take_key57(self) -> str:
        if self.asked:
            self.replans += 1
        self.asked = True
        if self.view_chunk() <= 0:
            return ''
        self.scan_filter()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: self.lane_found.get(n, 0) for n in self.lanes}
        self.lane_spent = {n: self.lane_spent.get(n, 0) for n in self.lanes}
        self.lane_used = {n: self.lane_used.get(n, set()) for n in self.lanes}
        self.follow_up = []
        self.plan_full = len(self.full)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.view_chunk() > 0 and (not self.finished) and (self.make_log28() > 6) and (self.seek_frame() < QUESTION_BY):
            batch = self.get_ref18(told)
            try:
                True if not batch else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            sent += batch
        if not sent:
            self.trace.append({'kind': 'nothing_asked', 'free': self.view_chunk(), 'opened': len(self.full)})
            return ''
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.trim_anchor(found_items) if found_items else ''
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
        try:
            True if pointed else (_ for _ in ()).throw(_Branch())
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        except _Branch:
            pass
        if searched:
            summary += '\n' + searched
        try:
            True if self.queued else (_ for _ in ()).throw(_Branch())
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        except _Branch:
            pass
        return summary

    def pull_cue(self, stage: str='order') -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Halo is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''.

        A message the request keeps for a step of this stage stays kept until that step's gate has passed; after the gate
        nothing is left to keep it for, and it goes to a question too. Every free message goes, until an answer adds a
        requirement; each call is bound by the message budget, so it cannot loop. `stage` is 'order', or 'end' before a
        decline. While nothing at all was asked yet, the clock allows it later (LATE_BY_UNASKED)."""
        if not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        need = 0
        if self.scan_set(stage):
            need = max([p['messages'] for p in self.process if p['when'] == stage] + [0]) - self.gate_msgs
        if self.msg_limit - self.msg_used - max(0, need) <= 0:
            return ''
        late_by = LATE_BY_UNASKED if not self.msg_used else LATE_BY
        if self.make_log28() < LATE_STEPS or self.seek_frame() > late_by or (not self.sort_slot()):
            return ''
        if not self.lanes or (not any(self.labels_left(n) for n in self.lanes) and len(self.full) > self.plan_full and self.replans < REPLANS):
            self.replans += 1 if self.lanes else 0
            self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: self.lane_found.get(n, 0) for n in self.lanes}
            self.lane_spent = {n: self.lane_spent.get(n, 0) for n in self.lanes}
            self.lane_used = {n: self.lane_used.get(n, set()) for n in self.lanes}
            self.plan_full = len(self.full)
        told: list[str] = []
        budget = self.budget
        batch: list[tuple[int, str, str]] = []
        while not told and self.msg_limit - self.msg_used - max(0, need) > 0 and self.make_log28() >= LATE_STEPS and self.seek_frame() <= late_by and not self.finished:
            sent = self.get_ref18(told, LATE_STEPS - 2)
            if not sent:
                break
            batch = sent
            self.trace.append({'kind': 'late_ask', 'q': sent[0][1], 'a': sent[0][2][:200], 'told': list(told)})
            if self.budget != budget:
                told.append(f'budget now {self.budget}')
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        number, question, answer = batch[0]
        try:
            True if not told else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.version += 1
        summary = f'Halo asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        try:
            True if pointed else (_ for _ in ()).throw(_Branch())
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        except _Branch:
            pass
        searched = self.trim_anchor([number])
        return summary + ('\n' + searched if searched else '')

    def fill_path73(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
        _seq68 = list(reversed(self.searched))
        _ix69 = 0
        while _ix69 < len(_seq68):
            args = _seq68[_ix69]
            _ix69 = _ix69 + 1
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            try:
                True if not named else (_ for _ in ()).throw(_Branch())
                labels = {str(r.get(self.sort_chunk89()) or '').strip().lower() for r in self.rows.values()} - {''} if self.sort_chunk89() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            except _Branch:
                pass
            try:
                True if fields and tool else (_ for _ in ()).throw(_Branch())
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.sort_chunk89()
                shelves = {_note_trace58(item.shelf or '') for item in self.items} | {_note_trace58(item.name) for item in self.items}
                shelves |= {_note_trace58(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_note_trace58(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    try:
                        True if isinstance((props.get(k) or {}).get('enum'), list) else (_ for _ in ()).throw(_Branch())
                        used = [_trim_log(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _trim_log(a[k]) == best))
                    except _Branch:
                        pass
                return (tool, keep, field)
            except _Branch:
                pass
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        out = []
        told = self.items[number - 1].told
        for n, need in self.get_slot81():
            try:
                True if n != number or need in [m for _, m in self.refs] or ':' not in need else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            value = need.split(':', 1)[1].strip()
            edge = re.search(FLOOR, value, re.I)
            try:
                True if edge else (_ for _ in ()).throw(_Branch())
                try:
                    True if need in told else (_ for _ in ()).throw(_Branch())
                    out.append((edge.group(1).strip(), True))
                except _Branch:
                    pass
                continue
            except _Branch:
                pass
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append((value, need in told))
        return out

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.fill_path73()
        tool = self.roles.get('search')
        if plan or not tool or (not self.searched):
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v) and re.search(QUERY_WORDS, _snake(k), re.I)]
        last = self.searched[-1]
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            else:
                try:
                    True if key in required and len(options) == 2 and (len(others) == 1) else (_ for _ in ()).throw(_Branch())
                    modes.append((key, others[0]))
                except _Branch:
                    pass
        try:
            True if len(named) != 1 or len(modes) > 1 or (enums and (not modes)) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def trim_anchor(self, numbers: list[int]) -> str:
        plan = self.value_template()
        try:
            True if not plan or self.make_log28() < 22 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        _seq72 = list(numbers)
        _ix73 = 0
        while _ix73 < len(_seq72):
            number = _seq72[_ix73]
            _ix73 = _ix73 + 1
            pairs = self.value_words(number)
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
            _seq70 = list(tries[:2])
            _ix71 = 0
            while _ix71 < len(_seq70):
                words = _seq70[_ix71]
                _ix71 = _ix71 + 1
                try:
                    True if not words else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                try:
                    True if faults or self.finished else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                reply = self.fill_row80(tool, args)
                budget -= 1
                try:
                    True if reply is None else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                observation = reply.get('observation')
                self.pull_tip(tool, observation)
                cat_key = self.sort_chunk89()
                found = [i for i, _ in self.note_ref(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.send_log(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.take_ledger(i) and (not cat_key or self.send_frame(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.send_log(i) or 0) <= cap)]
                home = self.home_label(number)
                try:
                    True if cat_key and home else (_ for _ in ()).throw(_Branch())
                    rows.sort(key=lambda i: _note_trace58(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                except _Branch:
                    pass
                seen.append(f'{_trim_log(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.make_log73([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        try:
            True if not seen else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Halo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.sort_chunk89()
        try:
            True if not cat_key else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                try:
                    True if label and self.send_frame(item, label) else (_ for _ in ()).throw(_Branch())
                    counts[_note_trace58(label)] = counts.get(_note_trace58(label), 0) + 1
                except _Branch:
                    pass
            _seq74 = list((item.shelf, item.name))
            _ix75 = 0
            while _ix75 < len(_seq74):
                name = _seq74[_ix75]
                _ix75 = _ix75 + 1
                try:
                    True if _note_trace58(name) in counts else (_ for _ in ()).throw(_Branch())
                    return _note_trace58(name)
                except _Branch:
                    pass
            try:
                True if counts else (_ for _ in ()).throw(_Branch())
                return max(counts, key=lambda label: counts[label])
            except _Branch:
                pass
        return ''

    def pick_cue(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (n in self.puzzled, found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _note_trace58(label) not in used[number] and (not self.note_filter(number, label))]
            try:
                True if left else (_ for _ in ()).throw(_Branch())
                return (number, left[:LIST_SIZE if self.listing else 1])
            except _Branch:
                pass
        return (0, [])

    def note_filter(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_pack_bundle13(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def store_prefixes(self) -> set[str]:
        """Short leading words the store puts on several of its property keys as a namespace (e.g. `fa_` in
        fa_season, fa_pattern, fa_general_styles). They mean nothing to a shopper. Read from this episode's records."""
        starts: dict[str, set[str]] = {}

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    words = _note_trace58(key).split()
                    if len(words) >= 2 and len(words[0]) <= 2 and words[0].isalpha():
                        starts.setdefault(words[0], set()).add(' '.join(words))
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node[:20]:
                    walk(value, depth + 1)
        for record in self.full.values():
            walk(record, 0)
        return {word for word, keys in starts.items() if len(keys) >= PREFIX_KEYS}

    def spoken(self, labels: list[str]) -> list[str]:
        """The labels as a shopper should read them: a store namespace prefix dropped, unless dropping it would make the
        label empty or the same as another label in the question."""
        prefixes = self.store_prefixes()
        out = []
        for label in labels:
            words = label.split()
            plain = ' '.join(words[1:]) if len(words) >= 2 and words[0].lower() in prefixes else label
            clash = any(_note_trace58(plain) == _note_trace58(other) for other in labels if other != label)
            out.append(label if clash else plain)
        return out

    def mark_chunk(self, number: int, labels: list[str], shown: bool=False) -> str:
        said = self.spoken(labels)
        if len(labels) >= 2:
            return f"What {', '.join(said[:-1])} or {said[-1]} would you like for the {self.item_said(number)}?"
        return self.seek_probe(number, said[0], self.fill_key(number, labels[0]) if shown else None)

    def send_sketch(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.mark_chunk(number, labels[:size], shown)
            args = self.trim_ref(text)
            try:
                True if args and text in args.values() else (_ for _ in ()).throw(_Branch())
                return (labels[:size], text)
            except _Branch:
                pass
        return ([], '')

    def joint_group(self) -> list[tuple[int, list[str]]]:
        """The items the next message asks about together, each with its leading unasked labels, when the items not yet
        asked outnumber the free messages; [] otherwise (one item per message, as before). The waiting items are split
        into the free messages as evenly as possible, in plan order, so that no item goes unasked. Race 17 allowed two
        messages on every task, 41 of 90 tasks wanted 3 to 5 items, and an agent asking one item per message left the
        rest unasked; a message naming several items drew needs on clear tasks about as often as a one-item list."""
        free = self.view_chunk()
        if not self.listing or free <= 0 or self.follow_up or len(self.items) < 2:
            return []
        first = {n: i for i, n in enumerate(self.plan_order)}
        waiting = []
        for number in sorted(self.lanes, key=lambda n: first.get(n, len(first) + n)):
            used = self.lane_used.get(number, set())
            left = [label for label in self.lanes[number] if _note_trace58(label) not in used and (not self.note_filter(number, label))]
            if left and (not self.lane_spent.get(number)):
                waiting.append((number, left))
        if len(waiting) <= free:
            return []
        size = max(2, -(-len(waiting) // free))
        group = waiting[:size]
        each = max(1, min(JOINT_LABELS, LIST_SIZE // len(group)))
        return [(number, left[:each]) for number, left in group]

    def joint_text(self, group: list[tuple[int, list[str]]]) -> str:
        parts = []
        for k, (number, labels) in enumerate(group):
            said = self.spoken(labels)
            named = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} or {said[-1]}"
            parts.append(f'What {named} would you like for the {self.item_said(number)}' if k == 0 else f'what {named} for the {self.item_said(number)}')
        return ', and '.join(parts) + '?'

    def joint_ask(self, told: list[str], floor: int) -> list[tuple[int, str, str]] | None:
        """Ask one message about several items (see joint_group). None when no joint message applies or none fits the
        message tool; the caller then asks one item as before. Each item's labels are settled as a list of its own."""
        group = self.joint_group()
        while group:
            text = self.joint_text(group)
            args = self.trim_ref(text)
            if args and text in args.values():
                break
            longest = max(range(len(group)), key=lambda k: len(group[k][1]))
            if len(group[longest][1]) > 1:
                group[longest] = (group[longest][0], group[longest][1][:-1])
            else:
                group = group[:-1] if len(group) > 2 else []
        if not group:
            return None
        numbers = tuple(number for number, _ in group)
        batch = self.send([(numbers[0], group[0][1][0], text)], floor)
        if not batch:
            return []
        self.trace.append({'kind': 'joint_ask', 'items': list(numbers), 'labels': [labels for _, labels in group]})
        had = [len(item.told) for item in self.items]
        heard = self.send_row([(numbers, q, a) for _, q, a in batch], (LIGHT, HEAVY))
        heard.pop('free', None)
        self.send_cue(heard, told)
        crowded = any((_note_piece(x) == 1 for x in _seek_gate59(heard.get('crowded'))))
        if any((_note_piece(x) == 1 for x in _seek_gate59(heard.get('unclear')))):
            self.puzzled.update(number for number, labels in group if not self.items[number - 1].told[had[number - 1]:])
        for number, labels in group:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
            item = self.items[number - 1]
            new = item.told[had[number - 1]:]
            if new:
                self.lane_found[number] = self.lane_found.get(number, 0) + 1
            self.settle(number, labels, False, new, crowded and number == numbers[0])
        return batch

    def get_ref18(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else a message about several items when they outnumber
        the free messages, else the next planned labels), read its answer into the items' needs (appending each new one
        to `told`) and settle the labels it named. Returns the sent batch, or [] when nothing was sent."""
        joint = self.joint_ask(told, floor)
        if joint is not None:
            return joint
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.note_filter(f[0], f[1][0])]
            try:
                True if self.follow_up else (_ for _ in ()).throw(_Branch())
                number, labels, shown = self.follow_up.pop(0)
            except _Branch:
                number, labels = self.pick_cue(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                shown = False
            try:
                True if not labels else (_ for _ in ()).throw(_Branch())
                return []
            except _Branch:
                pass
            fitted, text = self.send_sketch(number, labels, shown)
            try:
                True if not fitted and shown else (_ for _ in ()).throw(_Branch())
                fitted, text = self.send_sketch(number, labels, False)
            except _Branch:
                pass
            try:
                True if fitted else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.lane_used.setdefault(number, set()).add(_note_trace58(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        try:
            True if not batch else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.send_row(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.send_cue(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_note_piece(x) == 1 for x in _seek_gate59(heard.get('crowded'))))
        unclear = any((_note_piece(x) == 1 for x in _seek_gate59(heard.get('unclear'))))
        self.settle(number, labels, shown, new, crowded, unclear)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool, unclear: bool=False) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets the base's follow-up (one product of an
        item name joining several, the property's head word, or example values). A list: the shopper answers about
        the first property of a message it tracks, and its reply to a message naming none of its properties is the
        same as to a property it does not mind. So a disclosed need settles its label and every label before it, and
        the labels after it lead the item's next list; a reply that discloses nothing settles only the first label
        (a need there would have been answered) and the others come next in reverse order, so whatever the shopper
        does not mind no longer stands in front of them, behind the labels not yet named (race 18: after such a reply a list of
        new labels drew a need 97 of 325 times, one repeating labels 17 of 88); after a second such reply in a row the
        reversed labels lead. A reply asking for one question at a time ends lists for
        the episode and the list's first label is asked alone. A reply saying the shopper did not follow a list (and
        disclosing nothing) has the list's first label asked alone next, once per label; lists stay on."""
        used = self.lane_used.setdefault(number, set())
        try:
            True if len(labels) == 1 else (_ for _ in ()).throw(_Branch())
            used.add(_note_trace58(labels[0]))
            try:
                True if new or not crowded else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            try:
                True if number not in self.one_product and (self.name_parts(number) or self.store_label(number)) else (_ for _ in ()).throw(_Branch())
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            except _Branch:
                pass
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _note_trace58(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            else:
                try:
                    True if not shown and len(self.fill_key(number, labels[0])) >= 2 else (_ for _ in ()).throw(_Branch())
                    self.follow_up.insert(0, (number, labels, True))
                except _Branch:
                    pass
            return
        except _Branch:
            pass
        if crowded and (not new):
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        if unclear and (not new) and self.go_elsewhere(number, labels):
            return
        if unclear and (not new) and labels[0] not in self.list_unclear:
            self.list_unclear.add(labels[0])
            self.trace.append({'kind': 'list_unclear', 'item': number, 'next': labels[:1]})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.mark_signal94(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_note_trace58(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = ahead + rest if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else rest + ahead
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def go_elsewhere(self, number: int, labels: list[str]) -> bool:
        """After a puzzled reply about an item, leave it for the items not yet asked about, when there are any. Race 18,
        seven agents: after a puzzled reply a further message about the same item was never answered firmly (0 of 33), one
        about another item 21 of 130 times."""
        self.puzzled.add(number)
        others = [n for n in self.lanes if n != number and n not in self.puzzled and self.labels_left(n)]
        if not others:
            return False
        self.lane_used.setdefault(number, set()).update(_note_trace58(label) for label in labels)
        self.trace.append({'kind': 'puzzled', 'item': number, 'next': others})
        return True

    def mark_signal94(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.send_trace():
            seen.setdefault(_note_trace58(key), set()).update((w for v in values for w in _roots(v)))
        hits = []
        _seq76 = list(new)
        _ix77 = 0
        while _ix77 < len(_seq76):
            need = _seq76[_ix77]
            _ix77 = _ix77 + 1
            prop, _, value = need.partition(':')
            scores = [len(_spelled(label) & _spelled(prop)) for label in labels]
            try:
                True if max(scores, default=0) == 0 else (_ for _ in ()).throw(_Branch())
                scores = [len(seen.get(_note_trace58(label), set()) & _roots(value)) for label in labels]
            except _Branch:
                pass
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
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
        keys = [key for _, key, _, _ in rows if not any((_pack_bundle13(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def make_tip(self) -> None:
        cat_key = self.sort_chunk89()
        try:
            True if len(self.items) < 2 or not cat_key else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
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
        words = _note_trace58(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            try:
                True if words == _note_trace58(item.shelf or '') else (_ for _ in ()).throw(_Branch())
                score = 3.0
            except _Branch:
                try:
                    True if words == _note_trace58(item.name) else (_ for _ in ()).throw(_Branch())
                    score = 2.0
                except _Branch:
                    mine = _roots(item.shelf or '') | _roots(item.name)
                    score = len(long & mine) / max(1, len(long | mine))
            try:
                True if score > top else (_ for _ in ()).throw(_Branch())
                best, top = (number, score)
            except _Branch:
                pass
        return best

    def send_frame(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def pick_bag89(self, number: int) -> int:
        cat_key = self.sort_chunk89()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.send_frame(item, str(record.get(cat_key) or '')) and _make_probe(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.make_tip()
        table = self.send_trace()
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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.pack_gate(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _seek_gate59(self.brief.get('ask_first')) if str(x).strip()]
        try:
            True if first else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        except _Branch:
            pass
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.sort_gauge(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        _seq78 = list(_seek_gate59(plan.get('ask')))
        _ix79 = 0
        while _ix79 < len(_seq78):
            pick = _seq78[_ix79]
            _ix79 = _ix79 + 1
            try:
                True if not isinstance(pick, dict) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            number = min(max(1, _note_piece(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            try:
                True if not label or len(label.split()) > 5 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            try:
                True if all((_note_trace58(label) != _note_trace58(other) for other in lanes[number])) else (_ for _ in ()).throw(_Branch())
                lanes[number].append(label)
            except _Branch:
                pass
            try:
                True if str(pick.get('hint') or '').strip() else (_ for _ in ()).throw(_Branch())
                hinted[number].add(_note_trace58(label))
                try:
                    True if number not in self.plan_order else (_ for _ in ()).throw(_Branch())
                    self.plan_order.append(number)
                except _Branch:
                    pass
            except _Branch:
                pass
        _seq86 = list(lanes)
        _ix87 = 0
        while _ix87 < len(_seq86):
            number = _seq86[_ix87]
            _ix87 = _ix87 + 1
            keys = self.lane_keys(number, table)
            read = max(1, self.pick_bag89(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                try:
                    True if self.owner_of(label) == number else (_ for _ in ()).throw(_Branch())
                    own[key] = max(own.get(key, 0.0), carried / read)
                except _Branch:
                    pass

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _note_trace58(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                try:
                    True if _note_trace58(label) in own else (_ for _ in ()).throw(_Branch())
                    return own[_note_trace58(label)]
                except _Branch:
                    pass
                near = [s for k, s in own.items() if _pack_bundle13(_stem(label), _stem(k)) or _pack_bundle13(_stem(k), _stem(label))]
                return max(near) if near else 0.0
            hinted[number] = {_note_trace58(spelled(h)) for h in hinted[number]}
            lane = []
            _seq80 = list(lanes[number])
            _ix81 = 0
            while _ix81 < len(_seq80):
                pick = _seq80[_ix81]
                _ix81 = _ix81 + 1
                try:
                    True if all((_note_trace58(spelled(pick)) != _note_trace58(other) for other in lane)) else (_ for _ in ()).throw(_Branch())
                    lane.append(spelled(pick))
                except _Branch:
                    pass
            _seq82 = list([k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS])
            _ix83 = 0
            while _ix83 < len(_seq82):
                key = _seq82[_ix83]
                _ix83 = _ix83 + 1
                try:
                    True if any((_note_trace58(key) == _note_trace58(other) for other in lane[:3])) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                weak = next((i for i, other in enumerate(lane[:3]) if _note_trace58(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                try:
                    True if weak is None and len(lane) >= 3 else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                placed = [other for other in lane if _note_trace58(other) != _note_trace58(key)]
                try:
                    True if weak is not None and len(self.items) > 1 else (_ for _ in ()).throw(_Branch())
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                except _Branch:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                try:
                    True if any((_note_trace58(x) in hinted[number] and all((_note_trace58(x) != _note_trace58(y) for y in placed[:3])) for x in lane[:3])) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                lane = placed
            lanes[number] = lane
            _seq84 = list(keys)
            _ix85 = 0
            while _ix85 < len(_seq84):
                key = _seq84[_ix85]
                _ix85 = _ix85 + 1
                try:
                    True if all((_note_trace58(key) != _note_trace58(other) for other in lanes[number])) else (_ for _ in ()).throw(_Branch())
                    lanes[number].append(key)
                except _Branch:
                    pass
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _note_trace58(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _note_trace58(x))]
        self.trace.append({'kind': 'plan', 'ask': _do_slot(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        name = self.items[number - 1].name
        try:
            True if not re.search('\\band\\b', name, re.I) else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first). Before any one-at-a-time reply, the store's own label for the item when one is
        known (see store_label)."""
        item = self.items[number - 1]
        if number not in self.one_product:
            store_name = self.store_label(number)
            if store_name:
                return store_name
        parts = self.name_parts(number)
        try:
            True if not parts else (_ for _ in ()).throw(_Branch())
            return item.name
        except _Branch:
            pass
        try:
            True if number not in self.one_product else (_ for _ in ()).throw(_Branch())
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        except _Branch:
            pass
        cat_key = self.sort_chunk89()
        rows = [row for row in self.rows.values() if not cat_key or self.send_frame(item, str(row.get(cat_key) or ''))]
        named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def store_label(self, number: int) -> str:
        """The item's category label exactly as the store's listings print it (lower case, spaces squeezed), for
        naming the item in a question; '' when it is unknown, shared with another requested item, or does not clearly
        name this item (it must equal the request's shelf name or contain the head noun of the item's name). A seen
        label of the item's that holds every word of its name wins over home_label's choice ("computer monitors"
        over "portable monitors" for a computer monitor).
        Race 14 and 15 shoppers answered firmly 63% of the time when a question named the item by this label and 22%
        otherwise (831 questions, label taken from other episodes of the same task); their firm answers repeat it."""
        cat_key = self.sort_chunk89()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        printed_of: dict[str, str] = {}
        seen_count: dict[str, int] = {}
        for pool in (self.full, self.rows):
            for record in pool.values():
                printed = ' '.join(str(record.get(cat_key) or '').split())
                if printed and self.send_frame(item, printed):
                    words = _note_trace58(printed)
                    printed_of.setdefault(words, printed.lower())
                    seen_count[words] = seen_count.get(words, 0) + 1
        name_roots = _roots(item.name)
        whole = [w for w in seen_count if name_roots and name_roots <= _roots(w)]
        if whole:
            chosen = max(whole, key=lambda w: seen_count[w])
        else:
            chosen = self.home_label(number)
            head_noun = _roots(item.name.split()[-1]) if item.name.split() else set()
            if not chosen or not (chosen == _note_trace58(item.shelf or '') or head_noun & _roots(chosen)):
                return ''
        for other in range(1, len(self.items) + 1):
            if other != number and self.home_label(other) == chosen:
                return ''
        return printed_of.get(chosen, '')

    def seek_probe(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def fill_key(self, number: int, label: str) -> list[str]:
        want = _note_trace58(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.send_trace():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            try:
                True if self.finished or self.make_log28() <= floor or self.msg_used >= self.msg_limit else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            args = self.trim_ref(text)
            if not args:
                break
            reply = self.fill_row80(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.read_chunk(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and (answer.get('error') or not re.search('[a-z]\\s+[a-z]', _trim_log(answer), re.I)):
                self.msg_limit = self.msg_used
                self.trace.append({'kind': 'message_refused', 'reply': _do_slot(answer, 120)})
                break
        return sent

    def send_cue(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        _seq88 = list(_seek_gate59(heard.get('needs')))
        _ix89 = 0
        while _ix89 < len(_seq88):
            entry = _seq88[_ix89]
            _ix89 = _ix89 + 1
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _note_piece(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            try:
                True if need not in self.items[number - 1].told else (_ for _ in ()).throw(_Branch())
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            except _Branch:
                pass
            try:
                True if entry.get('points') is True and (number, need) not in self.refs else (_ for _ in ()).throw(_Branch())
                self.refs.append((number, need))
            except _Branch:
                pass
        for entry in _seek_gate59(heard.get('drop')):
            try:
                True if isinstance(entry, dict) and str(entry.get('property') or '').strip() else (_ for _ in ()).throw(_Branch())
                item = self.items[min(max(1, _note_piece(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _note_trace58(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _note_trace58(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
            except _Branch:
                pass
        _seq90 = list(_seek_gate59(heard.get('free')))
        _ix91 = 0
        while _ix91 < len(_seq90):
            entry = _seq90[_ix91]
            _ix91 = _ix91 + 1
            try:
                True if isinstance(entry, dict) and entry.get('property') else (_ for _ in ()).throw(_Branch())
                number = min(max(1, _note_piece(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
            except _Branch:
                pass
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def send_row(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        about = lambda n: 'items ' + ' and '.join((str(x) for x in n)) if isinstance(n, tuple) else f'item {n}'
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about {about(n)}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.sort_gauge(prompt, list(models), 2500) or {}

    def hold_map37(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        try:
            True if not steps or self.finished or (not self.scan_set(stage)) else (_ for _ in ()).throw(_Branch())
            return (True, '')
        except _Branch:
            pass
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_trim_log(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        _ri92 = 0
        while _ri92 < GATE_ROUNDS:
            _ = _ri92
            _ri92 = _ri92 + 1
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.fill_signal() + '\n\nCART NOW: ' + self.scan_row38() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.make_log28()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.sort_gauge(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            try:
                True if said else (_ for _ in ()).throw(_Branch())
                self.send_cue({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            except _Branch:
                pass
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _do_slot(call, 200), 'steps': _do_slot(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _seek_gate59(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_note_piece(n) or 0) in proof for n in _seek_gate59(e.get('lines'))))]
            try:
                True if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.make_log28() >= 2) else (_ for _ in ()).throw(_Branch())
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _do_slot(unproven, 120)})
                self.done_log.append(f'(Halo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.make_log28()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            except _Branch:
                pass
            try:
                True if not call or name not in self.schemas or name in ending else (_ for _ in ()).throw(_Branch())
                try:
                    True if not call else (_ for _ in ()).throw(_Branch())
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                except _Branch:
                    pass
                break
            except _Branch:
                pass
            try:
                True if self.make_log28() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit) else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            try:
                True if name == self.roles.get('ask') else (_ for _ in ()).throw(_Branch())
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, '', text)], 1)
                try:
                    True if not batch else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            except _Branch:
                pass
            reply = self.fill_row80(name, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            seen = reply.get('observation')
            self.get_probe(seen)
            self.done_log.append(f'step {self.step}: {name} {_trim_log(args)} -> {self.seek_probe36(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_trim_log(args)} -> {self.seek_probe36(seen)[:400]}')
        return (True, 'Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def read_chunk(self, observation: Any) -> str:
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
                _seq93 = list(node.values())
                _ix94 = 0
                while _ix94 < len(_seq93):
                    value = _seq93[_ix94]
                    _ix94 = _ix94 + 1
                    walk(value, depth + 1)
            elif isinstance(node, list):
                _seq95 = list(node)
                _ix96 = 0
                while _ix96 < len(_seq95):
                    value = _seq95[_ix96]
                    _ix96 = _ix96 + 1
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _trim_log(observation)[:400]

    def scan_cap(self, observation: Any) -> list[str]:
        lines: list[str] = []
        try:
            True if isinstance(observation, dict) and (not self.ident(observation)) else (_ for _ in ()).throw(_Branch())
            for key, value in observation.items():
                try:
                    True if isinstance(value, str) and len(value.split()) >= 5 else (_ for _ in ()).throw(_Branch())
                    lines.append(value)
                except _Branch:
                    try:
                        True if isinstance(value, list) and value and all((isinstance(v, str) for v in value)) else (_ for _ in ()).throw(_Branch())
                        lines.extend((v for v in value if len(v.split()) >= 5))
                    except _Branch:
                        try:
                            True if isinstance(value, dict) and (not self.ident(value, False, 1)) else (_ for _ in ()).throw(_Branch())
                            _seq97 = list(value.values())
                            _ix98 = 0
                            while _ix98 < len(_seq97):
                                inner = _seq97[_ix98]
                                _ix98 = _ix98 + 1
                                try:
                                    True if isinstance(inner, str) and len(inner.split()) >= 6 else (_ for _ in ()).throw(_Branch())
                                    lines.append(inner)
                                except _Branch:
                                    pass
                        except _Branch:
                            pass
        except _Branch:
            pass
        return [line for line in lines if line not in self.heard]

    def do_log(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.fill_signal() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.sort_gauge(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _seek_gate59(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        try:
            True if self.note_sold and len(lines) == 1 else (_ for _ in ()).throw(_Branch())
            self.sold_lines.update(lines)
        except _Branch:
            pass
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _seek_gate59(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _seek_gate59(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.items[min(max(1, _note_piece(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _note_trace58(m.split(':')[0]) == _note_trace58(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, _note_piece(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            try:
                True if need in item.dropped else (_ for _ in ()).throw(_Branch())
                item.dropped.remove(need)
            except _Branch:
                pass
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
        try:
            True if got.get('price_moved') is True else (_ for _ in ()).throw(_Branch())
            changed.append('a price changed')
        except _Branch:
            pass
        summary = str(got.get('summary') or '').strip()
        try:
            True if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))) else (_ for _ in ()).throw(_Branch())
            self.wider_held = False
            self.cheaper_marks = set()
        except _Branch:
            pass
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def get_slot81(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                try:
                    True if need not in item.dropped else (_ for _ in ()).throw(_Branch())
                    out.append((number, need))
                except _Branch:
                    pass
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            try:
                True if depth > 6 else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            try:
                True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
                ident = self.ident(node, False, depth)
                if ident and _make_probe(node) and (ident not in ids):
                    ids.append(ident)
                    return
                _seq99 = list(node.values())
                _ix100 = 0
                while _ix100 < len(_seq99):
                    value = _seq99[_ix100]
                    _ix100 = _ix100 + 1
                    walk(value, depth + 1)
            except _Branch:
                try:
                    True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                    _seq101 = list(node)
                    _ix102 = 0
                    while _ix102 < len(_seq101):
                        value = _seq101[_ix102]
                        _ix102 = _ix102 + 1
                        walk(value, depth + 1)
                except _Branch:
                    pass
        walk(observation, 0)
        needs = self.get_slot81()
        try:
            True if not ids or not needs else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        many = len(self.items) > 1
        lines = []
        _seq103 = list(ids[:12])
        _ix104 = 0
        while _ix104 < len(_seq103):
            ident = _seq103[_ix104]
            _ix104 = _ix104 + 1
            block = _make_probe(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _pack_bundle13(_stem(prop), _stem(k))]
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

    def scan_row38(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _trim_log(_do_slot(self.cart_view, 300))[:3000]
        return "lines by Halo's record of the cart calls (listing: units): " + _trim_log(self.cart_lines)

    def run_gauge30(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    try:
                        True if ident in [v.strip() for v in node.values() if isinstance(v, str)] else (_ for _ in ()).throw(_Branch())
                        counts = [_note_piece(v) for k, v in node.items() if not isinstance(v, str) and _note_piece(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _note_trace58(k))]
                        try:
                            True if len(counts) == 1 else (_ for _ in ()).throw(_Branch())
                            return counts[0]
                        except _Branch:
                            pass
                    except _Branch:
                        pass
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def trim_bit(self) -> list[str]:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            text = _trim_log(self.cart_view)
            return [i for i in self.rows if i in text]
        except _Branch:
            pass
        return list(self.cart_lines)

    def pull_trace(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _trim_log(_do_slot(record, 400))[:3000] if record else '(no record seen)'

    def scan_key(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        try:
            True if self.brief.get('pick') == 'cheapest' else (_ for _ in ()).throw(_Branch())
            order.sort(key=lambda i: (i not in self.full, self.send_log(i) if self.send_log(i) is not None else float('inf')))
        except _Branch:
            pass
        _seq105 = list(order)
        _ix106 = 0
        while _ix106 < len(_seq105):
            ident = _seq105[_ix106]
            _ix106 = _ix106 + 1
            try:
                True if ident in skip or self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_trim_log(_do_slot(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def fill_pad(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        _seq107 = list(self.seen_order)
        _ix108 = 0
        while _ix108 < len(_seq107):
            ident = _seq107[_ix108]
            _ix108 = _ix108 + 1
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _take_piece(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fill_gate(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        try:
            True if mark.get('own') else (_ for _ in ()).throw(_Branch())
            found |= index['notes'].get(mark['own'], set())
        except _Branch:
            pass
        return found

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.sort_chunk89()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.send_frame(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        try:
            True if not self.bound else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        index = self.fill_pad()
        cart = self.trim_bit()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            try:
                True if len(devices) != 1 or not parts else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            try:
                True if not codes else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            _seq109 = list(parts)
            _ix110 = 0
            while _ix110 < len(_seq109):
                line = _seq109[_ix110]
                _ix110 = _ix110 + 1
                fits = self.fill_gate(line, index)
                try:
                    True if fits and (not fits & codes) else (_ for _ in ()).throw(_Branch())
                    good = [i for i in self.item_ids(part) if self.fill_gate(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.keep_set(short=True)))
                except _Branch:
                    pass
        return faults

    def keep_set(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fill_pad()
        lines = []
        _seq113 = list(sorted({m for _, m in self.bound}))
        _ix114 = 0
        while _ix114 < len(_seq113):
            main = _seq113[_ix114]
            _ix114 = _ix114 + 1
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            _seq111 = list(parts)
            _ix112 = 0
            while _ix112 < len(_seq111):
                p = _seq111[_ix112]
                _ix112 = _ix112 + 1
                for i in self.item_ids(p):
                    try:
                        True if self.sold_out(i) else (_ for _ in ()).throw(_Branch())
                        continue
                    except _Branch:
                        pass
                    for code in self.fill_gate(i, index):
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

    def make_ref(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.sort_chunk89()
        ids = [i for i in self.trim_bit() if (self.run_gauge30(i) or 0) > 0]
        try:
            True if not cat_key or len(ids) >= len(self.items) else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.send_frame(item, label) for label in labels)) and (not any((self.send_frame(item, label) for label in held)))]

    def get_filter9(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        block = _make_probe(self.full.get(ident) or {}) or {}
        for owner, need in self.get_slot81():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            split = lambda text: set(re.sub('[,/;|]', ' ', _note_trace58(text)).split())
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            try:
                True if not any((_pack_bundle13(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())) else (_ for _ in ()).throw(_Branch())
                return False
            except _Branch:
                pass
        return True

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.make_ref():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.take_ledger(i) and (self.send_log(i) is not None) and self.get_filter9(i, number)]
            fits.sort(key=lambda i: self.send_log(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            try:
                True if pick and isinstance(self.budget, (int, float)) else (_ for _ in ()).throw(_Branch())
                held = sum(((self.send_log(i) or 0.0) * (self.run_gauge30(i) or 0) for i in self.trim_bit()))
                cost = (self.send_log(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                try:
                    True if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09 else (_ for _ in ()).throw(_Branch())
                    pick = None
                except _Branch:
                    pass
            except _Branch:
                pass
            calls = self.view_gauge93({**{i: self.cart_lines[i] for i in self.trim_bit() if i in self.cart_lines}, pick: item.qty}, []) if pick else None
            try:
                True if not calls or len(calls) + 1 > self.make_log28() else (_ for _ in ()).throw(_Branch())
                return False
            except _Branch:
                pass
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), 'args', faults)
                try:
                    True if faults or self.fill_row80(cart, args) is None else (_ for _ in ()).throw(_Branch())
                    return False
                except _Branch:
                    pass
                self.get_gauge(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def send_row39(self, number: int, label: str) -> str:
        cat_key = self.sort_chunk89()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or (not cat_key) or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def keep_filter10(self) -> list[str]:
        self.make_tip()
        cat_key = self.sort_chunk89()
        ids = [i for i in self.trim_bit() if (self.run_gauge30(i) or 0) > 0]
        try:
            True if not ids or not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.send_frame(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.send_frame(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            try:
                True if n not in known else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            lines = [i for i in ids if n in owner[i]]
            try:
                True if not lines else (_ for _ in ()).throw(_Branch())
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            except _Branch:
                try:
                    True if len([i for i in lines if owner[i] == [n]]) > 1 else (_ for _ in ()).throw(_Branch())
                    faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
                except _Branch:
                    pass
            _seq115 = list(lines)
            _ix116 = 0
            while _ix116 < len(_seq115):
                i = _seq115[_ix116]
                _ix116 = _ix116 + 1
                have = self.run_gauge30(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.send_row39(n, label[i])
                try:
                    True if fuller else (_ for _ in ()).throw(_Branch())
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
                except _Branch:
                    pass
        if len(known) == len(self.items):
            _seq117 = list(ids)
            _ix118 = 0
            while _ix118 < len(_seq117):
                i = _seq117[_ix118]
                _ix118 = _ix118 + 1
                try:
                    True if not owner[i] else (_ for _ in ()).throw(_Branch())
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
                except _Branch:
                    pass
        _seq119 = list(ids)
        _ix120 = 0
        while _ix120 < len(_seq119):
            i = _seq119[_ix120]
            _ix120 = _ix120 + 1
            try:
                True if self.sold_out(i) else (_ for _ in ()).throw(_Branch())
                faults.append(f'{i} is sold out')
            except _Branch:
                pass
        prices = {i: self.send_log(i) for i in ids}
        try:
            True if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())) else (_ for _ in ()).throw(_Branch())
            try:
                True if self.brief.get('budget_scope') == 'each' else (_ for _ in ()).throw(_Branch())
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            except _Branch:
                total = sum((prices[i] * self.run_gauge30(i) for i in ids))
                try:
                    True if self.budget + 1e-09 < total <= 10 * self.budget else (_ for _ in ()).throw(_Branch())
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
                except _Branch:
                    pass
        except _Branch:
            pass
        return faults

    def make_path18(self) -> list[tuple[str, int, int]]:
        cat_key = self.sort_chunk89()
        try:
            True if not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        out = []
        _seq121 = list(self.trim_bit())
        _ix122 = 0
        while _ix122 < len(_seq121):
            ident = _seq121[_ix122]
            _ix122 = _ix122 + 1
            have = self.run_gauge30(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.send_frame(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.trim_bit() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.send_frame(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
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
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _read_sketch(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _read_sketch(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        try:
            True if not name or not target or (not count) else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.do_bag(name):
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
        try:
            True if len(calls) + 3 > self.make_log28() else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.fill_row80(name, args)
            if reply is None or self.finished:
                break
            self.get_gauge(name, args)
            self.cart_said = _trim_log(reply.get('observation'))
            self.pull_tip(name, reply.get('observation'))
            done.append(_trim_log(args) + ' -> ' + self.seek_probe36(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def sort_chunk59(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.trim_bit()
        key = self.sort_chunk59(ids)
        try:
            True if key in self.checked or not self.sort_slot(limit) or (not ids) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.fill_signal()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.scan_row38()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.pull_trace(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.scan_key(ids, 15)
        judges = self.sort_slot(limit)
        got = None
        try:
            True if len(judges) > 1 and (not lead) and (not self.audited) and (self.brief.get('pick') not in ('cheapest', 'most', 'least')) else (_ for _ in ()).throw(_Branch())
            got = self.sort_gauge(prompt, [LIGHT], 3000)
            try:
                True if got else (_ for _ in ()).throw(_Branch())
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.sort_slot(limit) else []
            except _Branch:
                pass
        except _Branch:
            pass
        try:
            True if judges and (not self.no_credit) else (_ for _ in ()).throw(_Branch())
            got = self.sort_gauge(prompt, judges, 3000) or got
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
        problems = [str(p) for p in _seek_gate59(got.get('problems'))][:5]
        better = [str(b) for b in _seek_gate59(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def run_row(self) -> tuple[str, bool]:
        live = {need for _, need in self.get_slot81()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.pack_gauge(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.make_log28() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _trim_log(_do_slot(record, 300))[:1400])
            _seq123 = list(reversed(self.seen_order))
            _ix124 = 0
            while _ix124 < len(_seq123):
                ident = _seq123[_ix124]
                _ix124 = _ix124 + 1
                try:
                    True if len(records) >= 60 else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                try:
                    True if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)) else (_ for _ in ()).throw(_Branch())
                    records.append('ROW ' + _trim_log(_do_slot(self.rows.get(ident), 80))[:300])
                except _Branch:
                    pass
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.sort_gauge(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _do_slot(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            _seq125 = list(_seek_gate59(got.get('chains')))
            _ix126 = 0
            while _ix126 < len(_seq125):
                chain = _seq125[_ix126]
                _ix126 = _ix126 + 1
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                try:
                    True if not need else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                links = [l for l in _seek_gate59(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                try:
                    True if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip() else (_ for _ in ()).throw(_Branch())
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                except _Branch:
                    pass
                try:
                    True if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()) else (_ for _ in ()).throw(_Branch())
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                except _Branch:
                    pass
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
            if not reach and gaps and tool and target and (self.make_log28() > 12) and (self.link_searches < 3):
                reach = self.scan_key21(*gaps[0])
                try:
                    True if self.trace and self.trace[-1].get('kind') == 'link_known' else (_ for _ in ()).throw(_Branch())
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
                except _Branch:
                    pass
            try:
                True if not reach or not tool or (not target) or (self.make_log28() < len(reach) + 4) or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            _seq127 = list(reach[:max(0, min(3, 6 - len(seen)))])
            _ix128 = 0
            while _ix128 < len(_seq127):
                ident = _seq127[_ix128]
                _ix128 = _ix128 + 1
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                try:
                    True if faults else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                reply = self.fill_row80(tool, args)
                try:
                    True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                self.pull_tip(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.seek_probe36(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Halo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        try:
            True if self.ref_values else (_ for _ in ()).throw(_Branch())
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        except _Branch:
            pass
        hold = bool(lost) and self.trail_blocks < (3 if self.make_log28() > 15 else 2) and (self.make_log28() > 5)
        try:
            True if hold else (_ for _ in ()).throw(_Branch())
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        except _Branch:
            pass
        return (text, hold)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        _seq129 = list(ids)
        _ix130 = 0
        while _ix130 < len(_seq129):
            ident = _seq129[_ix130]
            _ix130 = _ix130 + 1
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            try:
                True if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)) else (_ for _ in ()).throw(_Branch())
                out.append(ident)
            except _Branch:
                pass
        return out

    def trim_path(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            try:
                True if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)) else (_ for _ in ()).throw(_Branch())
                out.append(ident)
            except _Branch:
                pass
        return out

    def scan_key21(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.trim_path(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        _seq131 = list(self.trim_path(about))
        _ix132 = 0
        while _ix132 < len(_seq131):
            ident = _seq131[_ix132]
            _ix132 = _ix132 + 1
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            try:
                True if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)) else (_ for _ in ()).throw(_Branch())
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
            except _Branch:
                pass
        plan = self.fill_path73()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        else:
            try:
                True if plan and words else (_ for _ in ()).throw(_Branch())
                tool, base, field = plan
                faults: list[str] = []
                args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
                try:
                    True if not faults else (_ for _ in ()).throw(_Branch())
                    self.link_searches += 1
                    reply = self.fill_row80(tool, args)
                    try:
                        True if reply is not None else (_ for _ in ()).throw(_Branch())
                        observation = reply.get('observation')
                        self.pull_tip(tool, observation)
                        self.link_seen[memo] = [i for i, _ in self.note_ref(observation)]
                        known = self.shop_rows(about, self.link_seen[memo])
                        self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
                    except _Branch:
                        pass
                except _Branch:
                    pass
            except _Branch:
                pass
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _trim_log(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def do_slot(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def note_ledger(self) -> str:
        tool = self.do_slot()
        target = self.pack_gauge(tool)
        try:
            True if not tool or not target else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.trim_bit() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.make_log28() < len(stale) + 3:
            return ''
        lines = []
        _seq133 = list(stale)
        _ix134 = 0
        while _ix134 < len(_seq133):
            ident = _seq133[_ix134]
            _ix134 = _ix134 + 1
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            before = dict(self.rows.get(ident) or {})
            reply = self.fill_row80(tool, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.pull_tip(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            try:
                True if moved else (_ for _ in ()).throw(_Branch())
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_trim_log(_do_slot(observation, 300))[:900]}")
            except _Branch:
                pass
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def pick_frame(self) -> str:
        lost = [i for i in self.trim_bit() if i not in self.gone and self.sold_out(i)]
        try:
            True if not lost else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        _seq135 = list(lost)
        _ix136 = 0
        while _ix136 < len(_seq135):
            ident = _seq135[_ix136]
            _ix136 = _ix136 + 1
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.mark_set96(lost)
        text = f"[Halo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        try:
            True if picks and self.make_log28() <= 6 else (_ for _ in ()).throw(_Branch())
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        except _Branch:
            pass
        try:
            True if picks and self.brief.get('pick') == 'cheapest' else (_ for _ in ()).throw(_Branch())
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        except _Branch:
            pass
        try:
            True if picks else (_ for _ in ()).throw(_Branch())
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        except _Branch:
            pass
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        ids = [i for i in self.trim_bit() if i not in self.gone]
        try:
            True if any((self.sold_out(i) for i in ids)) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if len(ids) == 1 else (_ for _ in ()).throw(_Branch())
            self.unavailable.add(ids[0])
            return ''
        except _Branch:
            pass
        return self.note_ledger() if ids else ''

    def mark_set96(self, lost: list[str]) -> list[str]:
        try:
            True if not self.sort_slot() else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        records = []
        for ident in reversed(self.seen_order):
            try:
                True if ident in lost or ident in self.gone or self.sold_out(ident) else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            records.append(_trim_log(_do_slot(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            try:
                True if len(records) >= 25 else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        try:
            True if not records else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.fill_signal()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.pull_trace(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.scan_row38()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.sort_gauge(prompt, self.sort_slot()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        _seq137 = list(_seek_gate59((got or {}).get('fits')))
        _ix138 = 0
        while _ix138 < len(_seq137):
            entry = _seq137[_ix138]
            _ix138 = _ix138 + 1
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            try:
                True if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.trim_bit()) and (ident not in [i for _, i in fits]) else (_ for _ in ()).throw(_Branch())
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
            except _Branch:
                pass
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def decline_sweep(self) -> None:
        """Before the review of a decline, once and without the model: read the next page of the latest searches, then search
        each item's required values together, so that the review weighs what the store holds and not only what was seen.
        Of Mars v2's 43 wrong declines in race 18 another agent bought correctly on 16 tasks' episodes, and the listing it
        bought had been seen in only 4 of them."""
        if self.swept or self.audited or not self.roles.get('search'):
            return
        self.swept = True
        tool = self.roles['search']
        before = len(self.rows)
        pages = 0
        done: set[str] = set()
        for args, seen in reversed(list(self.search_log)):
            if pages >= SWEEP_PAGES or len(done) >= SWEEP_SEARCHES or self.finished:
                break
            plan = self.pull_chunk87(tool, seen)
            if not plan or plan[3] is None:
                continue
            key = _trim_log({k: v for k, v in args.items() if k != plan[2]})
            if key in done:
                continue
            done.add(key)
            for _ in range(2):
                if pages >= SWEEP_PAGES or self.make_log28() - self.tail() < SWEEP_ROOM or self.seek_frame() > WIND_DOWN or plan is None or plan[3] is None:
                    break
                faults: list[str] = []
                call = _shape({**args, plan[2]: plan[3]}, self.schemas.get(tool), 'args', faults)
                reply = None if faults else self.fill_row80(tool, call)
                if reply is None:
                    break
                pages += 1
                self.pull_tip(tool, reply.get('observation'))
                plan = self.pull_chunk87(tool, reply.get('observation'))
        searched = ''
        if self.seek_frame() < WIND_DOWN and not self.finished:
            searched = self.trim_anchor(list(range(1, len(self.items) + 1)))
        self.trace.append({'kind': 'decline_sweep', 'pages': pages, 'new_rows': len(self.rows) - before, 'searched': bool(searched)})

    def keep_frame(self) -> str:
        if self.audited or not self.sort_slot():
            return ''
        self.audited = True
        got = self.do_gate()
        named = self.take_ref(got)
        try:
            True if not named else (_ for _ in ()).throw(_Branch())
            rows = [str(i) for i in _seek_gate59((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            try:
                True if rows and self.run_tip(rows) else (_ for _ in ()).throw(_Branch())
                got = self.do_gate()
                named = self.take_ref(got)
            except _Branch:
                pass
        except _Branch:
            pass
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def do_gate(self) -> dict[str, Any] | None:
        records = []
        _seq139 = list(reversed(self.seen_order))
        _ix140 = 0
        while _ix140 < len(_seq139):
            ident = _seq139[_ix140]
            _ix140 = _ix140 + 1
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _trim_log(_do_slot(record, 200))[:1200] if record else 'ROW ' + _trim_log(_do_slot(self.rows.get(ident), 120))[:500])
            try:
                True if len(records) >= 30 else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.fill_signal()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        judges = self.sort_slot()
        got = self.sort_gauge(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and (not self.no_credit) and (not self.take_ref(got)):
            got = self.sort_gauge(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def take_ref(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _seek_gate59(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def run_tip(self, rows: list[str]) -> bool:
        many, joiner = (self.keep_filter(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.pack_gauge(tool)
        try:
            True if many and joiner and (not self.compare_bare()) else (_ for _ in ()).throw(_Branch())
            plan = [(joiner, {many: rows[:self.read_bundle()]})]
        except _Branch:
            try:
                True if tool and target else (_ for _ in ()).throw(_Branch())
                plan = [(tool, {target: i}) for i in rows[:2]]
            except _Branch:
                return False
        if self.make_log28() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return False
            except _Branch:
                pass
            reply = self.fill_row80(name, args)
            if reply is None or self.finished:
                break
            self.pull_tip(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def view_gauge93(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _read_sketch(spec) else None
            try:
                True if isinstance(shape, dict) and isinstance(shape.get('properties'), dict) else (_ for _ in ()).throw(_Branch())
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _read_sketch(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _read_sketch(s)), None)
                try:
                    True if not id_key else (_ for _ in ()).throw(_Branch())
                    return None
                except _Branch:
                    pass
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
            except _Branch:
                pass
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _read_sketch(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _read_sketch(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        try:
            True if not target else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        calls: list[dict[str, Any]] = []
        _seq141 = list(drop)
        _ix142 = 0
        while _ix142 < len(_seq141):
            ident = _seq141[_ix142]
            _ix142 = _ix142 + 1
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            else:
                try:
                    True if count and self.do_bag(name) else (_ for _ in ()).throw(_Branch())
                    calls.append({target: ident, count: 0})
                except _Branch:
                    try:
                        True if count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have) else (_ for _ in ()).throw(_Branch())
                        calls.append({target: ident, count: -have})
                    except _Branch:
                        return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.trim_bit() else 0
            if have == want:
                continue
            if have and (not self.do_bag(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def pull_probe(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        try:
            True if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.completed = True
        cat_key = self.sort_chunk89()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else 1 if len(self.items) == 1 else None
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = self.trim_bit()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        try:
            True if len(plan) + len({kind(i) for i in keep}) < len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        lines = {**keep, **plan}
        calls = self.view_gauge93(lines, drop)
        if not calls or len(calls) + 1 > self.make_log28():
            return ''
        self.make_log73(list(lines), len(calls) + 3)
        self.show_links(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.sort_chunk59(list(lines))):
            return ''
        done = []
        _seq143 = list(calls)
        _ix144 = 0
        while _ix144 < len(_seq143):
            args = _seq143[_ix144]
            _ix144 = _ix144 + 1
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return ''
            except _Branch:
                pass
            reply = self.fill_row80(cart, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                return ''
            except _Branch:
                pass
            self.get_gauge(cart, args)
            self.pull_tip(cart, reply.get('observation'))
            done.append(_trim_log(args))
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            try:
                True if attempt else (_ for _ in ()).throw(_Branch())
                self.note_ledger()
                self.check_cart(list(lines), SPEND_HARD)
                try:
                    True if not self.verdicts.get(self.sort_chunk59(list(lines))) or self.make_log28() < 1 or self.finished or any((self.sold_out(i) for i in lines)) else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
            except _Branch:
                pass
            go, said = self.hold_map37('order')
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
            reply = self.fill_row80(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _do_slot(reply, 200)})
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.pull_tip(buy, reply.get('observation'))
            replies.append(self.seek_probe36(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        try:
            True if not replies else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        return 'Not run (Halo acted instead): the independent check confirmed ' + self.confirmed + ', so Halo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def view_log(self, name: str, args: dict[str, Any]) -> bool:
        try:
            True if name != self.roles.get('buy') else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def hold_bit28(self, name: str, args: dict[str, Any]) -> bool:
        try:
            True if name != self.roles.get('refuse') else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def get_roster(self) -> list[dict[str, Any]]:
        left = self.make_log28()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        else:
            try:
                True if left <= len(self.items) + 5 and self.buys == 0 else (_ for _ in ()).throw(_Branch())
                keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
            except _Branch:
                pass
        tools = [t for t in self.tools if t['function']['name'] != ask]
        try:
            True if keep is not None and ends else (_ for _ in ()).throw(_Branch())
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        except _Branch:
            pass
        return tools

    def view_slot(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _read_sketch(spec) or 'enum' in spec:
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

    def fill_map(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _read_sketch(v) or 'array' in _read_sketch(v)) and ('enum' not in v)]
        try:
            True if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args) else (_ for _ in ()).throw(_Branch())
            return (args, '')
        except _Branch:
            pass

        def fix(value: Any) -> tuple[Any, str]:
            try:
                True if isinstance(value, str) and value.strip() in self.rows else (_ for _ in ()).throw(_Branch())
                return (value.strip(), '')
            except _Branch:
                pass
            try:
                True if not isinstance(value, str) or _send_row(value, True) or len(value.strip()) < 8 else (_ for _ in ()).throw(_Branch())
                return (value, '')
            except _Branch:
                pass
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            try:
                True if hits else (_ for _ in ()).throw(_Branch())
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            except _Branch:
                pass
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        _seq145 = list(free)
        _ix146 = 0
        while _ix146 < len(_seq145):
            key = _seq145[_ix146]
            _ix146 = _ix146 + 1
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

    def do_bag(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        try:
            True if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        return any((isinstance(s, dict) and 'integer' in _read_sketch(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _read_sketch(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and (count not in args) else args

    def get_gauge(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            try:
                True if isinstance(options, list) and len(options) >= 2 and (value in options[1:]) else (_ for _ in ()).throw(_Branch())
                removal = True
            except _Branch:
                pass
        loose_count = next((_note_piece(v) for v in args.values() if _note_piece(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            try:
                True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                _seq147 = list(value)
                _ix148 = 0
                while _ix148 < len(_seq147):
                    line = _seq147[_ix148]
                    _ix148 = _ix148 + 1
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_note_piece(v) for v in line.values() if _note_piece(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
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
        try:
            True if whole else (_ for _ in ()).throw(_Branch())
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        except _Branch:
            pass
        sets = self.do_bag(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            try:
                True if removal else (_ for _ in ()).throw(_Branch())
                have = have - (count or 1) if any(('integer' in _read_sketch(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
            except _Branch:
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

    def seek_probe36(self, observation: Any) -> str:
        text = _trim_log(_do_slot(observation, SHOW_CHARS))
        try:
            True if len(text) > OBS_LIMIT else (_ for _ in ()).throw(_Branch())
            text = text[:OBS_LIMIT] + '...(cut)'
        except _Branch:
            pass
        return text

    def keep_filter(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _read_sketch(v)), None)
        try:
            True if not target or any((k != target for k in schema.get('required') or [])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return target

    def read_bundle(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.keep_filter() or '') or {}
        try:
            True if isinstance(spec.get('maxItems'), int) else (_ for _ in ()).throw(_Branch())
            return spec['maxItems']
        except _Branch:
            pass
        try:
            True if self.compare_bad else (_ for _ in ()).throw(_Branch())
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        except _Branch:
            pass
        return max(4, self.compare_ok)

    def view_gate(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.pack_gauge(self.roles.get('open')))
        many = self.keep_filter()
        try:
            True if not opener or not joiner or (not target) or (not many) or (opener == joiner) else (_ for _ in ()).throw(_Branch())
            return calls
        except _Branch:
            pass
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _seek_sketch94(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            try:
                True if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone) else (_ for _ in ()).throw(_Branch())
                picks.append((index, ident))
            except _Branch:
                pass
        picks = picks[:self.read_bundle()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _trim_log({many: [i for _, i in picks]})}})
            else:
                try:
                    True if index in {i for i, _ in picks} else (_ for _ in ()).throw(_Branch())
                    out.append({'id': call['id'], 'joined': True})
                except _Branch:
                    out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def send_signal(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.view_gate(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.get_roster()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Halo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            try:
                True if self.finished else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run: the episode is over.'))
                continue
            except _Branch:
                pass
            try:
                True if name == self.roles.get('ask') else (_ for _ in ()).throw(_Branch())
                out.append((cid, "Not run (no step used): Halo's interviewer handles every message to the shopper."))
                continue
            except _Branch:
                pass
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
                args = _seek_sketch94(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            except _Branch:
                pass
            args, wrong = self.fill_map(name, args)
            try:
                True if wrong else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            except _Branch:
                pass
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            many = self.keep_filter() if name == self.roles.get('compare') else None
            try:
                True if many and isinstance(args.get(many), list) and (len(args[many]) > self.read_bundle()) and self.compare_bad else (_ for _ in ()).throw(_Branch())
                room = self.read_bundle()
                trimmed = f"\n[Halo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            except _Branch:
                pass
            target = self.pack_gauge(name)
            try:
                True if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            except _Branch:
                pass
            try:
                True if name == self.roles.get('cart') and any(((_note_piece(v) or 0) < 0 for v in args.values() if not isinstance(v, str))) else (_ for _ in ()).throw(_Branch())
                view = _trim_log(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                try:
                    True if absent else (_ for _ in ()).throw(_Branch())
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.scan_row38()))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            ending = self.view_log(name, args) or self.hold_bit28(name, args)
            try:
                True if ending and len(calls) > 1 else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            except _Branch:
                pass
            if not ending and self.make_log28() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _trim_log(args)
            try:
                True if not ending and name != self.roles.get('cart') and (mark in self.recent) else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            except _Branch:
                pass
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            try:
                True if name in readers and cached and (cached[0] == self.version) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            except _Branch:
                pass
            try:
                True if target and name in readers and (name != self.roles.get('compare')) and self.make_bundle(name, str(args.get(target))) else (_ for _ in ()).throw(_Branch())
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.pull_trace(str(args[target]))))
                continue
            except _Branch:
                pass
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.make_bundle(name, str(i))]
                try:
                    True if not keep else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.pull_trace(str(i))}' for i in args[many]))[:4000]))
                    continue
                except _Branch:
                    pass
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Halo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _trim_log(args)
            if ending and self.make_log28() > 2:
                if self.view_log(name, args):
                    wider = self.do_bundle()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.scan_row38()))
                        continue
                    reread = self.note_ledger()
                    try:
                        True if self.note_sold else (_ for _ in ()).throw(_Branch())
                        self.find_sold()
                    except _Branch:
                        pass
                    try:
                        True if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.make_log28() > 3) else (_ for _ in ()).throw(_Branch())
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.fill_row80(self.roles['cart_view'], view_args) if not faults else None
                        try:
                            True if view is not None else (_ for _ in ()).throw(_Branch())
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.pull_tip(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.scan_row38() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                        except _Branch:
                            pass
                    except _Branch:
                        pass
                    swap = self.pick_frame()
                    held = [i for i in self.trim_bit() if i in self.gone] if self.held < 2 else []
                    try:
                        True if swap or held else (_ for _ in ()).throw(_Branch())
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.scan_row38()))
                        continue
                    except _Branch:
                        pass
                    chain, hold = self.run_row()
                    try:
                        True if hold else (_ for _ in ()).throw(_Branch())
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.scan_row38()))
                        continue
                    except _Branch:
                        pass
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.make_log28() > self.tail() + 3 else []
                    try:
                        True if misfit else (_ for _ in ()).throw(_Branch())
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.scan_row38()))
                        continue
                    except _Branch:
                        pass
                    faults = self.keep_filter10() if self.line_holds < 2 and self.make_log28() > self.tail() + 2 else []
                    fixed = []
                    try:
                        True if faults and all((' is held at ' in f for f in faults)) else (_ for _ in ()).throw(_Branch())
                        fixed = self.fix_counts(self.make_path18())
                        faults = self.keep_filter10() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.scan_row38()))
                            continue
                    except _Branch:
                        pass
                    try:
                        True if faults else (_ for _ in ()).throw(_Branch())
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.scan_row38()))
                        continue
                    except _Branch:
                        pass
                    lines = self.trim_bit()
                    read = self.make_log73(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Halo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.scan_row38()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Halo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.make_log28() > 5:
                    late = self.pull_cue('end') if self.msg_used < self.msg_limit else ''
                    if late:
                        out.append((cid, 'Not run yet. ' + late))
                        continue
                    self.decline_sweep()
                    review = self.keep_frame()
                    try:
                        True if review else (_ for _ in ()).throw(_Branch())
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    except _Branch:
                        pass
                    try:
                        True if self.leads and self.make_log28() > 4 and (self.refuse_blocks == 0) else (_ for _ in ()).throw(_Branch())
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    except _Branch:
                        pass
                    if self.leads and self.make_log28() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        try:
                            True if self.verdicts.get(self.sort_chunk59([first])) and first in self.leads else (_ for _ in ()).throw(_Branch())
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
                        except _Branch:
                            pass
            try:
                True if self.hold_bit28(name, args) and self.make_log28() <= 5 and (not self.late_held) else (_ for _ in ()).throw(_Branch())
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.trim_bit() or (i in self.dropped and (not fresh))]
                try:
                    True if len(lines) >= len(self.items) and len(missing) + 1 <= self.make_log28() else (_ for _ in ()).throw(_Branch())
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if self.hold_bit28(name, args) and (not self.leads) and (not self.pushed) and (self.make_log28() > self.max_steps // 3) else (_ for _ in ()).throw(_Branch())
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.make_log28()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            except _Branch:
                pass
            try:
                True if self.hold_bit28(name, args) else (_ for _ in ()).throw(_Branch())
                done = self.pull_probe()
                try:
                    True if done else (_ for _ in ()).throw(_Branch())
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                except _Branch:
                    pass
                args = self.view_slot(name, args)
            except _Branch:
                pass
            try:
                True if self.view_log(name, args) else (_ for _ in ()).throw(_Branch())
                late = self.pull_cue()
                try:
                    True if late else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.scan_row38()))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if ending and self.make_log28() >= 2 else (_ for _ in ()).throw(_Branch())
                go, said = self.hold_map37('order' if self.view_log(name, args) else 'end')
                if go and self.view_log(name, args):
                    late = self.pull_cue()
                    if late:
                        out.append((cid, 'Not run yet. ' + (said + '\n' if said else '') + late + '\nCart now: ' + self.scan_row38()))
                        continue
                try:
                    True if not go else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.scan_row38()))
                    continue
                except _Branch:
                    pass
                if said:
                    trimmed = '\n[Halo] ' + said + trimmed
            except _Branch:
                pass
            reply = self.fill_row80(name, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            except _Branch:
                pass
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _trim_log(observation)
                try:
                    True if not any((str(i) in shown for i in args[many])) else (_ for _ in ()).throw(_Branch())
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Halo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.read_bundle()} per call.'
                except _Branch:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.get_probe(observation)
            try:
                True if name == self.roles.get('compare') else (_ for _ in ()).throw(_Branch())
                rows = self.note_ref(observation)
                try:
                    True if rows else (_ for _ in ()).throw(_Branch())
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_make_probe(r) for _, r in rows))
                except _Branch:
                    pass
            except _Branch:
                pass
            try:
                True if name in readers - {self.roles.get('search')} else (_ for _ in ()).throw(_Branch())
                self.note_view(name, observation)
            except _Branch:
                pass
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.pack_gauge(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                try:
                    True if listing else (_ for _ in ()).throw(_Branch())
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_make_probe(observation))
                    try:
                        True if name == self.roles.get('open') and (not self.facts_from[name]) else (_ for _ in ()).throw(_Branch())
                        other = next((t for t in self.schemas if t != name and self.pack_gauge(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        try:
                            True if other else (_ for _ in ()).throw(_Branch())
                            self.roles['open'] = other
                        except _Branch:
                            pass
                    except _Branch:
                        pass
                    try:
                        True if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')) else (_ for _ in ()).throw(_Branch())
                        self.roles['offer'] = name
                    except _Branch:
                        pass
                    try:
                        True if name == self.do_slot() else (_ for _ in ()).throw(_Branch())
                        self.offer_seen[listing] = self.step
                    except _Branch:
                        pass
                except _Branch:
                    pass
            try:
                True if name == self.roles.get('cart') else (_ for _ in ()).throw(_Branch())
                self.get_gauge(name, args)
                self.cart_said = _trim_log(observation)
                self.recent = []
                counts = [_note_piece(v) for v in args.values() if not isinstance(v, str) and _note_piece(v) is not None]
                _seq149 = list(args.values())
                _ix150 = 0
                while _ix150 < len(_seq149):
                    value = _seq149[_ix150]
                    _ix150 = _ix150 + 1
                    try:
                        True if counts and isinstance(value, str) and (value in self.rows) else (_ for _ in ()).throw(_Branch())
                        try:
                            True if counts[0] <= 0 and (not _in_rows(observation, value)) else (_ for _ in ()).throw(_Branch())
                            self.dropped.add(value)
                        except _Branch:
                            try:
                                True if counts[0] > 0 else (_ for _ in ()).throw(_Branch())
                                self.dropped.discard(value)
                            except _Branch:
                                pass
                    except _Branch:
                        pass
                lines = [i for i in self.trim_bit() if not self.sold_out(i) and i not in self.dropped]
                try:
                    True if len(lines) >= len(self.built) else (_ for _ in ()).throw(_Branch())
                    self.built = lines
                except _Branch:
                    pass
            except _Branch:
                pass
            if name == self.roles.get('search') and self.note_ref(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            try:
                True if self.view_log(name, args) else (_ for _ in ()).throw(_Branch())
                self.buys += 1
                try:
                    True if not reply.get('finished') else (_ for _ in ()).throw(_Branch())
                    self.refused_at = self.step
                except _Branch:
                    pass
            except _Branch:
                pass
            text = self.seek_probe36(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.mark_sketch10([observation] + self.paged_obs) + self.run_trace(name, args, observation)
            base = text
            proof = self.evidence(observation)
            try:
                True if proof else (_ for _ in ()).throw(_Branch())
                text += '\n' + proof
            except _Branch:
                pass
            try:
                True if name == self.roles.get('cart') and any((_note_piece(v) == 0 for v in args.values() if not isinstance(v, str))) else (_ for _ in ()).throw(_Branch())
                try:
                    True if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())) else (_ for _ in ()).throw(_Branch())
                    text += '\n[Halo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
                except _Branch:
                    pass
            except _Branch:
                pass
            self.note_sold = False
            lines = self.scan_cap(observation)
            try:
                True if lines else (_ for _ in ()).throw(_Branch())
                meaning = self.do_log(lines)
                try:
                    True if meaning else (_ for _ in ()).throw(_Branch())
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
                except _Branch:
                    pass
            except _Branch:
                pass
            refused = self.view_log(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.read_chunk(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.view_log(name, args))))
            if (loose or again) and self.trim_bit() and (not self.finished):
                seen = self.find_sold()
                try:
                    True if seen else (_ for _ in ()).throw(_Branch())
                    text += "\n[Halo re-read the cart's lines after the refusal]\n" + seen
                except _Branch:
                    pass
            self.note_sold = False
            swap = self.pick_frame()
            try:
                True if swap else (_ for _ in ()).throw(_Branch())
                text += '\n' + swap
            except _Branch:
                pass
            if self.queued:
                text += "\n[notes seen on Halo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def pull_layer(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.view_slot(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        try:
            True if self.finished or self.broke or self.make_log28() <= 0 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.trim_bit()) if fresh else bool(self.cart_lines)
        whole = True
        try:
            True if holding and len(self.items) >= 2 else (_ for _ in ()).throw(_Branch())
            whole = self.fill_short_cart() or not self.pull_layer()
            if whole and len(self.trim_bit()) >= 2:
                self.make_log73(self.trim_bit(), self.tail())
                try:
                    True if self.make_log28() >= 3 and self.pick_layer() > 5 else (_ for _ in ()).throw(_Branch())
                    self.show_links(self.trim_bit(), self.tail())
                except _Branch:
                    pass
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.trim_bit()})
        except _Branch:
            pass
        try:
            True if self.finished or self.make_log28() <= 0 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.make_log28() >= 2))) else (_ for _ in ()).throw(_Branch())
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                try:
                    True if self.make_log28() >= 2 else (_ for _ in ()).throw(_Branch())
                    self.hold_map37('order')
                except _Branch:
                    pass
                try:
                    True if self.finished or self.make_log28() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                reply = self.fill_row80(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _do_slot(reply, 200)})
                try:
                    True if self.finished or self.make_log28() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.pull_tip(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.seek_probe36(reply.get('observation'))
            except _Branch:
                pass
        except _Branch:
            pass
        try:
            True if refuse else (_ for _ in ()).throw(_Branch())
            faults = []
            args = _shape(self.view_slot(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                try:
                    True if self.make_log28() >= 2 else (_ for _ in ()).throw(_Branch())
                    self.hold_map37('end')
                except _Branch:
                    pass
                try:
                    True if self.finished or self.make_log28() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                reply = self.fill_row80(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _do_slot(reply, 200)})
                if reply is not None and (not self.finished):
                    self.pull_tip(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.seek_probe36(reply.get('observation'))
            except _Branch:
                pass
        except _Branch:
            pass
        return ''

    def scan_chunk(self) -> None:
        size = sum((len(_trim_log(m.get('content'))) for m in self.chat))
        try:
            True if size <= HISTORY_LIMIT else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        _seq151 = list(tools[:max(1, len(tools) // 2)])
        _ix152 = 0
        while _ix152 < len(_seq151):
            index = _seq151[_ix152]
            _ix152 = _ix152 + 1
            content = self.chat[index].get('content') or ''
            try:
                True if len(content) > 600 else (_ for _ in ()).throw(_Branch())
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"
            except _Branch:
                pass

    def hold_ref(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.keep_row61()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        try:
            True if opening is not None else (_ for _ in ()).throw(_Branch())
            self.get_probe(opening)
        except _Branch:
            pass
        intro = "SHOPPER'S REQUEST:\n" + self.request
        try:
            True if opening is not None else (_ for _ in ()).throw(_Branch())
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.seek_probe36(opening)
        except _Branch:
            pass
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        try:
            True if not self.finished else (_ for _ in ()).throw(_Branch())
            turned = self.finish(again=True)
            try:
                True if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.make_log28() > 1) else (_ for _ in ()).throw(_Branch())
                self.chat.append({'role': 'user', 'content': '[Halo] Halo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            except _Branch:
                pass
            try:
                True if not self.finished else (_ for _ in ()).throw(_Branch())
                self.finish()
            except _Branch:
                pass
        except _Branch:
            pass
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.make_log28() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.seek_frame() < LAST_TURN or (not self.trim_bit() and self.pick_layer() > EMPTY_ROOM)):
            try:
                True if self.send_cue79() else (_ for _ in ()).throw(_Branch())
                summary = self.take_key57()
                try:
                    True if summary else (_ for _ in ()).throw(_Branch())
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                except _Branch:
                    pass
                continue
            except _Branch:
                pass
            self.scan_chunk()
            self.loop_calls += 1
            message = self.run_tag(self.chat, [LIGHT, SPARE], 3000, self.get_roster())
            try:
                True if message is None else (_ for _ in ()).throw(_Branch())
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            except _Branch:
                pass
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                try:
                    True if not isinstance(call, dict) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _seek_sketch94(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _trim_log(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _trim_log(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                try:
                    True if idle >= 3 else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            try:
                True if (not self.asked or self.more_to_ask()) and self.roles.get('ask') and (self.view_chunk() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.make_log28() > 8) and (self.seek_frame() <= ASK_BY) else (_ for _ in ()).throw(_Branch())
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                _seq153 = list(calls)
                _ix154 = 0
                while _ix154 < len(_seq153):
                    call = _seq153[_ix154]
                    _ix154 = _ix154 + 1
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.take_key57()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            except _Branch:
                pass
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.send_signal(calls)
            for index, (cid, text) in enumerate(replies):
                try:
                    True if index == len(replies) - 1 else (_ for _ in ()).throw(_Branch())
                    text += '\n' + self.status()
                except _Branch:
                    pass
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Halo(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.hold_ref()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            try:
                True if not episode.finished else (_ for _ in ()).throw(_Branch())
                episode.finish()
            except _Branch:
                pass
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']