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
AUTO_PAGES = 10
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
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Halo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first: the wanted value plus a short noun for the product, not the full name of its category, which every row on that shelf shares anyway.\n- The clock is tighter than the step count: every reply you write uses several seconds of a fixed episode time, while a tool call uses one step. Calls that do not wait on each other\'s results (searches for different items, opening the listings you are comparing, a page plus the next search) go out together in one reply; only a call that needs an earlier result waits for it. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Halo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Halo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Halo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Halo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Halo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Halo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule, and neither is a word on what to offer when nothing fits fully (the nearest listing, or one within some margin of the cheapest such listing).\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, except under a key that names choices on offer (options, variants, a selection to make) or when the listing\'s own option names a different element of that list, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); a set of alternatives the buyer would pick one from (sizes or counts to choose, often under a plural key or joined by "or") is an offer and states none of them; a word attached to the value that turns it into another model, grade or version (a qualifier in front of a name) is a different value, not the required one; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, except under a key that names choices on offer (options, variants, a selection to make) or when the listing\'s own option names a different element of that list, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a set of alternatives the buyer would pick one from (sizes or counts to choose, often under a plural key or joined by "or") is an offer and states none of them; a word attached to the value that turns it into another model, grade or version (a qualifier in front of a name) is a different value, not the required one; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nSome pages name a partner for their own value: a value they say goes with, pairs with or mates with it. When the request asks for that partner (the piece that goes with or mates with the pointed value) rather than the very same value, and the last page names one, "value" is that partner exactly as the page writes it, not the page\'s own value.\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|(?<!compatible )model|(?<!compatible )models|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{1,4}[Xx]{1,3}|\\d{3,6})'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [_code_norm(c) for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes = []
    for hit in re.finditer(MODEL_TAG + '(' + FIT_CODE + ')\\b', text):
        codes.extend(_codes(hit.group(1)))
    for hit in re.finditer(SOLD_AS, text, re.I):
        codes.extend(_code_run(hit.group(1))[0])
    works = []
    nots = []
    for hit in re.finditer(FIT_WORDS, text):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            yes, no = _code_run(text[hit.end():hit.end() + 300])
            works += yes
            nots += no
    also = [(hit.group(1).upper(), _code_run(hit.group(2))[0]) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;|]+)', text, re.I)]
    family = re.search('[\\[(]\\s*(' + SERIES_MARK + ')-\\d{2,6}\\s*[\\])]', text)
    takes = {hit.group(1).upper() for hit in re.finditer('\\bseries\\s*:?\\s*(' + SERIES_MARK + ')\\b', text, re.I)}
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    found = {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}
    found['nots'] = nots
    found['family'] = family.group(1).upper() if family else ''
    found['takes'] = takes
    return found

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

OFFER_KEY = '^(?:.*\\s)?(?:options?|variants?|choices?)$|^(?:select|choose)\\b'
OFFER_TAG = ' [offered choices, none of them stated]'


def _typed(text: str) -> Any:
    text = text.strip()
    if text in ('True', 'False'):
        return text == 'True'
    if text == 'None':
        return None
    if re.fullmatch('-?\\d+', text):
        return int(text)
    if re.fullmatch('-?\\d+\\.\\d+', text):
        return float(text)
    return text


def _offer_key(key: str) -> str:
    if OFFER_TAG in key or not re.match(OFFER_KEY, _words(key)):
        return key
    return key + OFFER_TAG


def _is_offer(key: Any) -> bool:
    return OFFER_TAG in str(key) or bool(re.match(OFFER_KEY, _words(str(key))))


def _pipe_record(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    fields = []
    for piece in re.split('\\s+\\|\\s+', text.strip()):
        head = re.match("\\s*([A-Za-z][\\w .'/-]{0,40}?)\\s*:(.*)$", piece, re.S)
        if head and head.group(1) not in [f[0] for f in fields]:
            fields.append([head.group(1), head.group(2).strip()])
        elif fields:
            fields[-1][1] += ' | ' + piece
        else:
            return None
    if len(fields) < 3:
        return None
    record = {}
    for key, raw in fields:
        pairs = [p.partition('=') for p in re.split(';\\s*', raw) if p.strip()]
        if pairs and all((eq and 0 < len(k.strip()) <= 40 for k, eq, _ in pairs)):
            record[_offer_key(key)] = {_offer_key(k.strip()): _typed(v) for k, _, v in pairs}
        else:
            record[_offer_key(key)] = _typed(raw)
    if not any((isinstance(v, str) and _id_like(v) and re.search('[A-Za-z]', v) for v in record.values())):
        return None
    return record


def _text_shape(text: str) -> Any:
    body = text.strip()
    if body[:1] in '{[' and body[-1:] in '}]':
        try:
            return json.loads(body)
        except ValueError:
            return text
    rows = [x.strip() for x in body.splitlines() if x.strip()]
    keyed = [x for x in rows if re.match("[A-Za-z][\\w .'/-]{0,40}:\\s", x)]
    heads = [x.split(':', 1)[0].strip().lower() for x in keyed]
    if len(rows) >= 3 and len(keyed) >= max(3, len(rows) - 1) and len(set(heads)) == len(heads):
        return ' | '.join(rows)
    return text


def _take_text_rows(node: Any, depth: int = 0) -> Any:
    if depth > 6:
        return node
    if isinstance(node, str) and depth == 0 and node.lstrip()[:1] in ('{', '['):
        shaped = _text_shape(node)
        node = node if isinstance(shaped, str) else shaped

    def record_of(value: Any, lined: bool) -> dict[str, Any] | None:
        if not isinstance(value, str):
            return None
        if lined and '\n' in value and value.strip()[:1] not in '{[':
            joined = _text_shape(value)
            value = joined if isinstance(joined, str) else value
        return _pipe_record(value)
    if isinstance(node, list):
        return [record_of(v, True) or _take_text_rows(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            lone = record_of(next(iter(node.values())), False)
            if lone:
                return lone
        return {k: record_of(v, False) or _take_text_rows(v, depth + 1) for k, v in node.items()}
    if depth == 0:
        return record_of(node, True) or node
    return node


def _facts_any(node: Any) -> dict[str, Any] | None:
    block = _fact_block(node)
    if block or not isinstance(node, dict):
        return block
    for value in node.values():
        if isinstance(value, str) and value.count('=') >= 2:
            pairs = [(k.strip(), v.strip()) for k, eq, v in (p.partition('=') for p in value.split(';')) if eq and 0 < len(k.strip()) <= 40]
            if len(pairs) >= 2:
                return dict(pairs)
    return None

LIGHT_WORDS = ('the', 'a', 'an', 'of', 'for', 'with', 'and')
MODEL_CODE = '\\b(?:[A-Za-z]{1,6}-?\\d{3,6}|[A-Z][a-z]{2,6} \\d{3})\\b'


def _singular(word: str) -> str:
    low = word.lower()
    if len(low) > 4 and low.endswith('ies'):
        return word[:-3] + 'y'
    if len(low) > 4 and re.search('(?:ch|sh|ss|x|z)es$', low):
        return word[:-2]
    if len(low) > 3 and low.endswith('s') and not re.search('(?:ss|us|is|ves|ics)$', low):
        return word[:-1]
    return word


def _token_set(text: Any) -> set[str]:
    return {w for w in re.findall('[a-z0-9]+', str(text or '').lower()) if w not in LIGHT_WORDS}


def _short_terms(value: str) -> list[str]:
    words = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'.+/-]*", str(value)) if w.lower() not in LIGHT_WORDS]
    return words if len(words) <= 3 else []


def _same_words(a: list[str], b: list[str]) -> bool:
    return [w.lower() for w in a] == [w.lower() for w in b]

AT_MOST = 'maximum|at most|or fewer|up to|or under|max|or less|or below|or lower|and under|and below|not more than|no more than'
AT_LEAST = 'at least|no less than|not less than|minimum|min|or more|or higher|or above|or over|and up|and over|and above'
BELOW = 'under|below|less than|fewer than|lower than|smaller than'
ABOVE = 'over|above|more than|greater than|higher than|bigger than|larger than'
FIGURE = '(\\d[\\d,]*(?:\\.\\d+)?)\\s*([a-z"]*)'
LOOK_TALK = '\\b(?:search\\w*|look(?:ing)? (?:for|up|into|through|at)|find\\w*|open\\w*|brows\\w*|next page|page (?:on|through)|check(?:ing)? (?!out\\b)\\w+)'


def _bound(value: str) -> tuple[bool, bool, float, str] | None:
    low = str(value or '').lower()
    for words, upper, strict in ((AT_MOST, True, False), (AT_LEAST, False, False), (BELOW, True, True), (ABOVE, False, True)):
        found = re.search('\\b(?:' + words + ')\\s*:?\\s*' + FIGURE, low) or re.search(FIGURE + '[\\s,]*\\(?(?:' + words + ')\\b', low)
        if found:
            try:
                figure = float(found.group(1).replace(',', ''))
                return (upper, strict, figure, found.group(2))
            except ValueError:
                return None
    return None


def _unit_like(got: str, unit: str) -> bool:
    if not got and not unit:
        return True
    if not got or not unit:
        return False
    if got == unit:
        return True
    short, long = sorted((got, unit), key=len)
    return len(short) >= 2 and long.startswith(short)


def _word_bag(text: Any) -> set[str]:
    return set(_spelled(re.sub('[,/;|()-]', ' ', str(text))))


def _chosen_other(facts: dict[str, Any], held: Any, value: str) -> str:
    listed = [p for p in (_words(x) for x in re.split('[,/;|]', str(held))) if p]
    if len(listed) < 2:
        return ''
    chosen = [p for k, v in facts.items() if _is_offer(k) for p in (_words(x) for x in re.split('[,/;|]', str(v))) if p]
    wanted = [w for w in (_words(o) for o in re.split(',\\s*|\\s+or\\s+', str(value).strip().lower())) if w]
    if not chosen or any(w in chosen for w in wanted):
        return ''
    return next((p for p in chosen if p in listed), '')

SERIES_MARK = '[A-Z]{1,4}\\d{1,3}'
FIT_WORDS = '(?i:(?<!not )(?<!never )(?<!in)(?:\\b(?:fits?|suits?)|\\b(?:goes|go|works?|pairs?|compatible|for use) with))\\b\\s*:?\\s*'
CODE_JOIN = '(?i:every|all|any|the|models?|series|and|or|plus|also)\\b|[,/&()]'
CODE_EXCEPT = '(?i:except|excluding|but not|other than|apart from)\\b'


def _code_norm(code: str) -> str:
    return code.lower().replace('-', '').replace(' ', '')


def _code_run(clause: str) -> tuple[list[str], list[str]]:
    yes = []
    no = []
    side, pos = yes, 0
    while pos < len(clause):
        hit = re.match('\\s*(?:(?P<code>' + FIT_CODE + ')\\b|(?P<but>' + CODE_EXCEPT + ')|(?:' + CODE_JOIN + '))', clause[pos:])
        if not hit or not hit.group(0).strip():
            break
        if hit.group('code'):
            side.append(_code_norm(hit.group('code')))
        elif hit.group('but'):
            side = no
        pos += hit.end()
    return (yes, no)


def _code_hit(fit: str, code: str) -> bool:
    if fit == code:
        return True
    family = fit.rstrip('x')
    if family == fit or not family[-1:].isdigit() or len(fit) != len(code):
        return False
    return code.startswith(family) and code[len(family):].isdigit()
MODEL_TAG = '(?i:\\bmodel)(?:\\s*(?i:no|number|nr|ref|reference|code|id)\\b\\.?|\\s*#)?\\s*[:.#]?\\s*'
SOLD_AS = '\\b(?:also sold as|also known as|sold also as)\\s+([^.;|]+)'

LIMIT_WORD = '(?:not?|least|most|under|over|unlike|different|min(?:imum)?|max(?:imum)?)'
LINK_FILLER = '(?:the|and|with|from|that|comes|listing|page|its|for|of)'


class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told = []
        self.dropped = []
        self.free = []

class Halo:

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
        self.paid_misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace = []
        self.brief = {}
        self.roles = {}
        self.items = []
        self.budget = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok = set()
        self.done_log = []
        self.done_steps = []
        self.asked = False
        self.qa = []
        self.rows = {}
        self.full = {}
        self.seen_order = []
        self.heard = set()
        self.notes = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view = None
        self.note_named = False
        self.cart_lines = {}
        self.checked = set()
        self.audited = False
        self.offer_seen = {}
        self.gone = {}
        self.unavailable = set()
        self.sold_lines = set()
        self.note_sold = False
        self.held = 0
        self.id_args = set()
        self.cart_said = ''
        self.refused_at = -1
        self.leads = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built = []
        self.dropped = set()
        self.late_held = False
        self.lead_items = {}
        self.verdicts = {}
        self.confirmed = None
        self.completed = False
        self.link_tried = set()
        self.lane_misses = {}
        self.adopted = {}
        self.capped = set()
        self.id_key = None
        self.facts_from = {}
        self.queued = []
        self.buys = 0
        self.refusals = 0
        self.recent = []
        self.chat = []
        self.opened = set()
        self.refs = []
        self.ref_values = {}
        self.trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys = {}
        self.view_keys = {}
        self.read_cache = {}
        self.pages_read = 0
        self.shapes = {}
        self.price_key = None
        self.stock_key = None
        self.cheaper_marks = set()
        self.wider_held = False
        self.paged_obs = []
        self.plan_order = []
        self.lanes = {}
        self.lane_found = {}
        self.lane_spent = {}
        self.lane_used = {}
        self.late_asked = False
        self.one_product = set()
        self.follow_up = []
        self.listing = True
        self.line_holds = 0
        self.bound = []
        self.fit_holds = 0
        self.searched = []
        self.link_searches = 0
        self.word_blind = {}
        self.blind_notes = 0
        self.blind_log = []
        self.echo_log = []
        self.lead_count = None
        self.echo_hits = 0
        self.stopped_lists = set()
        self.store_opened = False
        self.values_sought = False
        self.code_tries = [0, 0]
        self.pages_seen = set()
        self.early_walks = 0
        self.early_at = -99
        self.walk_rounds = 0
        self.chain_rows = set()
        self.link_skips = {}
        self.resolved_done = set()
        self.unread_spent = 0
        self.unread_ended = False
        self.fill_opened = False
        self.gap_done = False
        self.secured = False
        self.look_now = False
        self.look_turns = 0
        self.basket_at = -1
        self.basket_needs = ()
        self.last_note_at = -1
        self.views_of = {}
        self.found_by = {}
        self.longer_held = False
        self.other_held = False
        self.attr_held = False
        self.end_refused_at = -1
        self.redone = False
        self.note_held = False
        self.cart_at_refusal = ''
        self.fit_cache = (None, None)
        self.series_whole = set()
        self.fit_fixed = False
        self.series_offered = False
        self.series_notes = 0
        self.link_seen = {}

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
                    self.paid_misses = 0
                    return message
                passing = status in (None, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.paid_misses += 1
                if self.paid_misses >= 2:
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
        body = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
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
            for again in (False, True):
                message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
                got = _json_obj((message or {}).get('content'))
                if got is not None:
                    return got
                if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                    return None
                blank = message is None and bool(self.trace) and self.trace[-1].get('kind') == 'llm_fail' and self.trace[-1].get('status') == 0
                if again or not blank or self.model_room() < 2 * CALL_FLOOR:
                    break
                self.trace.append({'kind': 'blank_reply_again', 'model': model})
                got = self.plain_json(model, prompt, max_tokens)
                if got is not None:
                    return got
                break
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        self.pages_seen.add(self.page_sig(name, args, self.page_arg(args)))
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    data['observation'] = _take_text_rows(data['observation'])
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                self.note_reading(name, args, seen)
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
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
        merged = []
        became = {}
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
        keys = set()
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
            when = {'choose': 'before choosing', 'order': 'at the order (Halo does it when you check out)', 'end': 'at an ending without a purchase (Halo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Halo] step {self.step} of {self.max_steps} used, {left} left.'
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
        found = set()

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
        counts = {}
        present = {}
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
        groups = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done = set()
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
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept = []
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
        groups = {}
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
            found = []
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
        if name != self.roles.get('search'):
            return ''
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
        if self.wordless(name, args):
            return f"\n[Halo] This list matched {count} listings and showed {shown}. Halo pages on by itself only for a search carrying a required value: search the item's value with its product noun, or page on with {param}={_dump(value)}."
        allowance = self.page_allowance()
        pages = -(-(count - shown) // shown)
        if pages > 4 and self.pages_read + 2 <= allowance and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if pages > 4 or self.pages_read + pages > allowance or self.left() - self.tail() - pages < 12:
            return f"\n[Halo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        last_page = None
        for _ in range(pages):
            faults = []
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
            last_page = (call, seen)
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Halo] {count} listings matched and Halo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        if last_page and value is None:
            rest += self.list_stop(name, last_page[0], last_page[1])
        return '\n[Halo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

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
        return f'\n[Halo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Halo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

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
            faults = []
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
            faults = []
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
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and self.cap_named(k) and not re.search(CAP_NOT, k, re.I)]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(SHELF_ARG, _snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            base = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and not any(k != shelves[0] and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                if not named and len(options) == 2:
                    unread = [o for o in options if not (isinstance(o, str) and re.search(QUERY_WORDS, o, re.I))]
                    named = unread if len(unread) == 1 else named
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
            faults = []
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
        picks = []
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
        picks = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults = []
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
        lanes = []
        heads = []
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
        args = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults = []
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
        sent = []
        told = []
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
        summary = "Halo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
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
        told = []
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
        summary = f'Halo asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
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
                        mode = self.reading_mode(tool, k)
                        if mode is not None:
                            keep[k] = mode
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        out = []
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if n != number or ':' not in need or need in [m for _, m in self.refs] or self.fit_need(n, need):
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

    def value_search(self, numbers: list[int], from_request: bool=False) -> str:
        plan = self.value_template()
        if not plan or self.left() < 22:
            return ''
        tool, shared_args, field = plan
        seen = []
        budget = max(self.max_steps // 6, 8)
        width = self.query_width(tool, field)
        shelf = self.shelf_field(tool)
        shelf_needed = shelf in ((self.schemas.get(tool) or {}).get('required') or [])
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
            label = self.shelf_value(tool, number) if shelf else None
            base = dict(shared_args)
            if shelf in base and (label or not shelf_needed):
                del base[shelf]
            if label:
                base[shelf] = label
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if from_request or told]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [[values[0], item.name], values] if lead else [values]
            if from_request:
                terms = []
                for v in values:
                    terms.extend(_short_terms(v) or v.split())
                together = list(dict.fromkeys(terms)) + self.short_noun(number)[0]
                if len(together) in (3, 4, 5) and len(' '.join(together)) <= width:
                    if together in tries:
                        tries.remove(together)
                    tries.insert(0, together)
            for words in tries[:2]:
                if not words:
                    continue
                faults = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                self.echo_note(tool, args, observation)
                pages, spent = self.more_value_pages(tool, args, observation, number, budget)
                budget -= spent
                cat_key = self.category_key()
                found = []
                for one in pages:
                    found.extend(i for i, _ in self.records_in(one) if i not in found)
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                if self.brief.get('pick') == 'cheapest':
                    rows.sort(key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0))
                home = self.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                first_page = self.paging(tool, observation)
                about = f' ({first_page[0]} matches, {len(pages)} page(s) read)' if first_page else ''
                seen.append(_dump(args.get(field)) + about + ': ' + (', '.join(rows[:6]) if rows else 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Halo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts = {}
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
        seen = {}
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
        lanes = {n: [] for n in range(1, len(self.items) + 1)}
        hinted = {n: set() for n in lanes}
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
            own = {}
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
        sent = []
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
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Halo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
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
        return (True, 'Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found = []

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
        lines = []
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
        added = set()
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
            self.last_note_at = self.step
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
        ids = []

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
        return "lines by Halo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack = [self.cart_view]
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
        marks = {}
        notes = {}
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

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally = {}
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
            if not words or any(re.fullmatch(LIMIT_WORD, w) for w in words):
                return False
            if not any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def fill_short_cart(self) -> bool:
        return self.fill_missing()

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
        faults = []
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
        calls = []
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
            faults = []
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

    def trail(self, early: bool=False) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen = []
        lost = []
        floor = self.chain_reserve() if early else 4
        if early:
            limit = self.trails + 3
        else:
            limit = self.walk_rounds + (6 if self.left() > 15 else 4)
        while pending and self.cost < SPEND_HARD and self.trails < limit:
            self.trails += 1
            if early:
                self.walk_rounds += 1
            records = self.trail_records()
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + self.request_quotes(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach = []
            lost = []
            gaps = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                self.chain_rows.update(i for i in ids if i in self.rows)
                skips = self.link_skips.get(need, 0)
                better = self.better_starts(need, ids[0]) if skips < 2 and ids and ids[0] in self.opened else []
                if better:
                    self.link_skips[need] = skips + 1
                    for ident in better[:2]:
                        if ident not in reach:
                            reach.append(ident)
                    lost.append(f"{need} -> the opened {ids[0]} misses values the request names for the first page; {' and '.join(better[:2])} from the same shop fit that description better")
                    self.trace.append({'kind': 'first_link_swapped', 'need': need[:120], 'page': ids[0], 'next': better[:2]})
                    continue
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
            if not reach and gaps and tool and target and (self.left() > max(12, floor)) and (self.link_searches < self.link_room()):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if self.finished or not (reach and tool and target) or self.left() < floor + len(reach):
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                if tool == self.offer_tool():
                    self.offer_seen[ident] = self.step
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Halo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        if early:
            if lost:
                text += "[Halo] A requirement takes its value from a chain of pages, and the order counts only once every page of that chain was opened. Still to find and open: " + ' | '.join(lost[:3]) + ". Search for it with its shop's name and the values its description gives, then open it.\n"
            return (text, False)
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
        known = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if (len(w) >= 3 or w.isdigit()) and not re.fullmatch(LINK_FILLER, w)} - named
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
        known, wanted = self.link_lookup(about, words)

        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower()) if not re.fullmatch(LINK_FILLER, w)} - named

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        pool = of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))) or known
        return sorted(pool, key=lambda i: len(self.page_lacks(i, wanted)))[:3]

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
            faults = []
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
        text = f"[Halo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        fits = []
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
            faults = []
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
        calls = []
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
        plan = {}
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
            faults = []
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
        return 'Not run (Halo acted instead): the independent check confirmed ' + self.confirmed + ', so Halo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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
        keep = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        lines_short = self.lines_short()
        if left <= self.tail():
            keep = set(ends)
        elif self.look_now and left >= self.tail() + 2 + lines_short or (lines_short and left >= 2 * lines_short + self.tail()):
            keep = None
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
        lines = []
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
        picks = []
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
        out = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Halo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Halo's interviewer handles every message to the shopper."))
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
            faults = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6]) + self.page_limit_note(name, faults)))
                continue
            args, wrong = self.resolve_ids(name, args)
            wrong = wrong or self.shop_as_brand(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Halo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            pkey = self.page_arg(args) if name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')) else None
            if pkey and args[pkey] >= 2 and self.page_sig(name, args, pkey) not in self.pages_seen:
                trimmed += f'\n[Halo] Page 1 of this list comes first ({pkey}={args[pkey]} was asked before any earlier page of it was read).'
                args = {**args, pkey: 1}
            mark = name + _dump(args)
            coded = name == self.roles.get('search') and any(isinstance(v, str) and re.search(MODEL_CODE, v) for v in args.values())
            if coded and self.code_tries[0] >= 2 and not self.code_tries[1]:
                out.append((cid, f'Not run (no step used): {self.code_tries[0]} searches by a model or part code found nothing in this store, so codes are not searchable here. Search by the kind of product and its plain values, then read the codes on the rows and pages.'))
                continue
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            repeat = self.lead_repeat(name, args) if name == self.roles.get('search') and not ending else ''
            if repeat:
                self.trace.append({'kind': 'lead_repeat', 'args': _clip(args, 80), 'twin': repeat})
                out.append((cid, f'Not run (no step used): this store reads only the first {self.lead_count} word(s) of a query, so this search would list the same rows as {repeat} (its result is above). Start with other words: another name for the product, a word of its store label, or a value fitting listings state.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Halo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.is_refuse(name, args) and not self.redone and self.end_refused_at >= 0 and self.last_note_at >= self.end_refused_at and self.left() >= 3:
                self.redone = True
                self.audited = False
                self.refuse_blocks = 0
                self.leads = []
                self.trace.append({'kind': 'refused_end_changed', 'note': (self.notes or [''])[-1][:200]})
                out.append((cid, 'Not run yet (no step used). The store refused the ending and changed the request: ' + ((self.notes or [''])[-1] or 'see its last note') + '. The requirements in force now are:\n' + self.ledger() + '\nListings ruled out before may fit now (a higher budget, a requirement given up). Search with the changed values, open what may fit, put the best one in the cart and check out; end without a purchase only if nothing meets the requirements in force now.'))
                continue
            if ending and self.is_buy(name, args) and self.lines_unread():
                changed = self.read_cart_lines()
                if changed:
                    out.append((cid, "Not run yet (no step used for the order). Halo read the cart's lines with the store's listing views before the order, and the store showed a change:\n" + changed + '\nApply it, then check out.\nCart now: ' + self.cart_text()))
                    continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    if not self.note_held and self.refused_at >= 0 and self.last_note_at >= self.refused_at and self.cart_mark() == self.cart_at_refusal and self.brief.get('pick') in ('cheapest', 'most', 'least') and self.left() > self.tail() + 3:
                        self.note_held = True
                        self.trace.append({'kind': 'note_hold', 'note': (self.notes or [''])[-1][:200]})
                        out.append((cid, 'Not run yet (no step used). The refused checkout came with a change to the request: ' + ((self.notes or [''])[-1] or 'see the last note') + ". That note is the shopper's later word and replaces the request's text where they differ; the requirements in force now are:\n" + self.ledger() + '\nThe cart is still the one the store refused. A requirement given up (or a higher budget) lets listings ruled out before fit now, and one may beat the cart on the pick rule: search the item without the dropped value, read the best rows, put the best fitting listing in the cart, then check out.\nCart now: ' + self.cart_text()))
                        continue
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
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
                    roomy = not misfit and self.left() > self.tail() + 3
                    loose = self.inside_longer() if roomy and not self.longer_held else []
                    if loose:
                        self.longer_held = True
                        self.trace.append({'kind': 'longer_value_hold', 'lines': loose[:3]})
                        out.append((cid, 'Not run yet (no step used). A cart line states a required value only inside a longer one, while another opened listing of that item states it exactly: ' + '; '.join(loose[:3]) + ". The listing stating the value exactly is the safer fit: swap to it (keeping the pick rule and the budget) unless the request's own words accept the longer value, then check out.\nCart now: " + self.cart_text()))
                        continue
                    other = self.other_value_lines() if roomy and not self.other_held else []
                    if other:
                        self.other_held = True
                        self.trace.append({'kind': 'other_value_hold', 'lines': other[:3]})
                        out.append((cid, "Not run yet (no step used). A cart line's attribute for a required property states another value: " + '; '.join(other[:3]) + '. A value in different words does not state the requirement (the reading rules): swap to a listing that states it (the best of those named on the pick rule, or one of the unread rows named once its page shows the value), unless the two values are one word spelled differently; then check out.\nCart now: ' + self.cart_text()))
                        continue
                    bare = self.no_attribute_lines() if roomy and not loose and not self.attr_held else []
                    if bare:
                        self.attr_held = True
                        self.trace.append({'kind': 'no_attr_hold', 'lines': bare[:3]})
                        out.append((cid, 'Not run yet (no step used). A cart line has no attribute for a property a requirement sets, while other opened listings of that item state every requirement in their attributes: ' + '; '.join(bare[:3]) + '. A title or description is weaker proof than an attribute: swap to such a listing (keeping the pick rule, the budget and any fit to another item), unless another attribute of the line states the value; then check out.\nCart now: ' + self.cart_text()))
                        continue
                    gauge = self.no_figure_line() if roomy and not loose and not self.attr_held else None
                    if gauge:
                        self.attr_held = True
                        line, limits, opened, rows = gauge
                        read = [i for i in rows[:2] if self.full.get(i)] if rows and self.left() > self.tail() + 6 and self.model_room() > 40 and self.fill_facts(rows[:2], self.tail() + 4) else []
                        self.trace.append({'kind': 'figure_hold', 'line': line, 'limits': limits, 'opened': opened, 'rows': rows, 'read': read})
                        text = f"Not run yet (no step used). {line} gives no figure for {'; '.join(limits)} anywhere on its record: no attribute for that property holds a number and no field states one in that unit, so nothing shows the limit is met."
                        if opened:
                            text += ' Opened listings of that item that state the figure and break no requirement their record states, cheapest first: ' + self.priced(opened) + '.'
                        if read:
                            text += ' Halo opened the cheapest unread rows of that item (one step each):\n' + '\n'.join(f'{i}: {self.attrs_first(i)}' for i in read) + '\n'
                        rest = [i for i in rows if i not in read]
                        if rest:
                            text += ' Unread rows of that item in the budget, those showing a figure in that unit first, then by price: ' + self.priced(rest[:5]) + '.'
                        text += ' Move to the listing that wins on the pick rule among those whose record gives a figure within the limit and meets every other requirement; keep the line only if there is none, then check out.'
                        out.append((cid, f'{text}\nCart now: {self.cart_text()}'))
                        continue
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A cart line does not work with the device it has to fit: ' + '; '.join(misfit[:3]) + '. Make the repair named (the cheaper one when two are named), then check out. Only when the request accepts a listing that falls short and no fitting one is seen, keep the line and check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
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
                        seen = "Halo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Halo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    basket = self.series_set() if self.is_refuse(name, args) and not self.series_offered else None
                    if basket and any(i not in self.cart_ids() for i in basket[1].values()):
                        self.series_offered = True
                        self.trace.append({'kind': 'series_set', 'basket': basket[:2]})
                        out.append((cid, 'Not run yet (no step used). Seen listings make a full set in one part series for the items that carry series marks: ' + self.series_words(basket) + ". Every line of such a set has to share one series, and the host listing may be switched to reach one. Put these lines in the cart (with the other items' lines) and check out; end without a purchase only if one of them clearly breaks a requirement."))
                        continue
                    fixed = self.fix_counts(self.count_fixes()) if self.is_refuse(name, args) and self.cart_ids() else []
                    if fixed:
                        self.trace.append({'kind': 'counts_set_before_end', 'fixed': fixed})
                        out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + ". The cart tool's count sets how many units a line holds, so a wrong count is no reason to end without a purchase. If every line now meets the requirements, check out; end without a purchase only if a line clearly breaks one.\nCart now: " + self.cart_text()))
                        continue
                    unread = self.unread_before_end() if self.is_refuse(name, args) else ''
                    if unread:
                        out.append((cid, 'Held (the ending used no step). ' + unread + '\nCart now: ' + self.cart_text()))
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
            kept = self.kept_basket() if ending and self.is_refuse(name, args) and not self.late_held and self.left() > 6 else []
            if kept:
                self.late_held = True
                missing = [i for i in kept if i not in self.cart_ids() or i in self.dropped]
                self.trace.append({'kind': 'kept_basket', 'lines': kept, 'missing': missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                out.append((cid, f"Not run yet (no step used). The basket built at step {self.basket_at} ({', '.join(kept)}) covers every requested item, nothing changed since it was built, and an independent check of its records says it meets every requirement. Ending without a purchase earns nothing while such a basket exists. " + redo + ' End without a purchase only if a record clearly breaks a requirement in force, and say which.'))
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
            if self.is_buy(name, args) and self.missing_items():
                name, args, said = self.short_order_turn(name, args)
                trimmed += said
            if self.is_refuse(name, args):
                gap = self.gap_fill()
                if gap:
                    out.append((cid, f'{gap}\nCart now: {self.cart_text()}'))
                    continue
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
                    trimmed = '\n[Halo] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Halo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
            echo = self.echo_note(name, args, observation) if name == self.roles.get('search') and not ending else ''
            if echo:
                self.paged_obs = []
                trimmed += echo
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) and not echo else ''
            if name == self.roles.get('search') and not ending and not echo:
                trimmed += self.list_stop(name, args, observation)
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                misfit = self.fit_faults() if self.cart_ids() else []
                basket = self.series_set() if self.cart_ids() and self.series_notes < 2 else None
                takes = {c for i in self.cart_ids() for c in (self.fit_view()['marks'].get(i) or {}).get('takes') or ()} if basket else set()
                if basket and takes and not takes & self.series_whole:
                    self.series_notes += 1
                    trimmed += f"\n[Halo] The cart's host listing takes series {', '.join(sorted(takes))}, and no seen set of that series has a listing for every item with series marks; a full set is seen in {self.series_words(basket)}. Switch to one series that has every part rather than searching on in this one."
                if misfit:
                    self.trace.append({'kind': 'fit_notice', 'faults': misfit[:3]})
                    trimmed += '\n[Halo] The cart as it stands fails a fit check: ' + '; '.join(misfit[:2]) + '. Fix this before checkout; a part picked for an earlier device line may not suit the current one.'
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
                    if lines != self.built:
                        self.basket_at = self.step
                        self.basket_needs = tuple(self.needs_now())
                    self.built = lines
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if coded:
                self.code_tries[1 if self.records_in(observation) else 0] += 1
            if name == self.roles.get('search') and not ending and not coded and not self.records_in(observation):
                size = max((len(v.split()) for v in args.values() if isinstance(v, str)), default=0)
                if size >= 3:
                    trimmed += f'\n[Halo] No listing matched these {size} words. This store may want every word to match: try the one or two words naming the product and its most telling value.'
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
                    self.cart_at_refusal = self.cart_mark()
            if self.is_refuse(name, args) and not reply.get('finished') and not self.finished:
                self.end_refused_at = self.step
            if name != self.roles.get('cart') and not ending:
                trimmed += self.cart_by_two_thirds()
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Halo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Halo re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Halo's own calls] " + ' | '.join(self.queued)
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
        faults = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        whole = True
        if buy and not holding and self.left() >= 2 and self.missing_items():
            self.fill_short_cart()
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
            if self.finished or self.left() <= 0:
                return ''
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or bool(self.gap_fill()) or not self.can_refuse()
            if whole and self.bound and not self.finished:
                self.fit_fix()
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            self.read_cart_lines()
            if self.finished or self.left() <= 0:
                return ''
            faults = []
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
        self.page_tool_role()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        first = self.first_searches()
        if first:
            intro += '\n\n' + first + '\n\nThose rows are in front of you already: weigh them first and open in full the ones that may fit (their attributes decide). Search further only for what they leave open: an item with no promising row, a required value no row shows, or more candidates when the pick rule compares every fitting listing.\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Halo] Halo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
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
            if not self.values_sought and self.searched and not self.ready_to_ask():
                self.tell(self.stated_values_once())
            if not self.ready_to_ask():
                self.tell(self.walk_early(), short=True)
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
                looking = self.look_turn(content)
                idle += 0 if looking else 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': ('[Halo] Search and the listing views are offered again for this one turn.\n' if looking else '') + 'Call one of the tools to continue.\n' + self.status()})
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
            self.look_now = False
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def shop_names(self) -> dict[str, bool]:
        sellers = {}
        brands = set()
        for row in self.rows.values():
            for key, value in row.items():
                if not isinstance(value, str):
                    continue
                name = value.strip().lower()
                if not 2 < len(name) <= 40:
                    continue
                if re.search('seller|shop|store|merchant', str(key), re.I):
                    sellers.setdefault(name, False)
                if re.search('brand|maker|manufactur|vendor|make', str(key), re.I):
                    brands.add(name)
        for name in sellers:
            sellers[name] = name in brands
        return sellers

    def store_labels(self) -> set[str]:
        names = {self.label_of(i) for i in self.rows}
        for tool in self.schemas:
            shelf = self.shelf_field(tool)
            spec = self.tool_props(tool).get(shelf or '') or {}
            for choice in spec.get('enum') or []:
                if isinstance(choice, str):
                    names.add(choice.strip())
        return {n for n in names if n and len(n) <= 60}

    @staticmethod
    def whole_word(word: str, text: str) -> bool:
        return re.search('(?<![a-z0-9])' + re.escape(word.lower()) + '(?![a-z0-9])', text.lower()) is not None

    def pointer_shop(self, text: str, hint: str, common: set[str]) -> str:
        known = list(self.shop_names())
        for source in (hint, text):
            places = []
            for name in known:
                found = re.search('(?<![A-Za-z0-9])' + re.escape(name) + '(?![A-Za-z0-9])', source, re.I)
                if found:
                    places.append((found.start(), found.group(0)))
            if places:
                return min(places)[1]
        for found in re.finditer("\\b[A-Z][a-z][A-Za-z'-]+", text):
            lead = text[:found.start()].rstrip()
            if not lead or lead[-1] in '.:;!?':
                continue
            word = found.group(0)
            if word.endswith("'s"):
                word = word[:-2]
            if word.lower() not in common:
                return word
        return ''

    def pointer_values(self, rest: str, shop: str) -> list[str]:
        opener = re.search('\\b(?:with|that has|that have|having|in)\\b', rest)
        if opener is None:
            return []
        shop_words = shop.lower().split()
        values = []
        for piece in re.split(',|;|\\band\\b|\\(|\\)', rest[opener.end():]):
            numbers = re.findall('(?<![\\w.])\\d+(?:\\.\\d+)?(?![\\w.])', piece)
            if numbers:
                values.append(numbers[0])
                continue
            words = [w for w in re.findall('[a-z0-9][a-z0-9-]*', piece.lower()) if w not in LIGHT_WORDS]
            if words and words[-1] not in shop_words and words[-1] not in values:
                values.append(words[-1])
        words_first = [v for v in values if not re.fullmatch('[\\d.]+', v)]
        return (words_first + [v for v in values if re.fullmatch('[\\d.]+', v)])[:3]

    def pointer_terms(self, text: str, hint: str = '') -> tuple[str, str, list[str]]:
        head, colon, body = text.partition(':')
        if colon and len(head.split()) <= 4:
            text = body.strip()
        text = re.split('\\s*\\((?:or|read|either)\\b|\\bor wherever\\b|\\bin either direction\\b', text)[0]
        labels = self.store_labels()
        common = set()
        for item in self.items:
            common.update(re.findall('[a-z0-9]+', item.name.lower()))
        for name in labels:
            common.update(re.findall('[a-z0-9]+', name.lower()))
        shop = self.pointer_shop(text, hint, common)
        kind = ''
        for name in sorted(labels, key=len, reverse=True):
            if self.whole_word(name, text):
                kind = name
                break
        start = text.lower().find(shop.lower()) if shop else -1
        rest = text[start + len(shop):] if start >= 0 else text
        rest = re.sub('(?<=\\d),(?=\\d{3}\\b)', '', rest)
        return (shop, kind, self.pointer_values(rest, shop))

    def hop_words(self, text: str, hint: str = '') -> tuple[str, list[str]]:
        shop, kind, values = self.pointer_terms(text, hint)
        if not shop:
            return ('', [])
        drafts = []
        if values:
            drafts.append(' '.join([shop] + values))
        if kind:
            drafts.append(' '.join([shop] + re.findall('[A-Za-z0-9][A-Za-z0-9-]*', kind)))
        drafts.append(shop)
        queries = []
        for draft in drafts:
            if draft.lower() not in {q.lower() for q in queries}:
                queries.append(draft)
        return (shop, queries)

    def link_room(self) -> int:
        return max(self.max_steps // 8, 5)

    def chain_reserve(self) -> int:
        return max(6, min(self.max_steps // 6, 14))

    def hops_due(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        due = []
        for _, need in self.refs:
            if need in live and need not in self.ref_values:
                due.append(need)
        return due

    def hop_openers(self, plan: tuple[str, dict[str, Any], str, str | None], width: int, hops: list[str], spare: Any) -> list[str]:
        tool, fixed, field, _ = plan
        notes = []
        for need in hops:
            for query in self.hop_words(need)[1][:2]:
                if self.link_searches >= self.link_room() or not spare():
                    break
                words = ' '.join(self.trim_words(query.split(), width, 0))
                got = self.probe(tool, {**fixed, field: words})
                if got is None:
                    break
                self.link_searches += 1
                self.link_seen[words.lower()] = got[2]
                shop_hits = self.shop_rows(need, got[2])
                notes.append(f'the listing that "{need}" points at: {got[3]}')
                self.trace.append({'kind': 'hop_search', 'words': words, 'rows': shop_hits[:4]})
                if shop_hits:
                    break
        return notes

    def record_text(self, ident: str) -> str:
        pending = [self.full.get(ident) or self.rows.get(ident) or {}]
        pieces = []
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                pieces.extend(_words(k) for k in node)
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
            else:
                pieces.append(str(node))
        return re.sub('(?<=\\d),(?=\\d{3}\\b)', '', ' '.join(pieces).lower())

    def page_lacks(self, ident: str, values: list[str]) -> list[str]:
        text = self.record_text(ident)
        numbers = {float(n) for n in re.findall('\\d+(?:\\.\\d+)?', text)}
        vocab = set(re.findall('[a-z0-9][a-z0-9-]*', text)) | _roots(text)
        missing = []
        for value in values:
            if re.fullmatch('\\d+(?:\\.\\d+)?', value):
                present = float(value) in numbers
            else:
                present = value in vocab or bool(_roots(value) & vocab)
            if not present:
                missing.append(value)
        return missing

    def better_starts(self, need: str, first: str) -> list[str]:
        _, kind, values = self.pointer_terms(need)
        if not values:
            return []
        short = len(self.page_lacks(first, values))
        if short == 0:
            return []
        candidates = []
        for place, ident in enumerate(self.shop_rows(need.split(':', 1)[-1], list(reversed(self.seen_order)))):
            if ident == first:
                continue
            if kind and not _roots(kind) & _roots(_dump(self.rows.get(ident) or {})):
                continue
            misses = len(self.page_lacks(ident, values))
            if misses < short:
                candidates.append((misses, place, ident))
        return [c[2] for c in sorted(candidates)]

    def request_words_for(self, need: str) -> str:
        tail = need.split(':', 1)[-1]
        wanted = set(re.findall('[a-z0-9]{3,}', tail.lower()))
        proper = {w.lower() for w in re.findall('\\b[A-Z][a-z]{3,}', tail)}
        chosen, most = '', 2
        for sentence in re.split('\\n+|(?<=[.!?])\\s+', self.request):
            hits = {w for w in wanted if self.whole_word(w, sentence)}
            if proper and not proper & hits:
                continue
            if len(hits) > most:
                chosen, most = sentence.strip(), len(hits)
        return chosen[:400]

    def walk_early(self) -> str:
        if self.finished or self.early_walks >= max(2, self.max_steps // 30) or self.model_room() < 90:
            return ''
        gap = min(8, max(4, self.max_steps // 8))
        if self.step - self.early_at < gap or self.left() <= self.chain_reserve() + 2:
            return ''
        if not self.searched or not self.hops_due():
            return ''
        self.early_walks += 1
        self.early_at = self.step
        before = set(self.ref_values)
        walked, _ = self.trail(early=True)
        fresh = [need for need in self.ref_values if need not in before]
        searched = self.search_resolved(fresh)
        self.trace.append({'kind': 'walk_early', 'step': self.step, 'resolved': sorted(self.ref_values)[:3], 'text': walked[:160]})
        return '\n'.join(x for x in (walked.strip(), searched) if x)

    def search_resolved(self, needs: list[str]) -> str:
        if self.brief.get('pick') != 'cheapest' or self.finished or self.left() < self.chain_reserve() + 6:
            return ''
        plan = self.search_template()
        if not plan:
            opener = self.word_tool()
            plan = opener[:3] if opener else None
        if not plan:
            return ''
        tool, fixed, field = plan
        width = self.query_width(tool, field)
        shelf = self.shelf_field(tool)
        lines = []
        for need in needs:
            value = self.ref_values.get(need, '')
            owner = 0
            for n, ref in self.refs:
                if ref == need:
                    owner = n
                    break
            how = need.split(':', 1)[-1].lower()
            if not owner or not value or need in self.resolved_done or value.startswith('as stated on'):
                continue
            if re.search('\\b(?:unlike|differ\\w*|other than|anything but|not|within|inside|at most|at least|above|below)\\b', how):
                continue
            self.resolved_done.add(need)
            item = self.items[owner - 1]
            value_words = re.findall('[A-Za-z0-9][A-Za-z0-9.-]*', value)[:3]
            taken = {w.lower() for w in value_words}
            noun = [w for w in re.findall('[A-Za-z0-9][A-Za-z0-9-]*', item.name) if w.lower() not in taken]
            args = {k: v for k, v in fixed.items() if k != shelf}
            shelf_name = self.shelf_value(tool, owner) if shelf else None
            if shelf_name:
                args[shelf] = shelf_name
            args[field] = ' '.join(self.trim_words(value_words + noun, width, len(noun)))
            got = self.probe(tool, args, 3000)
            if got is None:
                break
            fits = [i for i in got[2] if self.price_of(i) is not None and self.buyable(i) and (not self.category_key() or self.owns(item, self.label_of(i)))]
            fits.sort(key=lambda i: self.price_of(i) or 0.0)
            self.trace.append({'kind': 'search_resolved', 'need': need[:120], 'value': value[:60], 'rows': fits[:6]})
            listed = f'{self.priced(fits[:6])} (cheapest first; open them to check the value and the other requirements)' if fits else 'no row of that kind'
            lines.append(f"[The pointed pages gave {value} for item {owner}, and the pick is the cheapest listing that has it, so Halo searched {got[0].get(field)!r}] {listed}")
        return '\n'.join(lines)

    def shop_as_brand(self, name: str, args: dict[str, Any]) -> str:
        if any(name == self.roles.get(job) for job in ('cart', 'buy', 'refuse', 'ask', 'open', 'offer', 'compare')):
            return ''
        sellers = self.shop_names()
        for key, value in args.items():
            if not isinstance(value, str) or not re.search('brand|maker|manufactur|vendor|make', _snake(key), re.I):
                continue
            if sellers.get(value.strip().lower()) is False:
                field = self.word_field(name) or self.word_field(self.roles.get('search'))
                where = f' in {field}' if field else ''
                return f"{value} is a shop that sells here (the rows' shop field), not a brand or maker, so filtering {key} by it would miss its listings; put its name in the search words{where}, followed by two or three values its description gives"
        return ''

    def trail_records(self) -> list[str]:
        out = []
        for ident in reversed(self.seen_order):
            if len(out) >= 45:
                break
            if ident not in self.opened and ident not in self.full:
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            if record:
                tag = 'OPENED ' if ident in self.opened else 'ATTRIBUTES READ, PAGE NOT OPENED '
                out.append(tag + _dump(_clip(record, 300))[:1400])
        for ident in reversed(self.seen_order):
            if len(out) >= 60:
                break
            if ident in self.full or (ident in self.opened and self.rows.get(ident)):
                continue
            out.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
        return out

    def request_quotes(self, pending: list[str]) -> str:
        lines = []
        for need in pending:
            said = self.request_words_for(need)
            if said:
                lines.append(f'- {need}: "{said}"')
        if not lines:
            return ''
        return "\n\nTHE REQUEST'S OWN SENTENCE FOR EACH (the relation it states, such as the same value, a different one or its partner, settles the value):\n" + '\n'.join(lines)

    def link_lookup(self, about: str, words: str) -> tuple[list[str], list[str]]:
        wanted = self.pointer_terms(about, words)[2]
        queries = self.hop_words(about, words)[1] or ([words.strip()] if words.strip() else [])
        plan = self.search_template()
        if not plan:
            opener = self.word_tool()
            plan = opener[:3] if opener else None
        found = []
        if not plan:
            return (found, wanted)
        tool, fixed, field = plan
        width = self.query_width(tool, field)
        described = f'{about} {words}'
        for query in queries:
            if found or self.finished or self.left() <= 12 or self.link_searches >= self.link_room():
                break
            text = ' '.join(self.trim_words(query.split(), width, 0))
            key = text.lower()
            self.link_searches += 1
            if key not in self.link_seen:
                got = self.probe(tool, {**fixed, field: text}, 0)
                if got is None:
                    break
                self.link_seen[key] = got[2]
                found = self.shop_rows(described, got[2])
                self.trace.append({'kind': 'hop_lookup', 'words': text, 'rows': found[:4]})
            else:
                found = self.shop_rows(described, self.link_seen[key])
        return (found, wanted)

    def need_state(self, ident: str, number: int, need: str, facts: dict[str, Any], record: dict[str, Any]) -> str:
        if self.fit_need(number, need):
            return 'pending'
        if need in [r for _, r in self.refs]:
            value = self.ref_values.get(need, '')
            if not value or value.startswith('as stated on') or re.search('\\b(?:unlike|differ\\w*|other than|not|partner|goes with|mates?|pairs?)\\b', need.partition(':')[2].lower()):
                return 'pending'
            need = need.partition(':')[0] + ': ' + value
        prop, colon, value = need.partition(':')
        low = value.strip().lower()
        if not colon or not low:
            return 'unsettled'
        keys = [k for k in facts if not _is_offer(k) and _names(_spelled(prop), _spelled(k))]
        held = [_chosen_other(facts, facts[k], low) or str(facts[k]) for k in keys]
        rest = _word_bag(' '.join(str(v) for k, v in record.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool) and not _is_offer(k)) + ' ' + ' '.join(str(v) for k, v in facts.items() if k not in keys and not _is_offer(k)))
        limit = _bound(low)
        if limit:
            upper, strict, figure, unit = limit
            for text in held:
                got = re.search(FIGURE, text.lower())
                if not got or not _unit_like(got.group(2), unit):
                    continue
                try:
                    amount = float(got.group(1).replace(',', ''))
                except ValueError:
                    continue
                if upper:
                    met = amount < figure if strict else amount <= figure
                else:
                    met = amount > figure if strict else amount >= figure
                return 'ok' if met else 'conflict'
            return 'unsettled'
        negative = re.match('\\s*(?:not|no|anything but|other than|except|without)\\b\\s*(.*)$', low)
        if negative:
            barred = _word_bag(negative.group(1))
            if not barred:
                return 'unsettled'
            if held:
                return 'conflict' if any(barred <= _word_bag(t) for t in held) else 'ok'
            return 'unsettled'
        options = [_word_bag(o) for o in re.split('\\s+or\\s+', low) if _word_bag(o)] or [_word_bag(low)]
        if not held:
            return 'ok' if any(o <= rest for o in options) else 'unsettled'
        if any(o <= _word_bag(t) for o in options for t in held):
            return 'ok'
        wanted = set(re.findall('\\d+(?:\\.\\d+)?', low))
        shown = {n for t in held for n in re.findall('\\d+(?:\\.\\d+)?', t)}
        if wanted and shown:
            return 'unsettled' if wanted & shown else 'conflict'
        if any(o & _word_bag(t) for o in options for t in held):
            return 'unsettled'
        return 'conflict'

    def judge_record(self, ident: str, number: int) -> tuple[list[str], list[str], list[str]]:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        facts = _facts_any(self.full.get(ident) or {}) or {}
        sorted_out = {'conflict': [], 'unsettled': [], 'pending': []}
        for owner, need in self.needs_now():
            if owner != number:
                continue
            verdict = self.need_state(ident, number, need, facts, record)
            if verdict in sorted_out:
                sorted_out[verdict].append(need.partition(':')[0].strip()[:40] or need[:40])
        if self.price_of(ident) is None or not self.buyable(ident):
            sorted_out['conflict'].append('stock')
        return (sorted_out['conflict'], sorted_out['unsettled'], sorted_out['pending'])

    def money_room(self, number: int) -> tuple[float | None, int]:
        limit = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        count = self.items[number - 1].qty if self.brief.get('budget_scope') != 'each' else 1
        return (limit, count)

    def ready_rows(self, number: int) -> list[str]:
        limit, count = self.money_room(number)
        ranked = []
        for ident in self.item_ids(number):
            if ident not in self.full or self.sold_out(ident) or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if price is None or (limit is not None and price * count > limit + 1e-09):
                continue
            clash, open_needs, _ = self.judge_record(ident, number)
            if not clash and not open_needs:
                ranked.append((price, ident))
        ranked.sort()
        return [ident for _, ident in ranked]

    def fill_choice(self, number: int, spent: float, taken: list[str]) -> str | None:
        limit, count = self.money_room(number)
        per_item = self.brief.get('budget_scope') == 'each'
        in_cart = set(self.cart_ids())
        cat_key = self.category_key()
        best = None
        for ident in self.ready_rows(number):
            if ident in in_cart or ident in taken:
                continue
            price = self.price_of(ident) or 0.0
            if limit is not None and price * count + (0.0 if per_item else spent) > limit + 1e-09:
                continue
            if cat_key and self.fuller_label(number, self.label_of(ident)):
                continue
            key = (ident in self.dropped, len(self.judge_record(ident, number)[2]), price, ident)
            if best is None or key < best:
                best = key
        return best[3] if best else None

    def lines_short(self) -> int:
        live = sum(1 for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0)
        return max(0, len(self.items) - live)

    def row_score(self, ident: str, number: int) -> tuple[int, int]:
        shown = 0
        for words in self.item_values(number):
            if self.row_has(ident, words):
                shown += 1
        return (shown, 2 if self.kind_match(ident, number) else 0)

    def unread_order(self, number: int) -> list[str]:
        limit, count = self.money_room(number)
        by_price = self.brief.get('pick') == 'cheapest'
        in_cart = set(self.cart_ids())
        ranked = []
        mine = set(self.item_ids(number))
        for place, ident in enumerate(self.seen_order):
            if ident not in mine:
                continue
            if ident in self.full or ident in self.opened or ident in in_cart or self.sold_out(ident) or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if limit is not None and price is not None and price * count > limit + 1e-09:
                continue
            shown, kind = self.row_score(ident, number)
            if not shown and kind < 2:
                continue
            second = place
            if by_price:
                second = price if price is not None else float('inf')
            ranked.append((-(shown + kind), second, ident))
        ranked.sort()
        return [r[2] for r in ranked]

    def open_best_unread(self, numbers: list[int], count: int, reserve: int) -> tuple[list[str], list[str]]:
        allowed = max(2, self.max_steps // 6) - self.unread_spent
        room = min(allowed, self.left() - self.tail() - reserve)
        if not numbers or count <= 0 or room <= 0 or self.finished or self.model_room() < 15:
            return ([], [])
        queues = [self.unread_order(n) for n in numbers]
        chosen = []
        depth = 0
        while any(depth < len(q) for q in queues):
            for queue in queues:
                if depth < len(queue) and queue[depth] not in chosen:
                    chosen.append(queue[depth])
            depth += 1
        chosen = chosen[:min(count, room)]
        if not chosen:
            return ([], [])
        before = set(self.full)
        self.fill_facts(chosen, self.tail() + reserve)
        fresh = [i for i in chosen if i in self.full and i not in before]
        self.unread_spent += len(fresh)
        fitting = []
        for number in numbers:
            fitting.extend(i for i in self.ready_rows(number) if i in fresh)
        self.trace.append({'kind': 'unread_opened', 'items': numbers, 'chosen': chosen, 'fresh': fresh, 'fitting': fitting})
        return (fresh, fitting)

    def attrs_first(self, ident: str, limit: int = 1200) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        facts = _facts_any(record) or {}
        rest = {k: v for k, v in record.items() if not isinstance(v, dict) and v not in facts.values()}
        return (_dump(facts) + ' | ' + _dump(_clip(rest, 160)))[:limit]

    def unread_before_end(self) -> str:
        if self.unread_ended or self.model_room() < 45 or self.left() - self.tail() <= 5:
            return ''
        self.unread_ended = True
        open_items = [n for n in range(1, len(self.items) + 1) if not self.ready_rows(n)]
        fresh, fitting = self.open_best_unread(open_items, min(8, 1 + 3 * len(open_items)), 4)
        if not fresh or not all(self.ready_rows(n) for n in open_items):
            return ''
        shown = '\n'.join(f'{i}: {self.attrs_first(i, 900)}' for i in fresh)
        return f"Before ending, Halo opened {len(fresh)} listing(s) that searches had listed but no one had opened, picking the rows that show the most requested values:\n{shown}\nBy the code's reading, {self.priced(fitting)} settle every value the request states. A listing like that means buy, not end: cart the one that wins on the pick rule (with the other items' lines) and check out. End without a purchase only if one of them plainly breaks a requirement in force, and name it."

    def gap_fill(self) -> str:
        if self.gap_done or len(self.items) < 2 or not self.roles.get('cart') or self.model_room() < 25 or not self.judges(SPEND_HARD):
            return ''
        if any(self.sold_out(i) for i in self.cart_ids()):
            return ''
        kept = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        gaps = self.missing_items()
        if not kept or not gaps or len(gaps) >= len(self.items) or self.left() - self.tail() < len(gaps) + 6:
            return ''
        self.gap_done = True
        per_item = self.brief.get('budget_scope') == 'each'
        spent = 0.0
        if not per_item:
            spent = sum((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in kept)
        chosen = {}
        for number in gaps:
            pick = self.fill_choice(number, spent, list(chosen.values()))
            if pick is None:
                unopened = [i for i in self.item_ids(number) if i not in self.full and self.buyable(i)]
                unopened = sorted(unopened, key=lambda i: self.price_of(i) or 0.0)[:3]
                if unopened and self.left() - self.tail() > len(gaps) + len(unopened) + 3:
                    self.fill_facts(unopened, self.tail() + len(gaps) + 3)
                    pick = self.fill_choice(number, spent, list(chosen.values()))
            if pick is None:
                self.trace.append({'kind': 'gap_left', 'item': number})
                return ''
            chosen[number] = pick
            if not per_item:
                spent += (self.price_of(pick) or 0.0) * self.items[number - 1].qty
        basket = kept + [chosen[n] for n in chosen]
        self.check_cart(basket, SPEND_HARD)
        if not self.verdicts.get(self.check_key(basket)):
            self.trace.append({'kind': 'gap_not_confirmed', 'chosen': chosen})
            return ''
        lines = {i: self.held_units(i) or self.cart_lines.get(i, 1) for i in kept}
        for number, pick in chosen.items():
            lines[pick] = self.items[number - 1].qty
        if not self.cart_put(lines):
            return ''
        self.trace.append({'kind': 'gap_closed', 'chosen': chosen})
        added = '; '.join(f'{self.items[n - 1].name} -> {p}' for n, p in chosen.items())
        return f'Held (the ending used no step). The cart covered {len(kept)} of the {len(self.items)} requested items, so Halo put in, for each missing one, the opened listing that fits it ({added}); an independent check of the full basket finds it meets every requirement. Ordering this cart can be right, ending without one cannot: check out now, unless a line plainly breaks a requirement in force (then fix that line first).'

    def cart_put(self, lines: dict[str, int]) -> bool:
        tool = self.roles.get('cart')
        steps = self.cart_calls(lines, []) if tool else None
        if not steps or self.left() < len(steps) + 1:
            return False
        for call in steps:
            problems = []
            call = _shape(call, self.schemas.get(tool), 'args', problems)
            if problems:
                return False
            reply = self.act(tool, call)
            if reply is None or self.finished:
                return False
            self.note_cart(tool, call)
            self.observe(tool, reply.get('observation'))
        return True

    def kept_basket(self) -> list[str]:
        basket = [i for i in self.built if not self.sold_out(i)]
        if not self.category_key() or len(basket) < len(self.items):
            return []
        if self.basket_at <= max(self.last_note_at, self.refused_at) or self.basket_needs != tuple(self.needs_now()):
            return []
        for item in self.items:
            if not any(self.owns(item, self.label_of(i)) for i in basket):
                return []
        to_restore = sum(1 for i in basket if i in self.dropped or i not in self.cart_ids())
        if self.left() < to_restore + 2:
            return []
        self.check_cart(basket, SPEND_HARD)
        return basket if self.verdicts.get(self.check_key(basket)) else []

    def cart_by_two_thirds(self) -> str:
        if self.secured or self.buys or self.cart_touched or self.cart_ids() or not self.roles.get('cart'):
            return ''
        if self.step < self.max_steps * 2 // 3 or self.model_room() < 40 or self.left() - self.tail() < len(self.items) + 5:
            return ''
        if self.brief.get('pick') not in (None, '', 'any', 'cheapest'):
            return ''
        if any(p.get('when') not in ('order', 'end') for p in self.process):
            return ''
        self.secured = True
        per_item = self.brief.get('budget_scope') == 'each'
        chosen = {}
        spent = 0.0
        for number in range(1, len(self.items) + 1):
            pick = self.fill_choice(number, spent, list(chosen.values()))
            if pick is None:
                return ''
            chosen[number] = pick
            if not per_item:
                spent += (self.price_of(pick) or 0.0) * self.items[number - 1].qty
        lines = {p: self.items[n - 1].qty for n, p in chosen.items()}
        if len(self.cart_calls(lines, []) or []) + self.tail() + 3 > self.left() or not self.cart_put(lines):
            return ''
        self.trace.append({'kind': 'cart_by_two_thirds', 'chosen': chosen})
        named = '; '.join(f'{self.items[n - 1].name} -> {p}' for n, p in chosen.items())
        return f"\n[Halo] Two thirds of the steps are used and the cart was still empty, so Halo carted, per item, the cheapest opened listing whose record meets every requirement ({named}; one step each). Keep looking for better listings if the pick rule calls for it, and only swap a line for a listing that also meets everything."

    def look_turn(self, text: str) -> bool:
        self.look_now = False
        search = self.roles.get('search')
        if not search or self.look_turns >= 2:
            return False
        if self.left() < self.tail() + 2 + self.lines_short() or not re.search(LOOK_TALK, text or '', re.I):
            return False
        if any(t['function']['name'] == search for t in self.offered()):
            return False
        self.look_turns += 1
        self.look_now = True
        self.trace.append({'kind': 'look_turn', 'step': self.step, 'text': (text or '')[:160]})
        return True

    def fill_missing(self) -> bool:
        gaps = self.missing_items()
        if not gaps:
            return True
        if not self.roles.get('cart'):
            return False
        hosts = {main for _, main in self.bound}
        gaps = sorted(gaps, key=lambda n: (n not in hosts, gaps.index(n)))
        per_item = self.brief.get('budget_scope') == 'each'
        kept = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        spent = 0.0
        if not per_item:
            spent = sum((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in kept)
        unproven = [n for n in gaps if not self.ready_rows(n)]
        if unproven and not self.fill_opened and self.model_room() > 35:
            self.fill_opened = True
            self.open_best_unread(unproven, min(6, 1 + 2 * len(unproven)), 1 + len(gaps))
        chosen = {}
        for number in gaps:
            pick = self.fill_choice(number, spent, list(chosen.values()))
            if pick is None:
                self.trace.append({'kind': 'fill_missing', 'item': number, 'chosen': chosen})
                return False
            chosen[number] = pick
            if not per_item:
                spent += (self.price_of(pick) or 0.0) * self.items[number - 1].qty
        lines = {i: self.cart_lines[i] for i in kept if i in self.cart_lines}
        for number, pick in chosen.items():
            lines[pick] = self.items[number - 1].qty
        if not self.cart_put(lines):
            self.trace.append({'kind': 'fill_short_of_steps', 'chosen': chosen})
            return False
        self.trace.append({'kind': 'missing_filled', 'chosen': chosen})
        return not self.missing_items()

    def short_order_turn(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        gaps = [self.items[n - 1].name for n in self.missing_items()]
        if self.fill_short_cart():
            return (name, args, f"\n[Halo] No line covered {', '.join(gaps)}, so before the order Halo carted the opened listing that fits it best.")
        refuse = self.roles.get('refuse')
        if not refuse or not self.can_refuse() or not self.missing_items():
            return (name, args, '')
        problems = []
        ending = _shape(dict(self.brief.get('refuse_args') or {}), self.schemas.get(refuse), 'args', problems)
        if problems:
            return (name, args, '')
        self.trace.append({'kind': 'short_order_to_end', 'missing': self.missing_items(), 'cart': self.cart_ids()})
        return (refuse, ending, '\n[Halo] An order that leaves out a requested item cannot pass, and no opened listing of that item settles every stated value, so Halo ended without a purchase instead.')

    def fit_need(self, number: int, need: str) -> bool:
        hosts = [main for part, main in self.bound if part == number]
        if not hosts or any(need == r for _, r in self.refs):
            return False
        text = need.partition(':')[2] if ':' in need else need
        text = text.lower()
        if re.search('\\b(?:if|unless|when|whether|than)\\b', text):
            return False
        if re.search('\\d', re.sub('\\bitem \\d+\\b', '', text)):
            return False
        verb = re.search('\\b(?:fits?|fitting|bought|buying|end up|works? with|goes with)\\b', need.lower()) or re.search('\\b(?:compatible|suits?)\\b', text)
        if not verb:
            return False
        squashed = re.sub('[^a-z0-9]', '', text)
        for host in hosts:
            host_name = self.items[host - 1].name
            if re.search(f'\\bitem {host}\\b', text):
                return True
            names = set(_roots(host_name))
            names.add(re.sub('[^a-z0-9]', '', host_name.lower()))
            if any(len(n) > 3 and n in squashed for n in names):
                return True
        return False

    def fit_view(self) -> dict[str, Any]:
        stamp = (len(self.seen_order), len(self.full))
        if stamp != self.fit_cache[0]:
            self.fit_cache = (stamp, self.fit_index())
        return self.fit_cache[1]

    def mark_of(self, index: dict[str, Any], ident: str) -> dict[str, Any]:
        return index['marks'].get(ident) or {}

    def fits_codes(self, ident: str, wanted: set[str], index: dict[str, Any]) -> bool | None:
        works = self.fit_set(ident, index)
        if not wanted or not works:
            return None
        hit = any(_code_hit(w, c) for w in works for c in wanted)
        barred = any(_code_hit(n, c) for n in self.mark_of(index, ident).get('nots') or [] for c in wanted)
        return hit and not barred

    def device_codes(self, number: int, index: dict[str, Any]) -> set[str]:
        for part, main in self.bound:
            if part != number:
                continue
            hosts = [i for i in self.cart_ids() if not self.sold_out(i) and i in self.item_ids(main)]
            if len(hosts) == 1:
                return set(self.mark_of(index, hosts[0]).get('codes') or [])
        return set()

    def told_codes(self, item_no: int, index: dict[str, Any]) -> set[str]:
        said = set()
        for owner, need in self.needs_now():
            if owner == item_no:
                said |= {c for c in _codes((need.partition(':')[2] or need).upper()) if not c.endswith('x')}
        seen = set()
        for mark in index['marks'].values():
            for field in ('codes', 'works', 'nots'):
                seen.update(mark.get(field) or [])
        return {c for c in said if any(_code_hit(s, c) for s in seen)}

    def meets_own(self, ident: str, number: int) -> bool:
        clash, open_needs, _ = self.judge_record(ident, number)
        return not (clash or open_needs)

    def fit_candidates(self, part: int, codes: set[str], index: dict[str, Any], cart: list[str]) -> tuple[list[str], list[str]]:
        proven, unread = [], []
        for ident in self.item_ids(part):
            if ident in cart or not self.buyable(ident) or self.price_of(ident) is None:
                continue
            if not self.fits_codes(ident, codes, index):
                continue
            if ident not in self.full:
                unread.append(ident)
            elif self.meets_own(ident, part):
                proven.append(ident)
        proven.sort(key=lambda i: self.price_of(i) or 0.0)
        unread.sort(key=lambda i: (-sum(self.row_score(i, part)), self.price_of(i) or 0.0))
        return (proven, unread)

    def owner_map(self) -> dict[str, int]:
        owners = {}
        for number in range(1, len(self.items) + 1):
            for ident in self.item_ids(number):
                owners[ident] = number
        return owners

    def series_faults(self, index: dict[str, Any], lines: list[str]) -> list[str]:
        owners = self.owner_map()

        def by_price(ids: list[str]) -> list[str]:
            return sorted(ids, key=lambda i: self.price_of(i) or 0.0)

        def spare_rows(number: int) -> list[str]:
            return [i for i in self.item_ids(number) if i not in lines and self.price_of(i) is not None and not self.sold_out(i)]
        found = []
        for host in lines:
            accepts = set(self.mark_of(index, host).get('takes') or [])
            if host not in owners or not accepts:
                continue
            host_item = owners[host]
            for line in lines:
                series = self.mark_of(index, line).get('family') or ''
                if line == host or not series or series in accepts:
                    continue
                line_item = owners.get(line, host_item)
                if line_item == host_item:
                    continue
                swaps = by_price([i for i in spare_rows(line_item) if self.mark_of(index, i).get('family') in accepts])[:4]
                rehost = by_price([i for i in spare_rows(host_item) if series in (self.mark_of(index, i).get('takes') or ())])[:3]
                accepted = ', '.join(sorted(accepts))
                text = f'{line} ({self.items[line_item - 1].name}) is from series {series}, but {host} ({self.items[host_item - 1].name}) accepts series {accepted}'
                if swaps:
                    text += f"; {accepted} listings seen for {line}'s item, cheapest first: {self.priced(swaps)}"
                if rehost:
                    text += f"; or {host}'s item in a version that accepts {series}, cheapest first: {self.priced(rehost)}"
                if not swaps and not rehost:
                    whole = self.series_set()
                    if whole:
                        text += f'; a complete set in one series is seen: {self.series_words(whole)}'
                    else:
                        text += f'; search that item with the series code ({sorted(accepts)[0]}) and keep every line in a single series'
                found.append(text)
        return found

    def series_set(self) -> tuple[str, dict[int, str], float] | None:
        index = self.fit_view()
        owners = self.owner_map()
        takers = {}
        members = {}
        for ident, mark in index['marks'].items():
            number = owners.get(ident)
            if number is None or self.price_of(ident) is None or self.sold_out(ident):
                continue
            if ident in self.full and not self.meets_own(ident, number):
                continue
            for series in mark.get('takes') or ():
                takers.setdefault(series, {}).setdefault(number, []).append(ident)
            if mark.get('family'):
                members.setdefault(mark['family'], {}).setdefault(number, []).append(ident)
        host_slots = set()
        for by_item in takers.values():
            host_slots |= set(by_item)
        part_slots = set()
        for by_item in members.values():
            part_slots |= set(by_item)
        part_slots -= host_slots
        if not host_slots or not part_slots:
            return None
        limit = None
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each':
            limit = self.budget
        best = None
        self.series_whole = set()
        for series in sorted(set(takers) & set(members)):
            slots = {}
            for number in host_slots:
                slots[number] = takers[series].get(number)
            for number in part_slots:
                slots[number] = members[series].get(number)
            if any(not pool for pool in slots.values()):
                continue
            self.series_whole.add(series)
            choice = {}
            total = 0.0
            for number, pool in slots.items():
                choice[number] = min(pool, key=lambda i: self.price_of(i) or 0.0)
                total += (self.price_of(choice[number]) or 0.0) * self.items[number - 1].qty
            if limit is not None and total > limit + 1e-09:
                continue
            if best is None or total < best[2]:
                best = (series, choice, total)
        return best

    def series_words(self, basket: tuple[str, dict[int, str], float] | None) -> str:
        if not basket:
            return ''
        series, choice, total = basket
        parts = [f'{self.items[n - 1].name} {choice[n]} ({self.price_of(choice[n]):g})' for n in sorted(choice)]
        return f"series {series}: {', '.join(parts)} (total {total:g})"

    def fit_fix(self) -> None:
        if self.fit_fixed or not self.bound or not self.roles.get('cart'):
            return
        self.fit_fixed = True
        cart_tool = self.roles['cart']
        for part, main in self.bound:
            index = self.fit_index()
            current = self.cart_ids()
            hosts = [i for i in current if i in self.item_ids(main)]
            mine = [i for i in current if i in self.item_ids(part)]
            if len(mine) != 1 or len(hosts) != 1:
                continue
            line = mine[0]
            host_codes = set(self.mark_of(index, hosts[0]).get('codes') or [])
            if not host_codes or self.fits_codes(line, host_codes, index) is not False:
                continue
            proven, unread = self.fit_candidates(part, host_codes, index, current)
            if unread and not proven and self.model_room() > 6 and self.left() - self.tail() >= 5:
                self.fill_facts(unread[:2], self.tail() + 3)
                proven = self.fit_candidates(part, host_codes, self.fit_index(), current)[0]
            wanted = self.items[part - 1].qty
            if isinstance(self.budget, (int, float)):
                each = self.brief.get('budget_scope') == 'each'
                others = 0.0
                if not each:
                    others = sum((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in current if i != line)
                units = 1 if each else wanted
                proven = [i for i in proven if others + (self.price_of(i) or 0.0) * units <= self.budget + 1e-09]
            if not proven:
                continue
            stay = {}
            for ident in current:
                if ident != line and ident in self.cart_lines:
                    stay[ident] = self.cart_lines[ident]
            stay[proven[0]] = wanted
            steps = self.cart_calls(stay, [line])
            if not steps or self.left() < len(steps) + self.tail():
                continue
            for call in steps:
                problems = []
                call = _shape(call, self.schemas.get(cart_tool), 'args', problems)
                if problems or self.act(cart_tool, call) is None:
                    return
                self.note_cart(cart_tool, call)
            self.trace.append({'kind': 'fit_fix', 'line': line, 'pick': proven[0]})

    def repair_text(self, line: str, part: int, main: int, host: str, codes: set[str], index: dict[str, Any], proven: list[str], unread: list[str]) -> str:
        held = lambda ident: max(1, self.held_units(ident) or 1)
        cost = lambda ident: self.price_of(ident) or 0.0
        alone = not any(m == main and p != part for p, m in self.bound)
        rehost = []
        for ident in self.item_ids(main):
            if ident == host or self.sold_out(ident) or self.price_of(ident) is None:
                continue
            if self.fits_codes(line, set(self.mark_of(index, ident).get('codes') or []), index):
                rehost.append(ident)
        rehost = sorted(rehost, key=cost)[:2]
        part_way = cost(host) * held(host) + cost(proven[0]) * self.items[part - 1].qty if proven else None
        host_way = cost(rehost[0]) * held(host) + cost(line) * held(line) if rehost and alone else None
        text = f"{line} ({self.items[part - 1].name}) suits {', '.join(sorted(self.fit_set(line, index))[:4])}, not {host} ({', '.join(sorted(codes))})"
        if proven:
            text += f'; opened parts of that kind that meet their own requirements and suit it, cheapest first: {self.priced(proven[:4])}'
        if part_way is not None and (host_way is None or part_way <= host_way):
            text += f'; the cheaper fix is the part: {proven[0]} (pair total {part_way:g})'
            if host_way is not None:
                text += f', against changing the device ({rehost[0]} with {line}: {host_way:g})'
        elif host_way is not None:
            text += f'; the cheaper fix is the device: {rehost[0]}, which {line} suits (pair total {host_way:g}'
            text += f'; the part {proven[0]} instead: {part_way:g})' if part_way is not None else ')'
        elif not unread:
            text += '; no opened part of that kind meets its requirements and suits it, so search that kind of part, or move the device and its parts together to a model code whose parts are seen: ' + self.fit_plan(short=True)
        return text

    def named_code_faults(self, index: dict[str, Any], cart: list[str]) -> list[str]:
        found = []
        parts_bound = {p for p, _ in self.bound}
        for number, item in enumerate(self.items, 1):
            if number in parts_bound:
                continue
            named = self.told_codes(number, index)
            if not named:
                continue
            mine = set(self.item_ids(number))
            for line in cart:
                if line not in mine:
                    continue
                mark = self.mark_of(index, line)
                if named & set(mark.get('codes') or []) or (mark.get('own') or '').lower() in named:
                    continue
                if self.fits_codes(line, named, index) is not False:
                    continue
                alternatives = [i for i in mine if i not in cart and not self.sold_out(i) and self.price_of(i) is not None and self.fits_codes(i, named, index)]
                alternatives = sorted(alternatives, key=lambda i: self.price_of(i) or 0.0)[:4]
                text = f"{line} ({item.name}) suits {', '.join(sorted(self.fit_set(line, index))[:4])}"
                if mark.get('nots'):
                    text += f" but not {', '.join(sorted(mark['nots'])[:3])}"
                text += f", while the requirements name {', '.join(sorted(named))}"
                if alternatives:
                    text += f'; seen ones that suit those, cheapest first: {self.priced(alternatives)}'
                found.append(text)
        return found

    def fit_faults(self) -> list[str]:
        index = self.fit_index()
        cart = self.cart_ids()
        found = []
        for part, main in self.bound:
            hosts = [i for i in cart if i in self.item_ids(main)]
            lines = [i for i in cart if i in self.item_ids(part)]
            if len(hosts) != 1 or not lines:
                continue
            host = hosts[0]
            codes = set(self.mark_of(index, host).get('codes') or [])
            if not codes:
                continue
            for line in lines:
                proven, unread = self.fit_candidates(part, codes, index, cart)
                tail = f'; unread rows naming it, best first (read one before relying on it): {self.priced(unread[:4])}' if unread else ''
                if self.fit_set(line, index):
                    if self.fits_codes(line, codes, index) is False:
                        found.append(self.repair_text(line, part, main, host, codes, index, proven, unread) + tail)
                    continue
                others_name = any(self.fit_set(i, index) for i in self.item_ids(part) if i != line)
                if proven or unread or others_name:
                    text = f"{line} ({self.items[part - 1].name}) names no model it suits, so nothing shows it suits {host} ({', '.join(sorted(codes))}), while other listings of that kind name theirs"
                    if proven:
                        text += f'; opened ones that name it and meet their own requirements, cheapest first: {self.priced(proven[:4])}'
                    found.append(text + tail)
        return found + self.series_faults(index, cart) + self.named_code_faults(index, cart)

    def reading_tools(self) -> list[str]:
        tools = []
        for candidate in (self.facts_tool(), self.offer_tool()):
            if candidate and candidate not in tools and self.single_target(candidate):
                tools.append(candidate)
        return tools

    def lines_unread(self) -> list[tuple[str, str]]:
        todo = []
        views = self.reading_tools()
        for line in self.cart_ids():
            if line in self.gone:
                continue
            for view in views:
                if view not in self.views_of.get(line, set()):
                    todo.append((view, line))
        return todo

    def read_cart_lines(self) -> str:
        todo = self.lines_unread()
        if self.finished or not todo or self.left() <= len(todo):
            return ''
        stamp = self.version
        texts = []
        for view, line in todo:
            problems = []
            call = _shape({self.single_target(view): line}, self.schemas.get(view), 'args', problems)
            if problems:
                continue
            reply = self.act(view, call)
            if self.finished or reply is None:
                break
            if view == self.offer_tool():
                self.offer_seen[line] = self.step
            self.observe(view, reply.get('observation'))
            texts.append(f"{view} {line}: {self.show(reply.get('observation'))[:600]}")
        self.trace.append({'kind': 'read_cart_lines', 'reads': todo})
        return '\n'.join(texts) if stamp != self.version else ''

    def page_tool_role(self) -> None:
        if self.roles.get('offer') is not None or self.single_target(self.roles.get('open')) is None:
            return
        assigned = set(self.roles.values())
        free = []
        for entry in self.tools:
            name = entry['function']['name']
            if name in assigned or not self.single_target(name) or self.word_field(name):
                continue
            about = (_words(name) + ' ' + str(entry['function'].get('description') or '')).lower()
            if re.search('\\b(?:add|remove|delete|cart|basket|bag|buy|checkout|purchase|order|message|send|ask|change|update|set)\\b', about):
                continue
            free.append(name)
        if len(free) == 1:
            self.roles['offer'] = free[0]
            self.trace.append({'kind': 'page_tool_role', 'tool': free[0]})

    def plain_needs(self, number: int) -> list[tuple[str, str]]:
        refs = [r for _, r in self.refs]
        out = []
        for owner, need in self.needs_now():
            prop, colon, value = need.partition(':')
            low = value.strip().lower()
            if owner == number and colon and low and need not in refs and not self.fit_need(number, need) and not re.search('\\d', low) and not re.search('\\b(?:not|no|except|without|other than|or|unlike|same)\\b', low):
                out.append((prop.strip(), low))
        return out

    def held_values(self, ident: str, prop: str, value: str) -> list[str]:
        facts = _facts_any(self.full.get(ident) or {}) or {}
        return [_chosen_other(facts, v, value) or str(v) for k, v in facts.items() if not _is_offer(k) and _names(_spelled(prop), _spelled(k))]

    def cart_of(self, number: int) -> list[str]:
        ids = set(self.item_ids(number))
        return [i for i in self.cart_ids() if i in ids]

    def priced(self, ids: list[str]) -> str:
        return ', '.join(f'{i} ({self.price_of(i):g})' if self.price_of(i) is not None else i for i in ids)

    def inside_longer(self) -> list[str]:
        out = []
        for number in range(1, len(self.items) + 1):
            for line in self.cart_of(number):
                for prop, value in self.plain_needs(number):
                    held = self.held_values(line, prop, value)
                    want = _word_bag(value)
                    if not held or any(_word_bag(t) == want or want in [_word_bag(e) for e in re.split('[,/;|]', t)] for t in held) or not any(want <= _word_bag(t) for t in held):
                        continue
                    exact = [i for i in self.ready_rows(number) if i != line and any(_word_bag(t) == want for t in self.held_values(i, prop, value))]
                    if exact:
                        out.append(f"{line}'s {prop} is '{held[0][:60]}', not just '{value}'; opened listings stating '{value}' exactly, cheapest first: " + self.priced(exact[:3]))
        return out

    def other_value_lines(self) -> list[str]:
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        out = []
        for number in range(1, len(self.items) + 1):
            needs = self.plain_needs(number)
            for line in self.cart_of(number) if needs else []:
                facts = _facts_any(self.full.get(line) or {}) or {}
                record = self.full.get(line) or self.rows.get(line) or {}
                for prop, value in needs:
                    held = self.held_values(line, prop, value)
                    if not held or any(_word_bag(value) & _word_bag(t) for t in held):
                        continue
                    if self.need_state(line, number, f'{prop}: {value}', facts, record) != 'conflict':
                        continue
                    fits = [i for i in self.ready_rows(number) if i != line]
                    if fits:
                        out.append(f"{line}'s {prop} is '{held[0][:60]}', which does not state '{value}'; opened listings meeting every requirement: " + self.priced(fits[:4]))
                    else:
                        words = _token_set(value)
                        rows = sorted((i for i in self.item_ids(number) if i != line and i not in self.full and self.buyable(i) and self.price_of(i) is not None and (cap is None or (self.price_of(i) or 0.0) <= cap + 1e-09) and words <= self.found_by.get(i, set()) | _token_set(_dump(self.rows.get(i) or {}))), key=lambda i: self.price_of(i) or 0.0)
                        out.append(f"{line}'s {prop} is '{held[0][:60]}', which does not state '{value}', and no opened listing of that item meets every requirement" + (': unread rows of that item that the value\'s words reached, cheapest first: ' + self.priced(rows[:4]) if rows else ''))
                    break
        return out

    def attr_ready(self, number: int, skip: str) -> list[str]:
        out = []
        for ident in self.ready_rows(number):
            if ident == skip:
                continue
            facts = {k: v for k, v in (_facts_any(self.full.get(ident) or {}) or {}).items() if not _is_offer(k)}
            if all(not colon or need in [r for _, r in self.refs] or self.fit_need(number, need) or any(_names(_spelled(prop), _spelled(k)) for k in facts) for owner, need in self.needs_now() if owner == number for prop, colon, _ in [need.partition(':')]):
                out.append(ident)
        return out

    def figure_shown(self, ident: str, prop: str, unit: str) -> bool:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        facts = {k: v for k, v in (_facts_any(record) or {}).items() if not _is_offer(k)}
        tops = {k: v for k, v in record.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if any(_names(_spelled(prop), _spelled(k)) and re.search('\\d', str(v)) for k, v in list(facts.items()) + list(tops.items())):
            return True
        text = ' '.join(str(v) for k, v in tops.items() if not _is_offer(k)) + ' ' + ' '.join(str(v) for v in facts.values())
        return any(_unit_like(m.group(2), unit) for m in re.finditer(FIGURE, text.lower()) if m.group(2))

    def plain_limits(self, number: int) -> list[tuple[str, str, tuple[bool, bool, float, str] | None]]:
        pointed = {r for _, r in self.refs}
        out = []
        for owner, need in self.needs_now():
            if owner != number or need in pointed or self.fit_need(number, need):
                continue
            prop, colon, value = need.partition(':')
            if colon:
                out.append((prop.strip(), value.strip(), _bound(value)))
        return out

    def no_attribute_lines(self) -> list[str]:
        by_price = self.brief.get('pick') == 'cheapest'
        found = []
        for number in range(1, len(self.items) + 1):
            needs = self.plain_limits(number)
            if not needs:
                continue
            for line in self.cart_of(number):
                facts = {k: v for k, v in (_facts_any(self.full.get(line) or {}) or {}).items() if not _is_offer(k)}
                if not facts:
                    continue
                stated_words = _token_set(' '.join(str(v) for v in facts.values()))
                missing, figureless = [], []
                for prop, value, limit in needs:
                    if not prop:
                        continue
                    named = any(_names(_spelled(prop), _spelled(k)) for k in facts)
                    words = _token_set(value)
                    if not named and (not words or not words <= stated_words):
                        missing.append(prop)
                    if limit and limit[3] and not self.figure_shown(line, prop, limit[3]):
                        figureless.append(prop)
                for prop in figureless:
                    if prop not in missing:
                        missing.append(prop)
                if not missing:
                    continue
                better = self.attr_ready(number, line)
                price = self.price_of(line)
                if by_price and price is not None and not figureless:
                    better = [i for i in better if (self.price_of(i) or 0.0) <= price]
                if better:
                    extra = f" (nor a figure for {', '.join(figureless)} anywhere on its record)" if figureless else ''
                    found.append(f"{line} carries no attribute for {', '.join(missing)}{extra}; opened listings of that item whose attributes state every requirement, cheapest first: {self.priced(better[:3])}")
        return found

    def no_figure_line(self) -> tuple[str, list[str], list[str], list[str]] | None:
        for number in range(1, len(self.items) + 1):
            limits = [(p, v, lim) for p, v, lim in self.plain_limits(number) if lim and lim[3]]
            if not limits:
                continue
            ceiling, count = self.money_room(number)

            def affordable(ident: str) -> bool:
                price = self.price_of(ident)
                return self.buyable(ident) and price is not None and (ceiling is None or price * count <= ceiling + 1e-09)
            mine = self.item_ids(number)
            for line in self.cart_of(number):
                if line not in self.full:
                    continue
                unshown = [(p, v, lim) for p, v, lim in limits if not self.figure_shown(line, p, lim[3])]
                if not unshown:
                    continue

                def shows_all(ident: str) -> bool:
                    return all(self.figure_shown(ident, p, lim[3]) for p, _, lim in unshown)
                opened = [i for i in mine if i != line and i in self.full and affordable(i) and shows_all(i) and not self.judge_record(i, number)[0]]
                opened.sort(key=lambda i: self.price_of(i) or 0.0)
                unread = [i for i in mine if i != line and i not in self.full and affordable(i)]
                unread.sort(key=lambda i: (not shows_all(i), self.price_of(i) or 0.0))
                if opened or unread:
                    return (line, [f'{p}: {v}' for p, v, _ in unshown], opened[:3], unread[:6])
        return None

    def note_reading(self, name: str, args: dict[str, Any], seen: Any) -> None:
        failed = isinstance(seen, dict) and seen.get('error')
        target = self.single_target(name)
        if target and not failed and isinstance(args.get(target), str):
            self.views_of.setdefault(args[target].strip(), set()).add(name)
        if name != self.roles.get('search'):
            return
        words = self.args_words(args)
        if not words:
            return
        for ident, _ in self.records_in(seen):
            self.found_by.setdefault(ident, set()).update(words)

    def plain_json(self, model: str, prompt: str, max_tokens: int) -> dict[str, Any] | None:
        if self.calls >= ALL_CALLS or self.model_room() < CALL_FLOOR:
            return None
        self.calls += 1
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': 0}
        data = _llm_within(self.model_room()).post('/inference/chat/completions', body)
        if not isinstance(data, dict) or not data.get('choices'):
            return None
        spent = (data.get('usage') or {}).get('cost')
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        return _json_obj((data['choices'][0].get('message') or {}).get('content'))

    def tool_props(self, tool: str | None) -> dict[str, Any]:
        schema = self.schemas.get(tool or '')
        props = schema.get('properties') if isinstance(schema, dict) else None
        return props if isinstance(props, dict) else {}

    def label_of(self, ident: str) -> str:
        cat_key = self.category_key()
        if not cat_key:
            return ''
        return str((self.rows.get(ident) or {}).get(cat_key) or '').strip()

    def query_width(self, tool: str, field: str) -> int:
        spec = self.tool_props(tool).get(field)
        size = spec.get('maxLength') if isinstance(spec, dict) else None
        return size if isinstance(size, int) else 60

    @staticmethod
    def lone(found: list[Any]) -> Any:
        return found[0] if len(found) == 1 else None

    def word_field(self, tool: str | None) -> str | None:
        found = []
        for key, spec in self.tool_props(tool).items():
            if isinstance(spec, dict) and 'enum' not in spec and 'string' in _kinds(spec) and re.search(QUERY_WORDS, _snake(key), re.I):
                found.append(key)
        return self.lone(found)

    def shelf_field(self, tool: str | None) -> str | None:
        words = self.word_field(tool)
        found = []
        for key, spec in self.tool_props(tool).items():
            if key != words and isinstance(spec, dict) and 'string' in _kinds(spec) and re.search(SHELF_ARG, _snake(key), re.I):
                found.append(key)
        return self.lone(found)

    def reading_mode(self, tool: str, key: str) -> Any:
        spec = self.tool_props(tool).get(key)
        choices = spec.get('enum') if isinstance(spec, dict) else None
        if key == self.shelf_field(tool) or not isinstance(choices, list) or len(choices) < 2:
            return None
        readers = [c for c in choices if isinstance(c, str) and re.search(QUERY_WORDS, c, re.I)]
        blind = self.word_blind.get(tool)
        if not blind or blind[0] != key:
            return self.lone(readers)
        return self.lone([c for c in (readers or choices) if c != blind[1]])

    def word_tool(self) -> tuple[str, dict[str, Any], str, str | None] | None:
        tool = self.roles.get('search')
        if self.word_field(tool) is None:
            used = set(self.roles.values())
            tool = self.lone([t for t in self.schemas if t not in used and self.word_field(t)])
        field = self.word_field(tool) if tool else None
        if field is None:
            return None
        needed = (self.schemas.get(tool) or {}).get('required') or []
        shelf = self.shelf_field(tool)
        if shelf is not None and shelf in needed:
            return None
        fixed = {}
        for key, spec in self.tool_props(tool).items():
            if key == field or key == shelf or not isinstance(spec, dict):
                continue
            mode = self.reading_mode(tool, key)
            if mode is None and key in needed:
                return None
            if mode is not None:
                fixed[key] = mode
        return (tool, fixed, field, shelf)

    def probe(self, tool: str, args: dict[str, Any], limit: int = 4000) -> tuple[dict[str, Any], Any, list[str], str] | None:
        if self.finished:
            return None
        problems = []
        args = _shape(args, self.schemas.get(tool), 'args', problems)
        reply = None if problems else self.act(tool, args)
        if reply is None:
            return None
        result = reply.get('observation')
        self.observe(tool, result)
        echo = self.echo_note(tool, args, result)
        text = self.show(result)
        self.read_cache[tool + _dump(args)] = (self.version, self.step, text)
        ids = [ident for ident, _ in self.records_in(result)]
        if ids:
            self.searched = (self.searched + [dict(args)])[-8:]
        if len(text) > limit:
            text = text[:limit] + '...(cut)'
        return (args, result, ids, f'{_dump(args)} (step {self.step}): {text}{echo}')

    @staticmethod
    def trim_words(words: list[str], width: int, keep: int) -> list[str]:
        kept = [w for w in words if w]
        while len(kept) > 1 and len(' '.join(kept)) > width:
            if keep >= 2:
                kept.pop(len(kept) - keep)
                keep -= 1
            elif len(kept) > keep:
                kept.pop(len(kept) - keep - 1)
            else:
                break
        return kept

    @staticmethod
    def dedupe_words(words: list[str]) -> list[str]:
        out = []
        seen = set()
        for w in words:
            if w and w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
        return out

    def short_noun(self, number: int) -> list[list[str]]:
        groups = []
        for chunk in re.split('\\s*(?:,|&|/|\\band\\b|\\bor\\b)\\s*', self.items[number - 1].name):
            words = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'-]*", chunk) if w.lower() not in LIGHT_WORDS]
            if words:
                groups.append(words)
        if not groups:
            return [['product']]

        def head(words: list[str]) -> list[str]:
            tail = _singular(words[-1])
            if len(words) > 1 and len(tail) <= 4:
                return [words[-2], tail]
            return [tail]
        heads = [head(groups[-1])]
        if len(groups) > 1:
            front = head(groups[0])
            if len(groups[-1]) == 1 or len(groups[0]) > 1:
                heads = [front] + heads
            else:
                heads = heads + [front]
        unique = []
        for words in heads:
            if all(not _same_words(words, u) for u in unique):
                unique.append(words)
        return unique

    def item_values(self, number: int) -> list[list[str]]:
        noun_roots = _roots(' '.join(' '.join(words) for words in self.short_noun(number)))
        found = []
        for pair in self.value_words(number):
            terms = []
            for w in _short_terms(pair[0]):
                if not _roots(w) or not _roots(w) <= noun_roots:
                    terms.append(w)
            if terms and all(not _same_words(terms, f) for f in found):
                found.append(terms)

        def rank(terms: list[str]) -> tuple[bool, bool, int]:
            has_figure = any(re.search('\\d', t) for t in terms)
            return (len(terms) > 1, not has_figure, len(terms))
        return sorted(found, key=rank)

    def opening_queries(self, number: int, width: int, shelved: bool) -> list[list[str]]:
        values = self.item_values(number)[:2]
        heads = self.short_noun(number)
        noun = heads[0]
        named = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'-]*", self.items[number - 1].name) if w.lower() not in LIGHT_WORDS and w.lower() != 'or']
        plain = noun
        if len(noun) == 1 and len(named) > 1 and _singular(named[-1]).lower() == noun[0].lower():
            plain = [named[-2], noun[0]]
        if shelved and values and len(heads) > 1:
            noun = []
        drafts = []
        if len(values) > 1 and sum(len(v) for v in values[:2]) + len(noun) < 7:
            drafts.append(values[0] + values[1] + noun)
        if values:
            drafts.append(values[0] + noun)
        drafts.append(plain)
        drafts.extend(heads[1:])
        result = []
        for draft in drafts:
            words = self.trim_words(self.dedupe_words(draft), width, len(noun))
            if words and all(not _same_words(words, r) for r in result):
                result.append(words)
        return result

    def broad_query(self, number: int, width: int) -> list[str]:
        stated = list(dict.fromkeys(v for v, _ in self.value_words(number)))
        stated.sort(key=lambda v: len(v.split()))
        said = ' '.join(stated[:2]).split()
        if not said:
            return []
        taken = {w.lower() for w in said}
        noun = []
        for w in re.findall("[A-Za-z0-9][A-Za-z0-9'.-]*", self.items[number - 1].name):
            if w.lower() not in LIGHT_WORDS and w.lower() != 'or' and w.lower() not in taken:
                noun.append(w)
        noun = noun or ['product']
        return self.trim_words(said + noun, width, len(noun))

    def shelf_value(self, tool: str, number: int) -> str | None:
        shelf = self.shelf_field(tool)
        if shelf is None:
            return None
        item = self.items[number - 1]
        tally = {}
        for ident in self.item_ids(number):
            name = self.label_of(ident)
            if name:
                tally[name] = tally.get(name, 0) + 1
        home = self.home_label(number)
        seen = sorted(tally, key=lambda name: (_words(name) != home, -tally[name]))
        spec = self.tool_props(tool).get(shelf)
        choices = spec.get('enum') if isinstance(spec, dict) else None
        if not isinstance(choices, list):
            return seen[0] if seen else None
        texts = [c for c in choices if isinstance(c, str)]
        for want in [item.shelf, item.name, *seen]:
            if not want:
                continue
            for choice in texts:
                if _words(choice) == _words(want):
                    return choice
        roots = _roots(item.name) | _roots(item.shelf or '')
        return self.lone([c for c in texts if self.owner_of(c) == number and roots & _roots(c)])

    def row_has(self, ident: str, words: list[str]) -> bool:
        wanted = [w for w in words if w]
        if not wanted:
            return False
        parts = [str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        text = ' '.join(parts)
        pool = _token_set(text) | _roots(text)
        packed = re.sub('[^a-z0-9]', '', text.lower())
        for w in wanted:
            roots = _roots(w)
            if not ((roots and roots <= pool) or w.lower() in pool or re.sub('[^a-z0-9]', '', w.lower()) in packed):
                return False
        return True

    def kind_match(self, ident: str, number: int) -> bool:
        name = self.label_of(ident)
        if name:
            return self.owns(self.items[number - 1], name)
        texts = [str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str)]
        bag = _roots(' '.join(texts))
        return any(bag & _roots(' '.join(head)) for head in self.short_noun(number))

    def deep_pages(self, page_size: int) -> int:
        steps = self.max_steps
        if steps <= 40:
            wanted = 15
        elif steps >= 80:
            wanted = 30
        else:
            wanted = 15 + (steps - 40) * 15 // 40
        size = max(1, page_size)
        return max(2, (wanted + size - 1) // size)

    def page_shows_value(self, observation: Any, number: int) -> bool:
        values = self.item_values(number)
        if not values:
            return False
        for ident, _ in self.records_in(observation):
            if self.kind_match(ident, number) and any(self.row_has(ident, v) for v in values):
                return True
        return False

    def cheapest_known(self, number: int) -> float:
        best = None
        for ident in self.item_ids(number):
            price = self.price_of(ident)
            if price is not None and self.buyable(ident) and (best is None or price < best):
                best = price
        return best or 0.0

    def capped_rows(self, number: int, rows: list[str]) -> tuple[str, dict[str, Any]] | None:
        cheapest = self.brief.get('pick') == 'cheapest'
        has_budget = isinstance(self.budget, (int, float))
        plan = self.cap_template() if cheapest or has_budget else None
        if not plan or not self.category_key():
            return None
        tool, fixed, shelf, cap_key = plan
        item = self.items[number - 1]
        home = self.home_label(number)
        votes = {}
        for ident in rows:
            name = self.label_of(ident)
            if name and _words(name) == home and self.owns(item, name):
                votes[name] = votes.get(name, 0) + 1
        if not votes:
            return None
        shelf_name = max(votes, key=votes.get)
        values = self.item_values(number)
        prices = []
        for ident in rows:
            price = self.price_of(ident)
            if price is None or not self.buyable(ident) or not self.kind_match(ident, number):
                continue
            if values and not any(self.row_has(ident, v) for v in values):
                continue
            prices.append(price)
        limit = min(prices) if cheapest and prices else None
        if limit is None and has_budget:
            others = sum(self.cheapest_known(n) * it.qty for n, it in enumerate(self.items, 1) if n != number)
            limit = (self.budget - others) / max(1, item.qty)
            kind_prices = [self.price_of(i) for i in rows if self.price_of(i) is not None and self.kind_match(i, number)]
            if not cheapest and (not kind_prices or min(kind_prices) <= limit):
                return None
        if not limit or limit <= 0:
            return None
        spec = self.tool_props(tool).get(cap_key) or {}
        whole = 'integer' in _kinds(spec) or float(limit).is_integer()
        return (tool, {**fixed, shelf: shelf_name, cap_key: int(limit) if whole else round(limit, 2)})

    def open_item(self, plan: tuple[str, dict[str, Any], str, str | None], width: int, number: int, share: int, spare: Any) -> list[str]:
        tool, fixed, field, shelf = plan
        item = self.items[number - 1]
        tag = f'item {number} ({item.name})'
        shelf_name = self.shelf_value(tool, number) if shelf else None
        base = dict(fixed)
        if shelf_name:
            base[shelf] = shelf_name
        notes = []
        rows = []
        sent = set()
        first = None
        tries = self.opening_queries(number, width, bool(shelf_name))
        broad = self.broad_query(number, width)
        if broad and self.max_steps <= 40 and all(not _same_words(broad, t) for t in tries):
            tries.insert(0, broad)
        for words in tries:
            if not spare():
                break
            sent.add(' '.join(words).lower())
            got = self.probe(tool, {**base, field: ' '.join(words)})
            if got is None:
                break
            notes.append(f'{tag}: {got[3]}')
            rows.extend(got[2])
            if got[2]:
                first = (words, got)
                break
        began = self.step

        def own_room(extra: int = 0) -> bool:
            return spare(extra) and self.step - began + extra < share
        if first is None:
            if shelf_name and spare():
                listing = dict(fixed)
                for key in fixed:
                    spec = self.tool_props(tool).get(key) or {}
                    modes = [o for o in spec.get('enum') or [] if isinstance(o, str) and o != fixed[key] and re.search(LIST_MODE, o, re.I)]
                    if len(modes) == 1:
                        listing[key] = modes[0]
                if not fixed or listing != fixed:
                    got = self.probe(tool, {**listing, shelf: shelf_name})
                    if got is not None:
                        notes.append(f'{tag}, its category without words: {got[3]}')
            return notes
        alternates = [broad] + self.opening_queries(number, width, bool(shelf_name))[:1]
        for words in alternates:
            if not words or ' '.join(words).lower() in sent or not own_room():
                continue
            sent.add(' '.join(words).lower())
            got = self.probe(tool, {**base, field: ' '.join(words)})
            if got is not None:
                rows.extend(got[2])
                notes.append(f'{tag}, other wording: {got[3]}')
        noun = self.short_noun(number)[0]
        covered = {w.lower() for w in first[0]}
        for words in self.item_values(number)[:3]:
            if not own_room():
                continue
            lowered = {w.lower() for w in words}
            if lowered <= covered or any(self.row_has(i, words) for i in rows):
                continue
            if len(words) > 1 and not any(re.search('\\d', w) for w in words):
                continue
            query = self.trim_words(words + noun, width, len(noun))
            if ' '.join(query).lower() in sent:
                continue
            got = self.probe(tool, {**base, field: ' '.join(query)})
            if got is None:
                break
            sent.add(' '.join(query).lower())
            covered |= lowered
            rows.extend(got[2])
            notes.append(f'{tag}, one value: {got[3]}')
        extra_words = _token_set(' '.join(first[0])) - _token_set(item.name) - _token_set(' '.join(noun))
        if extra_words and spare():
            notes.extend(self.page_on(tool, first[1], number, rows, spare, own_room))
        capped = self.capped_rows(number, rows)
        if capped and own_room():
            got = self.probe(capped[0], capped[1])
            if got is not None:
                notes.append(f'{tag}, its category up to a price: {got[3]}')
                page = self.paging(capped[0], got[1])
                if page and page[3] is not None and page[0] > page[1] and own_room() and not self.price_sorted(got[1]):
                    nxt = self.probe(capped[0], {**got[0], page[2]: page[3]})
                    if nxt is not None:
                        notes.append(f'item {number}, its capped list, next page: {nxt[3]}')
        return notes

    def page_on(self, tool: str, found: tuple[dict[str, Any], Any, list[str], str], number: int, rows: list[str], spare: Any, own_room: Any) -> list[str]:
        args, result = found[0], found[1]
        page = self.paging(tool, result)
        if not page:
            return []
        most = self.deep_pages(page[1])
        reserve = 2 if self.capped_rows(number, rows) else 0
        notes = []
        read = 1
        while page and page[3] is not None and page[0] > page[1] * read and read < most:
            if not (spare() if read == 1 else own_room(reserve)):
                break
            nxt = self.probe(tool, {**args, page[2]: page[3]})
            if nxt is None:
                break
            read += 1
            self.pages_read += 1
            rows.extend(nxt[2])
            notes.append(f'item {number}, page {read}: {nxt[3]}')
            if not self.page_shows_value(nxt[1], number):
                break
            page = self.paging(tool, nxt[1])
        return notes

    def fit_notes(self, notes: list[str], total: int) -> list[str]:
        sizes = sorted(len(n) for n in notes)
        each = total
        used = 0
        for index, size in enumerate(sizes):
            each = (total - used) // max(1, len(sizes) - index)
            if size > each:
                break
            used += size
        each = max(300, each)
        out = []
        for note in notes:
            if len(note) <= each:
                out.append(note)
                continue
            ids = re.findall('"' + re.escape(self.id_key) + '":"([^"]+)"', note) if self.id_key else []
            out.append(note[:each] + '...(cut; listings in it: ' + ', '.join(dict.fromkeys(ids)) + ')')
        return out

    def first_searches(self) -> str:
        plan = self.word_tool()
        if plan is None or self.store_opened or self.finished or self.model_room() < 150:
            return ''
        self.store_opened = True
        tool, _, field, _ = plan
        width = self.query_width(tool, field)
        hops = self.hops_due()
        if self.max_steps <= 40:
            cap = max(4, self.max_steps // 5)
        else:
            cap = self.max_steps // 4
        cap += 2 if hops else 0
        began = self.step

        def spare(extra: int = 0) -> bool:
            if self.finished or self.left() <= 20 or self.model_room() <= 120:
                return False
            return self.step - began + extra < cap
        notes = self.hop_openers(plan, width, hops, spare)
        todo = [n for n in range(1, len(self.items) + 1) if self.items[n - 1].musts or self.items[n - 1].name != 'the requested product']
        for place, number in enumerate(todo):
            if not spare():
                break
            left_items = max(1, len(todo) - place)
            notes.extend(self.open_item(plan, width, number, max(2, (cap - (self.step - began)) // left_items), spare))
        if not notes:
            return ''
        notes = self.fit_notes(notes, 24000)
        if self.queued:
            notes.append('[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        self.values_sought = True
        self.trace.append({'kind': 'first_searches', 'steps': self.step - began, 'searches': len(notes)})
        head = '[Before your first turn Halo ran these searches (one step each): each item with the values the request states, inside its category when the search accepts one, and the shop of every listing a requirement points at]'
        return head + '\n' + '\n'.join(notes)

    def page_arg(self, args: dict[str, Any]) -> str | None:
        for key, value in args.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if re.search('page|leaf', key, re.I) and not re.search('size|per|limit|count', key, re.I):
                return key
        return None

    @staticmethod
    def page_sig(name: str, args: dict[str, Any], key: str | None) -> str:
        kept = {}
        for k, v in args.items():
            if k != key and v not in (None, '', [], {}):
                kept[k] = v
        return name + json.dumps(kept, sort_keys=True, default=str)

    @staticmethod
    def args_words(args: dict[str, Any]) -> set[str]:
        bag = set()
        for key, value in args.items():
            if isinstance(value, str) and re.search(QUERY_WORDS, _snake(key), re.I):
                bag |= _token_set(value)
        return bag

    def other_args(self, name: str, args: dict[str, Any], *skip: Any) -> str:
        return _dump(dict(sorted((k, v) for k, v in args.items() if k not in skip)))

    def page_field(self, name: str, args: dict[str, Any]) -> str | None:
        return (self.shapes.get(name) or {}).get('next_arg') or self.page_arg(args)

    def search_sig(self, name: str, args: dict[str, Any]) -> tuple[str, Any, list[str]] | None:
        field = self.word_field(name)
        words = str(args.get(field) or '').lower().split() if field is not None else []
        if not words:
            return None
        page = self.page_field(name, args)
        offset = args.get(page) if page else None
        return (self.other_args(name, args, field, page), offset if offset not in (None, '') else 0, words)

    def odd_modes(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        odd = {}
        for key, value in args.items():
            mode = self.reading_mode(tool, key)
            if mode is not None and mode != value:
                odd[key] = value
        return odd

    def matches_seen(self, name: str, observation: Any) -> int | None:
        if not self.shapes.get(name):
            return None
        page = self.paging(name, observation)
        return page[0] if page else None

    def blind_note(self, name: str, args: dict[str, Any], rows: tuple[str, ...], observation: Any) -> str:
        modes = self.odd_modes(name, args)
        if len(rows) < 2 or not modes:
            return ''
        key = next(iter(modes))
        value = modes[key]
        field = self.word_field(name)
        words = tuple(sorted(_token_set(args.get(field)))) if field else ()
        if self.word_blind.get(name) == (key, value):
            if not words or self.blind_notes >= 3:
                return ''
            self.blind_notes += 1
            return f'\n[Halo] {key}={_dump(value)} ignores search words (found earlier), so this shows the whole filter; only {key}={_dump(self.reading_mode(name, key))} lets words narrow it.'
        page = self.page_field(name, args)
        entry = {'tool': name, 'rest': self.other_args(name, args, field, page), 'offset': args.get(page) if page else None, 'words': words, 'rows': rows, 'count': self.matches_seen(name, observation), 'step': self.step}
        twin = None
        for old in self.blind_log:
            same_list = (old['tool'], old['rest'], old['offset'], old['rows']) == (entry['tool'], entry['rest'], entry['offset'], entry['rows'])
            same_count = old['count'] is None or entry['count'] is None or old['count'] == entry['count']
            if same_list and same_count and old['words'] != words:
                twin = old
                break
        self.blind_log = (self.blind_log + [entry])[-30:]
        if twin is None:
            return ''
        self.word_blind[name] = (key, value)
        other = self.reading_mode(name, key)
        self.trace.append({'kind': 'word_blind', 'tool': name, 'mode': f'{key}={value}', 'words': ' '.join(words), 'twin': ' '.join(twin['words'])})
        before = f"the search for {' '.join(twin['words'])!r} at step {twin['step']}" if twin['words'] else f"the same filter with no words at step {twin['step']}"
        hint = '' if other is None else f' Only {key}={_dump(other)} lets words narrow a search.'
        return f'\n[Halo] Under {key}={_dump(value)} the words of a search are not read: this gave exactly the rows and count of {before}.{hint}'

    def echo_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        rows = tuple(ident for ident, _ in self.records_in(observation))
        blind = self.blind_note(name, args, rows, observation)
        sig = self.search_sig(name, args)
        if sig is None or len(rows) < 3 or self.odd_modes(name, args):
            return blind
        rest, offset, words = sig
        count = self.matches_seen(name, observation)
        note = ''
        for old in self.echo_log:
            if old['tool'] != name or old['rest'] != rest or old['offset'] != offset or old['rows'] != rows or old['words'] == words:
                continue
            if count is not None and old['count'] is not None and count != old['count']:
                continue
            if sorted(words) != sorted(old['words']):
                lead = 0
                for a, b in zip(words, old['words']):
                    if a != b:
                        break
                    lead += 1
                if 0 < lead < min(len(words), len(old['words'])):
                    self.lead_count = lead if not self.lead_count else min(self.lead_count, lead)
                    self.echo_hits += 1
                    self.trace.append({'kind': 'echo', 'words': ' '.join(words), 'twin': ' '.join(old['words']), 'reads': lead})
                    note = f"\n[Halo] Same rows and same count as the search {' '.join(old['words'])!r} at step {old['step']}: this store's word search only reads the first {lead} word(s) of a query, so later words do nothing. Halo did not page this repeat. Begin each new search with a word that no earlier search began with, the one that narrows it most."
            break
        self.echo_log = (self.echo_log + [{'tool': name, 'rest': rest, 'offset': offset, 'words': words, 'rows': rows, 'count': count, 'step': self.step}])[-40:]
        return note

    def lead_repeat(self, name: str, args: dict[str, Any]) -> str:
        if not self.lead_count or self.echo_hits < 2:
            return ''
        sig = self.search_sig(name, args)
        if sig is None:
            return ''
        rest, offset, words = sig
        lead = words[:self.lead_count]
        twins = [e for e in self.echo_log if e['tool'] == name and e['rest'] == rest and e['offset'] == offset and e['words'][:self.lead_count] == lead]
        if not twins or any(e['words'] == words for e in twins):
            return ''
        last = twins[-1]
        return f"{' '.join(last['words'])!r} at step {last['step']}"

    def list_stop(self, name: str, args: dict[str, Any], observation: Any) -> str:
        ranked = self.brief.get('pick') in ('cheapest', 'most', 'least')
        page = self.paging(name, observation) if ranked else None
        if page is None or page[3] is not None:
            return ''
        total, size, param = page[0], page[1], page[2]
        offset = args.get(param) or 0
        if isinstance(offset, bool) or not isinstance(offset, int) or offset % size != 0:
            return ''
        reached = offset + size
        if reached >= total:
            return ''
        sig = self.search_sig(name, args)
        label = repr(' '.join(sig[2])) if sig else 'this list'
        mark = name + self.other_args(name, args, param)
        if mark in self.stopped_lists:
            return ''
        self.stopped_lists.add(mark)
        self.trace.append({'kind': 'list_stop', 'list': label, 'reached': reached, 'count': total})
        return f'\n[Halo] {label} ends here at {reached} of {total} matches with no next page, so the remaining {total - reached} can only turn up in other searches. The pick has to weigh every fitting listing: search again beginning with different words (another name for the product, a word from its store label, a value fitting listings carry) and watch for rows that may beat your current best.'

    def page_limit_note(self, name: str, faults: list[str]) -> str:
        if name != self.roles.get('search'):
            return ''
        page_key = (self.shapes.get(name) or {}).get('next_arg')
        for fault in faults:
            found = re.match('args\\.(\\w+) is above ', fault)
            if not found:
                continue
            arg = found.group(1)
            paged = arg == page_key or (re.search('page|offset|start|begin|skip|from|cursor', arg, re.I) and not re.search('size|per|limit|count', arg, re.I))
            if paged:
                return '. This store does not page a list that far, so rows beyond that point only appear in other searches: begin those with different words (another name for the product, a word from its store label, a value fitting listings carry).'
        return ''

    def page_allowance(self) -> int:
        return min(AUTO_PAGES, max(3, self.max_steps // 6))

    def wordless(self, name: str, args: dict[str, Any]) -> bool:
        words = self.args_words(args)
        blind = self.word_blind.get(name)
        if not words or (blind is not None and args.get(blind[0]) == blind[1]):
            return True
        names = set()
        for it in self.items:
            names |= _roots(it.name)
            names |= _roots(it.shelf or '')
        shelf = self.shelf_field(name)
        if shelf and isinstance(args.get(shelf), str):
            names |= _roots(args[shelf])
        for ident in self.rows:
            names |= _roots(self.label_of(ident))
        meaningful = _roots(' '.join(sorted(words)))
        meaningful |= {w for w in words if re.search('\\d', w)}
        return not meaningful or meaningful <= names

    def stated_values_once(self) -> str:
        if self.values_sought or not self.searched or self.left() < 22 or self.model_room() < 60:
            return ''
        self.values_sought = True
        used = set()
        for args in self.searched:
            used |= self.args_words(args)
        due = []
        for number in range(1, len(self.items) + 1):
            pairs = self.value_words(number)
            fresh = len(pairs) > 1 or (len(pairs) == 1 and not _token_set(pairs[0][0]) <= used)
            proven = any(self.states_needs(i, number) for i in self.item_ids(number) if i in self.full)
            if fresh and not proven:
                due.append(number)
        if not due:
            return ''
        return self.value_search(due, from_request=True)

    def tell(self, text: str, short: bool = False) -> None:
        if not text:
            return
        last = self.chat[-1] if self.chat else {}
        if last.get('role') in ('tool', 'user') and isinstance(last.get('content'), str):
            status = self.status().split('\n', 1)[0] if short else ''
            last['content'] = '\n'.join(x for x in (last['content'], text, status) if x)
        else:
            self.chat.append({'role': 'user', 'content': f'{text}\n{self.status()}'})

    @staticmethod
    def cap_named(key: str) -> bool:
        name = _snake(key)
        return bool(re.search(CAP_ARG, name, re.I) or re.search('budget|max_?price|(?:^|_)price_?(?:cap|limit|max)', name, re.I))

    def more_value_pages(self, tool: str, args: dict[str, Any], observation: Any, number: int, budget: int) -> tuple[list[Any], int]:
        pages = [observation]
        page = self.paging(tool, observation)
        cap = 3 if not page else max(3, self.deep_pages(page[1]))
        spent = 0
        while page and page[3] is not None and len(pages) < cap:
            short = page[0] <= 3 * page[1]
            if not (short or self.page_shows_value(pages[-1], number)):
                break
            if budget - spent <= 3 or self.left() <= self.tail() + 18:
                break
            problems = []
            call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', problems)
            reply = None if problems else self.act(tool, call)
            spent += 1
            if reply is None:
                break
            result = reply.get('observation')
            self.observe(tool, result)
            pages.append(result)
            page = self.paging(tool, result)
        return (pages, spent)

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
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
