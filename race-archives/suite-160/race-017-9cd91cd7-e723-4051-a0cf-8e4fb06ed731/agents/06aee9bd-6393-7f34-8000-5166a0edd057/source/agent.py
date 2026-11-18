from __future__ import annotations
import json
import random
import re
import time
from functools import lru_cache
from typing import Any
from src.agent.proxy_client import ProxyClient

class _Branch(Exception):
    pass
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
HEAVY_KIN = (HEAVY, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.052
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
EPISODE_SECONDS = 300
EXPLORE_RESERVE = 50
CALL_RESERVE = 15
TIGHT_MESSAGES = 2
RECOVERY_STEPS = 8
BUSY_SECONDS = 20
SWEEP_STEPS = 6
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"browse": how a tool lists one store category, when one can: {"tool": exact tool name, "category": the argument that takes the category label, "max_price": the argument that takes a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- The value is only what the property should be. Words that merely join a property to its value (such as matching, set to or coming in as) are not part of it: "size set to large" states large. This does not touch a value pointed at on another listing, which is copied whole.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nThe requirements were written down by a quick reader, so first compare each one with the REQUEST and the ANSWERS FROM THE SHOPPER. A requirement that only repeats the story (who it is for, when, where or why it will be used, what went wrong before, how it should feel) and that the request does not demand of the product goes under "not_firm". A value that picked up words which only join a property to its value goes under "corrected" with the plain value. Judge the listings without the not_firm requirements and with the corrected values. A requirement the request or an answer states outright stays, even when no listing meets it: ending without a purchase is then right.\nReturn ONE JSON object and nothing else: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "search": [at most 4 of {"item": item number, "query": "a keyword search of two or three words, not tried yet, that could surface a fitting listing: another name for the product with one required value (long queries often match nothing)"}], "reason": "one sentence"}'
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
FIT_CODE = '[A-Z][A-Za-z]{0,7}(?:[- ]?\\d{3,6}|[- ]?\\d{1,5}[xX]{1,4})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
PART_SERIES = '[A-Z]{1,5}\\d{1,3}'
FIT_WORDS = '(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|supports?)'
FIT_EXCEPT = '\\b(?:except(?:\\s+for)?|excluding|other than|but not|apart from|not including|not for)\\b'

def _codes(text: Any) -> list[str]:
    return [re.sub('[\\s-]+', '', c).lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _code_names(stated: str, code: str) -> bool:
    """Whether a stated code or series names a model code: a series ends in x's in place of digits."""
    tail = re.search('(?<=\\d)x+$', stated)
    if not tail:
        return stated == code
    return code.startswith(stated[:tail.start()]) and len(code) == len(stated)

def _fold_line66(code: str) -> str:
    found = re.match('[a-z]+', code)
    return found.group(0) if found else ''

@lru_cache(maxsize=4096)
def _lift_heap(text: str) -> dict[str, Any]:
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    _seq1 = list(re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I))
    _ix2 = 0
    while _ix2 < len(_seq1):
        hit = _seq1[_ix2]
        _ix2 = _ix2 + 1
        codes += _codes(hit.group(1))
    _seq3 = list(re.finditer('\\b(?:also sold as|also known as|sold also as|sold as)\\s+([^.;]+)', text, re.I))
    _ix4 = 0
    while _ix4 < len(_seq3):
        hit = _seq3[_ix4]
        _ix4 = _ix4 + 1
        codes += _codes(hit.group(1))
    works: list[str] = []
    deny: list[str] = []
    _seq5 = list(re.finditer('\\b(?<!not )(?<!never )' + FIT_WORDS + '\\s*:?\\s+([^.;]+)', text, re.I))
    _ix6 = 0
    while _ix6 < len(_seq5):
        hit = _seq5[_ix6]
        _ix6 = _ix6 + 1
        try:
            True if re.search('\\bpart\\s+' + PART_CODE + '\\s+(?:also\\s+)?$', text[max(0, hit.start() - 40):hit.start()], re.I) or re.match('part\\s+series', hit.group(1), re.I) else (_ for _ in ()).throw(_Branch())
            continue
        except _Branch:
            pass
        sides = re.split(FIT_EXCEPT, hit.group(1), maxsplit=1, flags=re.I)
        works += _codes(sides[0])
        try:
            True if len(sides) > 1 else (_ for _ in ()).throw(_Branch())
            deny += _codes(sides[1])
        except _Branch:
            pass
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    series: list[str] = []
    _seq7 = list(re.finditer('\\bfits?\\s+part\\s+series\\s*:?\\s*([^.;]+)', text, re.I))
    _ix8 = 0
    while _ix8 < len(_seq7):
        hit = _seq7[_ix8]
        _ix8 = _ix8 + 1
        series += [c.upper() for c in re.findall('\\b' + PART_SERIES + '\\b', hit.group(1))]
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'deny': deny, 'also': also, 'series': series}

def _fit_verdict(part: dict[str, Any], device: dict[str, Any], allowed: set[str]) -> bool | None:
    """True or False when a part's statements can be held against this device, None when they say nothing about it.
    A part is judged only against a device whose code is of the family its statements name."""
    if device.get('series') and re.match(PART_SERIES + '-', part.get('own') or ''):
        return any((part['own'] == s or part['own'].startswith(s + '-') for s in device['series']))
    try:
        True if not allowed or not device.get('codes') else (_ for _ in ()).throw(_Branch())
        return None
    except _Branch:
        pass
    families = {_fold_line66(c) for c in allowed}
    codes = [c for c in device['codes'] if _fold_line66(c) in families]
    if not codes:
        return None
    try:
        True if any((_code_names(d, c) for d in part.get('deny') or [] for c in codes)) else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return any((_code_names(a, c) for a in allowed for c in codes))

def _fetch_zone(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _wrap_line(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    try:
        True if isinstance(value, str) else (_ for _ in ()).throw(_Branch())
        return value if len(value) <= limit else value[:limit] + '...'
    except _Branch:
        pass
    if isinstance(value, dict):
        return {k: _wrap_line(v, limit, depth + 1) for k, v in value.items()}
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        return [_wrap_line(v, limit, depth + 1) for v in value]
    except _Branch:
        pass
    return value

def _check_node52(text: Any) -> dict[str, Any] | None:
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

def _open_pool(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    _seq9 = list(_words(text).split())
    _ix10 = 0
    while _ix10 < len(_seq9):
        w = _seq9[_ix10]
        _ix10 = _ix10 + 1
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _push_grid(schema: dict[str, Any]) -> list[str]:
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
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    try:
        True if kind == 'array' else (_ for _ in ()).throw(_Branch())
        return isinstance(value, list)
    except _Branch:
        pass
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _push_grid(schema)
    try:
        True if kinds and (not any((_fits_kind(value, k) for k in kinds))) else (_ for _ in ()).throw(_Branch())
        _seq11 = list(kinds)
        _ix12 = 0
        while _ix12 < len(_seq11):
            kind = _seq11[_ix12]
            _ix12 = _ix12 + 1
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
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            try:
                True if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')) else (_ for _ in ()).throw(_Branch())
                value = value.strip().lower() == 'true'
                break
            except _Branch:
                pass
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
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
            try:
                True if match else (_ for _ in ()).throw(_Branch())
                value = match[0]
            except _Branch:
                pass
        except _Branch:
            pass
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    except _Branch:
        pass
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    try:
        True if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']) else (_ for _ in ()).throw(_Branch())
        faults.append(f'{where} is too short')
    except _Branch:
        pass
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        try:
            True if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems'] else (_ for _ in ()).throw(_Branch())
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        except _Branch:
            pass
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        try:
            True if isinstance(schema.get('items'), dict) else (_ for _ in ()).throw(_Branch())
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
        except _Branch:
            pass
    except _Branch:
        pass
    try:
        True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        try:
            True if schema.get('additionalProperties') is False else (_ for _ in ()).throw(_Branch())
            _seq13 = list([n for n in value if n not in props])
            _ix14 = 0
            while _ix14 < len(_seq13):
                name = _seq13[_ix14]
                _ix14 = _ix14 + 1
                faults.append(f'{where}.{name} is not an allowed argument')
        except _Branch:
            pass
        _seq15 = list(schema.get('required') or [])
        _ix16 = 0
        while _ix16 < len(_seq15):
            name = _seq15[_ix16]
            _ix16 = _ix16 + 1
            try:
                True if name not in value else (_ for _ in ()).throw(_Branch())
                faults.append(f'{where}.{name} is required')
            except _Branch:
                pass
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    except _Branch:
        pass
    return value

def _snap_stash86(value: Any) -> list[Any]:
    try:
        True if isinstance(value, list) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
    try:
        True if isinstance(value, (str, dict)) and value else (_ for _ in ()).throw(_Branch())
        return [value]
    except _Branch:
        pass
    return []

def _bind_span(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        True if isinstance(value, int) else (_ for _ in ()).throw(_Branch())
        return value
    except _Branch:
        pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        True if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value) else (_ for _ in ()).throw(_Branch())
        return int(float(value))
    except _Branch:
        pass
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    try:
        True if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip() else (_ for _ in ()).throw(_Branch())
        return False
    except _Branch:
        pass
    return named or bool(re.search('\\d', value))

def _wrap_delta24(rows: list[dict[str, Any]]) -> str | None:
    best = None
    _seq17 = list(rows[0])
    _ix18 = 0
    while _ix18 < len(_seq17):
        key = _seq17[_ix18]
        _ix18 = _ix18 + 1
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        try:
            True if len(rows) > 1 and len(set(values)) < len(values) else (_ for _ in ()).throw(_Branch())
            continue
        except _Branch:
            pass
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        try:
            True if best is None or score > best[0] else (_ for _ in ()).throw(_Branch())
            best = (score, key)
        except _Branch:
            pass
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    try:
        True if not isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
        return None
    except _Branch:
        pass
    best = None
    _seq19 = list(node.values())
    _ix20 = 0
    while _ix20 < len(_seq19):
        value = _seq19[_ix20]
        _ix20 = _ix20 + 1
        try:
            True if isinstance(value, dict) and len(value) >= 2 else (_ for _ in ()).throw(_Branch())
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
        except _Branch:
            pass
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

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _bind_span(self.view.get('max_steps'))
        try:
            True if steps is None and isinstance(self.view.get('max_steps'), float) else (_ for _ in ()).throw(_Branch())
            steps = int(self.view['max_steps'])
        except _Branch:
            pass
        self.max_steps = steps if steps and steps > 0 else 30
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
        self.browse: dict[str, Any] | None = None
        self.browse_misses = 0
        self.model_busy: dict[str, float] = {}
        self.swept: set[str] = set()
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def walk_unit(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def wrap_core(self) -> str:
        return _fetch_zone(sorted(self.cart_lines.items()))

    def flip_grid43(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def drive_heap(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.fold_hook(t)), None)
        try:
            True if known else (_ for _ in ()).throw(_Branch())
            return known
        except _Branch:
            pass
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.fold_hook(tool) else None

    def shut_index33(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def spin_delta(self) -> bool:
        seen = self.shut_index33()
        return seen is False or (seen is None and (self.facts_from.get(self.drive_heap() or '') is True or bool(self.roles.get('offer'))))

    def push_cell(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.wrap_core() + f'#{self.version}') not in self.gate_ok

    def push_link(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        spare = 1 if self.buys == 0 else 0
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.push_cell('order') else 1)

    def cast_unit(self, limit: float=SPEND_SOFT) -> list[str]:
        try:
            True if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD else (_ for _ in ()).throw(_Branch())
            return [HEAVY, LIGHT]
        except _Branch:
            pass
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def flip_rail(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
        """One model reply. Providers limit requests per minute for each model, so a limited heavy model hands the call
        to the next of its kin at once, and rests a while; what is left of a limit does not count against the calls.
        Kin answer at low reasoning effort, which keeps a handed-over call as cheap and quick as a normal one, unless the
        call is `deep`."""
        try:
            True if self.time_left() < EXPLORE_RESERVE + 15 else (_ for _ in ()).throw(_Branch())
            models = list(dict.fromkeys((LIGHT if m == HEAVY else m for m in models)))
        except _Branch:
            pass
        for model in models:
            for choice in HEAVY_KIN if model == HEAVY else (model,):
                try:
                    True if self.calls >= ALL_CALLS or self.time_left() < CALL_RESERVE else (_ for _ in ()).throw(_Branch())
                    return None
                except _Branch:
                    pass
                try:
                    True if time.monotonic() < self.model_busy.get(choice, 0) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                self.calls += 1
                tries = 0
                while True:
                    _llm.timeout = max(8, min(50, int(self.time_left() - CALL_RESERVE)))
                    effort = 'low' if model == HEAVY and choice != HEAVY and (not deep) else None
                    message, status = self.spin_span(choice, messages, max_tokens, tools, effort)
                    if message is not None:
                        return message
                    if status == 429 and model == HEAVY:
                        self.model_busy[choice] = time.monotonic() + BUSY_SECONDS
                        self.calls -= 1
                        break
                    passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                    if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                        break
                    if self.time_left() < WAITS[tries] + CALL_RESERVE + 10:
                        break
                    pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                    self.waited += WAITS[tries]
                    tries += 1
                    time.sleep(pause)
                if status == 402:
                    self.no_credit = True
                    return None
                if status != 429:
                    break
        return None

    def spin_span(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        try:
            True if model.startswith('anthropic/') and len(messages) > 1 else (_ for _ in ()).throw(_Branch())
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                try:
                    True if isinstance(content, str) and content else (_ for _ in ()).throw(_Branch())
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
                except _Branch:
                    pass
        except _Branch:
            pass
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        if effort:
            body['reasoning'] = {'effort': effort}
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        try:
            True if isinstance(data, dict) and data.get('choices') else (_ for _ in ()).throw(_Branch())
            usage = data.get('usage') or {}
            try:
                True if isinstance(usage.get('cost'), (int, float)) else (_ for _ in ()).throw(_Branch())
                self.cost += float(usage['cost'])
            except _Branch:
                pass
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
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _bind_span(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, deep: bool=False) -> dict[str, Any] | None:
        for model in models:
            message = self.flip_rail([{'role': 'user', 'content': prompt}], [model], max_tokens, deep=deep)
            got = _check_node52((message or {}).get('content'))
            if got is not None:
                return got
            try:
                True if self.broke or self.no_credit or self.calls >= ALL_CALLS else (_ for _ in ()).throw(_Branch())
                return None
            except _Branch:
                pass
        return None

    def snap_link(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        _env.timeout = max(10, min(95, int(self.time_left() - 5)))
        _env.max_retries = 3 if self.time_left() > EXPLORE_RESERVE else 1
        _ri23 = 0
        while _ri23 < 4:
            _ = _ri23
            _ri23 = _ri23 + 1
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.flip_grid43() and (not (isinstance(seen, dict) and seen.get('error'))):
                    _seq21 = list(args.values())
                    _ix22 = 0
                    while _ix22 < len(_seq21):
                        value = _seq21[_ix22]
                        _ix22 = _ix22 + 1
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                try:
                    True if 'finish' in low or 'quarantin' in low or 'finaliz' in low else (_ for _ in ()).throw(_Branch())
                    self.finished = True
                    return None
                except _Branch:
                    pass
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
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def build_phase(self) -> None:
        lines = []
        _seq24 = list(self.tools)
        _ix25 = 0
        while _ix25 < len(_seq24):
            tool = _seq24[_ix25]
            _ix25 = _ix25 + 1
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_fetch_zone(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        try:
            True if self.view.get('opening') is not None else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _fetch_zone(_wrap_line(self.view.get('opening'), 1500))
        except _Branch:
            pass
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _snap_stash86(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        else:
            try:
                True if not self.no_credit and self.slot_mark(brief) else (_ for _ in ()).throw(_Branch())
                again = self.ask_json(prompt, [HEAVY], 2500) or {}
                try:
                    True if _snap_stash86(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy') else (_ for _ in ()).throw(_Branch())
                    brief = again
                except _Branch:
                    pass
            except _Branch:
                pass
        self.brief = brief
        self.push_zone(brief.get('browse'))
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
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _fetch_zone(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            try:
                True if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']) else (_ for _ in ()).throw(_Branch())
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
            except _Branch:
                pass
        except _Branch:
            pass
        _seq26 = list(_snap_stash86(brief.get('items')))
        _ix27 = 0
        while _ix27 < len(_seq26):
            entry = _seq26[_ix27]
            _ix27 = _ix27 + 1
            if not isinstance(entry, dict):
                continue
            quantity = _bind_span(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _snap_stash86(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and (_words(m.shelf) == _words(item.shelf)) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
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
        for pair in _snap_stash86(brief.get('fits')):
            try:
                True if isinstance(pair, dict) else (_ for _ in ()).throw(_Branch())
                part, main = (became.get(_bind_span(pair.get('item')) or 0), became.get(_bind_span(pair.get('with')) or 0))
                try:
                    True if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound) else (_ for _ in ()).throw(_Branch())
                    self.bound.append((part, main))
                except _Branch:
                    pass
            except _Branch:
                pass
        for lookup in _snap_stash86(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            try:
                True if text and (number, text) not in self.refs else (_ for _ in ()).throw(_Branch())
                self.refs.append((number, text))
            except _Branch:
                pass
        self.bind_vault()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _bind_span(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        else:
            try:
                True if self.roles.get('ask') else (_ for _ in ()).throw(_Branch())
                self.msg_limit = 3
            except _Branch:
                pass
        steps = _bind_span(brief.get('steps'))
        try:
            True if steps is not None and 10 <= steps < self.max_steps else (_ for _ in ()).throw(_Branch())
            self.max_steps = steps
        except _Branch:
            pass
        for entry in _snap_stash86(brief.get('process')):
            try:
                True if isinstance(entry, dict) and str(entry.get('step') or '').strip() else (_ for _ in ()).throw(_Branch())
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _bind_span(entry.get('messages')) or 0)})
            except _Branch:
                pass
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def push_zone(self, spec: Any) -> None:
        if not isinstance(spec, dict) or spec.get('tool') not in self.schemas:
            return
        props = self.schemas[spec['tool']].get('properties') or {}
        category, price = (spec.get('category'), spec.get('max_price'))
        if category not in props:
            return
        fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
        self.browse = {'tool': spec['tool'], 'category': category, 'price': price if price in props else None, 'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, price)}}

    @staticmethod
    def slot_mark(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _snap_stash86(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def bind_vault(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _snap_stash86(got.get('describe_other'))]]
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

    def wrap_link7(self, number: int, need: str) -> bool:
        try:
            True if need in [m for _, m in self.refs] or need in [str(x) for x in _snap_stash86(self.brief.get('lookups'))] else (_ for _ in ()).throw(_Branch())
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
            True if any(({w for w in _open_pool(other.name) if len(w) >= 4} & _open_pool(value) for other in self.items if other is not item)) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
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
        return read >= 3 and (not any((_open_pool(k) & _open_pool(prop) for k in keys)))

    def snap_stash(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.wrap_link7(number, m)]
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
            try:
                True if item.free else (_ for _ in ()).throw(_Branch())
                text += '; shopper does not mind: ' + ', '.join(item.free)
            except _Branch:
                pass
            parts.append(text)
        brief = self.brief
        try:
            True if self.budget is not None else (_ for _ in ()).throw(_Branch())
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        except _Branch:
            pass
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        try:
            True if brief.get('in_stock_only', True) is not False else (_ for _ in ()).throw(_Branch())
            parts.append('in stock only')
        except _Branch:
            pass
        for wish in _snap_stash86(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        _seq28 = list(_snap_stash86(brief.get('rules')))
        _ix29 = 0
        while _ix29 < len(_seq28):
            rule = _seq28[_ix29]
            _ix29 = _ix29 + 1
            parts.append(f'rule from the request: {rule}')
        try:
            True if brief.get('quiet_decline') is True else (_ for _ in ()).throw(_Branch())
            parts.append('if nothing fits: end without naming or suggesting any listing')
        except _Branch:
            pass
        _seq30 = list(self.process)
        _ix31 = 0
        while _ix31 < len(_seq30):
            entry = _seq30[_ix31]
            _ix31 = _ix31 + 1
            when = {'choose': 'before choosing', 'order': 'at the order (Kyo does it when you check out)', 'end': 'at an ending without a purchase (Kyo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.drive_heap()
        try:
            True if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')) else (_ for _ in ()).throw(_Branch())
            bare = sorted((t for t in self.flip_grid43() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.flip_grid43() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        except _Branch:
            pass
        return '\n'.join(('- ' + p for p in parts))

    def fetch_grid(self) -> str:
        left = self.left()
        text = f'[Kyo] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.push_link() + 1:
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
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        try:
            True if self.notes else (_ for _ in ()).throw(_Branch())
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        except _Branch:
            pass
        try:
            True if self.time_left() < EXPLORE_RESERVE + 45 else (_ for _ in ()).throw(_Branch())
            text += f'\n[time] about {max(0, int(self.time_left()))} s of this episode are left: settle on your picks and check out within the next few calls.'
        except _Branch:
            pass
        text += '\n[ledger]\n' + self.snap_stash()
        return text

    def wrap_pulse(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _wrap_delta24([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def fetch_wave(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            return True
        found = _wrap_delta24(rows)
        try:
            True if found else (_ for _ in ()).throw(_Branch())
            self.id_key = found
            return True
        except _Branch:
            pass
        return False

    def drive_cell(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            try:
                True if depth > 6 else (_ for _ in ()).throw(_Branch())
                return
            except _Branch:
                pass
            if isinstance(node, dict):
                ident = self.wrap_pulse(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
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
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
                _seq32 = list(node.values())
                _ix33 = 0
                while _ix33 < len(_seq32):
                    value = _seq32[_ix33]
                    _ix33 = _ix33 + 1
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.fetch_wave(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                try:
                    True if isinstance(option, str) else (_ for _ in ()).throw(_Branch())
                    found.add(option.strip().lower())
                except _Branch:
                    pass
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        _seq34 = list(self.schemas.values())
        _ix35 = 0
        while _ix35 < len(_seq34):
            schema = _seq34[_ix35]
            _ix35 = _ix35 + 1
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        _seq36 = list(self.rows.values())
        _ix37 = 0
        while _ix37 < len(_seq36):
            row = _seq36[_ix37]
            _ix37 = _ix37 + 1
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
            try:
                True if present[key] < 0.8 * total or len(values) > 40 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            try:
                True if best is None or score < best[0] else (_ for _ in ()).throw(_Branch())
                best = (score, key)
            except _Branch:
                pass
        return best[1] if best else None

    def fold_base52(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        _seq38 = list(self.seen_order)
        _ix39 = 0
        while _ix39 < len(_seq38):
            ident = _seq38[_ix39]
            _ix39 = _ix39 + 1
            record = self.full.get(ident)
            try:
                True if not record else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            block = _fact_block(record) or {}
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
                try:
                    True if text not in slot['values'] else (_ for _ in ()).throw(_Branch())
                    slot['values'].append(text)
                except _Branch:
                    pass
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _open_pool(item.shelf or '') | _open_pool(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _open_pool(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.push_mark(label)}
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
                try:
                    True if not values else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                mean = sum((len(v) for v in values)) / len(values)
                try:
                    True if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        try:
                            True if text not in twin['values'] else (_ for _ in ()).throw(_Branch())
                            twin['values'].append(text)
                        except _Branch:
                            pass
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def cast_edge70(self, label: str, name: str, values: list[str]) -> str:
        number = self.push_mark(label)
        try:
            True if not number else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _words(prop) != name and _open_pool(prop) & _open_pool(name) and words and (not words <= shown):
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
        try:
            True if not (a <= b or b <= a) else (_ for _ in ()).throw(_Branch())
            return len(both) >= 2 and same == len(both)
        except _Branch:
            pass
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        _seq40 = list(self.seen_order)
        _ix41 = 0
        while _ix41 < len(_seq40):
            ident = _seq40[_ix41]
            _ix41 = _ix41 + 1
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _open_pool(item.shelf or '') | _open_pool(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _open_pool(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.push_mark(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def fold_hook(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _push_grid(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        try:
            True if not tool or not target or any((k != target for k in schema.get('required') or [])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return target

    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        try:
            True if isinstance(value, dict) else (_ for _ in ()).throw(_Branch())
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        except _Branch:
            pass
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def tune_cell71(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.wrap_pulse(observation)
            try:
                True if ident else (_ for _ in ()).throw(_Branch())
                return [(ident, observation)]
            except _Branch:
                pass
            found: list[tuple[str, dict[str, Any]]] = []
            _seq42 = list(observation.values())
            _ix43 = 0
            while _ix43 < len(_seq42):
                value = _seq42[_ix43]
                _ix43 = _ix43 + 1
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.wrap_pulse(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.wrap_pulse(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.wrap_pulse(r, True)] if i]
        return []

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.tune_cell71(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def walk_token(self, name: str, observation: Any) -> dict[str, Any] | None:
        try:
            True if name in self.shapes else (_ for _ in ()).throw(_Branch())
            return self.shapes[name]
        except _Branch:
            pass
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        try:
            True if not lists else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _fetch_zone(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _fetch_zone(_wrap_line(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
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
        shape = self.walk_token(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        try:
            True if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        return (count, len(rows), shape['next_arg'], value)

    def check_vault(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.tune_cell71(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def snap_grid(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        try:
            True if value is None or later or count <= shown else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if pick == 'cheapest' and self.check_vault(observation) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.push_link() - 2 >= 12):
            pages = 2
        try:
            True if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.push_link() - pages < 12 else (_ for _ in ()).throw(_Branch())
            return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_fetch_zone(value)}."
        except _Branch:
            pass
        texts = []
        _ri44 = 0
        while _ri44 < pages:
            _ = _ri44
            _ri44 = _ri44 + 1
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.snap_link(name, call)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.check_span(seen)
            self.read_cache[name + _fetch_zone(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_fetch_zone(value)}: {shown_text}')
            after = self.paging(name, seen)
            value = after[3] if after else None
            try:
                True if value is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Kyo] {count} listings matched and Kyo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def drive_span(self, name: str, args: dict[str, Any], observation: Any) -> str:
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
        over = [i for i, _ in self.tune_cell71(observation) if (self.price_of(i) or 0) > cap]
        try:
            True if not over else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_fetch_zone(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Kyo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        try:
            True if self.brief.get('pick') != 'cheapest' or not self.price_key else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        found = {i for seen in observations for i, _ in self.tune_cell71(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        try:
            True if self.wider_held or self.brief.get('pick') != 'cheapest' else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.wider_held = True
        try:
            True if self.drive_sigma53() or self.left() < 14 + 2 * len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        lines = [f'{i} at {self.price_of(i):g}' for i in self.fold_ring35() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def other_values(self, number: int) -> list[str]:
        """Values listings of this item show for properties the request leaves open, taken in turn from its most carried
        keys: searched together with the required values, they bring up rows the plain search does not reach."""
        table = self.fold_base52()
        keys = self.lane_keys(number, table)[:3]
        said = ' '.join(self.value_words(number)).lower() + ' ' + self.items[number - 1].name.lower()
        lanes = []
        _seq45 = list(keys)
        _ix46 = 0
        while _ix46 < len(_seq45):
            key = _seq45[_ix46]
            _ix46 = _ix46 + 1
            values = next((v for label, name, _, v in table if name == key and (len(self.items) == 1 or self.push_mark(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and (v not in said) and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def drive_sigma53(self) -> bool:
        """For a cheapest pick: bring up rows priced below each cart line, for the reading and the check before checkout.
        Lists the line's category under its price where the store can, else searches the required values with other values
        listings show. Steps for reading, the cart changes and the checkout stay free. Returns whether it could look."""
        cat_key = self.category_key()
        plan = self.search_template()
        mark = self.wrap_core() + f'#{self.version}'
        room = self.left() - self.push_link() - len(self.items) - 3
        try:
            True if mark in self.swept or room < 1 or self.time_left() < EXPLORE_RESERVE + 25 or (not cat_key) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        self.swept.add(mark)
        limit = self.step + min(SWEEP_STEPS, room)
        ran = False
        cheaper: list[str] = []
        for line in self.fold_ring35():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            number = next((n for n, item in enumerate(self.items, 1) if label and self.owns(item, label)), 0)
            try:
                True if price is None or not number or self.step >= limit else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            found = self.list_category(number, int(price) - 1 if price > 1 else price * 0.9, 1 if limit - self.step >= 2 else 0)
            try:
                True if found is not None else (_ for _ in ()).throw(_Branch())
                ran = True
            except _Branch:
                try:
                    True if plan else (_ for _ in ()).throw(_Branch())
                    base = self.value_words(number)[:2]
                    for other in self.other_values(number)[:3]:
                        if self.step >= limit:
                            break
                        found = (found or []) + self.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, self.items[number - 1].name]))
                        ran = True
                except _Branch:
                    pass
            cheaper += [i for i in self.slot_delta(number, found or [], price) if i != line and i not in self.full]
        self.trace.append({'kind': 'sweep', 'ran': ran, 'cheaper': cheaper[:6]})
        return ran

    def fold_token(self, ids: list[str], keep: int) -> list[str]:
        tool = self.drive_heap()
        target = self.fold_hook(tool)
        try:
            True if not tool or not target else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            try:
                True if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or (self.time_left() < CALL_RESERVE + 10) else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.snap_link(tool, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.wrap_pulse(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.check_span(seen)[:1500])
        try:
            True if texts else (_ for _ in ()).throw(_Branch())
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        except _Branch:
            pass
        return texts

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.wrap_core() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.fold_ring35()
        picks: list[str] = []
        _seq47 = list(cart)
        _ix48 = 0
        while _ix48 < len(_seq47):
            line = _seq47[_ix48]
            _ix48 = _ix48 + 1
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            try:
                True if price is None or not kind else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:CHEAPER_OPENS if len(cart) == 1 else 2] if i not in picks]
        return self.fold_token(picks[:CHEAPER_OPENS], self.push_link() + 2)

    def observe(self, name: str, observation: Any) -> str:
        self.drive_cell(observation)
        lines = self.step_node62(observation)
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        meaning = self.push_flag(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _push_grid(v)), None)
        try:
            True if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.spin_delta() else (_ for _ in ()).throw(_Branch())
            self.bind_unit89()
            return
        except _Branch:
            pass
        if self.left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
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
        reply = self.snap_link(tool, args)
        try:
            True if reply is not None else (_ for _ in ()).throw(_Branch())
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.tune_cell71(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.bind_unit89()
                return
            if len(self.full) == before and len(picks) > 2 and (self.left() > 12):
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                try:
                    True if not faults else (_ for _ in ()).throw(_Branch())
                    reply = self.snap_link(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))
                except _Branch:
                    pass
        except _Branch:
            pass

    def bind_unit89(self) -> None:
        tool = self.drive_heap()
        target = self.fold_hook(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        _seq49 = list(self.item_groups())
        _ix50 = 0
        while _ix50 < len(_seq49):
            ids = _seq49[_ix50]
            _ix50 = _ix50 + 1
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _fetch_zone(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fold_token(first[:cap][:8], 22)
        self.fold_token((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def flip_phase(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.walk_unit() <= 0:
            return False
        if self.left() <= 8 or self.time_left() < EXPLORE_RESERVE + 40:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _push_grid(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        _seq51 = list(schema.get('required') or [])
        _ix52 = 0
        while _ix52 < len(_seq51):
            key = _seq51[_ix52]
            _ix52 = _ix52 + 1
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

    def snap_hook23(self) -> str:
        self.asked = True
        try:
            True if self.walk_unit() <= 0 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        try:
            True if self.time_left() > EXPLORE_RESERVE + 30 else (_ for _ in ()).throw(_Branch())
            self.widen()
        except _Branch:
            pass
        lanes = self.plan_lanes(self.walk_unit())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str]]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.walk_unit() > 0 and (not self.finished) and (self.left() > 6) and (self.time_left() > EXPLORE_RESERVE + 20):
            try:
                True if follow else (_ for _ in ()).throw(_Branch())
                number, labels = follow.pop(0)
            except _Branch:
                number, labels = self.build_index(lanes, found, spent, used)
                try:
                    True if not labels else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
            used[number].update((_words(label) for label in labels))
            batch = self.step_rail([(number, labels[0], self.spin_edge30(number, labels[0]))])
            try:
                True if not batch else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            try:
                True if self.cast_unit()[:1] == [HEAVY] and any((self.open_phase(entry) for entry in _snap_stash86(heard.get('needs')))) else (_ for _ in ()).throw(_Branch())
                heard = self.hear(batch, (HEAVY,)) or heard
            except _Branch:
                pass
            self.draw_phase(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_bind_span(x) for x in _snap_stash86(heard.get('crowded'))]:
                continue
            head = labels[0].split()[-1]
            try:
                True if len(labels[0].split()) >= 2 and _words(head) not in used[number] else (_ for _ in ()).throw(_Branch())
                follow.insert(0, (number, [head]))
            except _Branch:
                pass
        found_items = [n for n in found if found[n] > 0]
        searched = self.open_index(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        try:
            True if pointed else (_ for _ in ()).throw(_Branch())
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        except _Branch:
            pass
        try:
            True if searched else (_ for _ in ()).throw(_Branch())
            summary += '\n' + searched
        except _Branch:
            pass
        try:
            True if self.queued else (_ for _ in ()).throw(_Branch())
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        except _Branch:
            pass
        return summary

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _push_grid(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            try:
                True if not named else (_ for _ in ()).throw(_Branch())
                labels = {str(r.get(self.category_key()) or '').strip().lower() for r in self.rows.values()} - {''} if self.category_key() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            except _Branch:
                pass
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.category_key()
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _open_pool(v) if len(w) >= 4} & {w for w in _open_pool(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                _seq53 = list(list(keep))
                _ix54 = 0
                while _ix54 < len(_seq53):
                    k = _seq53[_ix54]
                    _ix54 = _ix54 + 1
                    try:
                        True if isinstance((props.get(k) or {}).get('enum'), list) else (_ for _ in ()).throw(_Branch())
                        used = [_fetch_zone(a[k]) for a in worded if k in a]
                        try:
                            True if used else (_ for _ in ()).throw(_Branch())
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _fetch_zone(a[k]) == best))
                        except _Branch:
                            pass
                    except _Branch:
                        pass
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            try:
                True if n != number or need in [m for _, m in self.refs] or ':' not in need else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def open_index(self, numbers: list[int]) -> str:
        plan = self.search_template()
        try:
            True if not plan or self.left() < 22 or self.time_left() < EXPLORE_RESERVE + 25 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                try:
                    True if not words else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.snap_link(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                cat_key = self.category_key()
                found = [i for i, _ in self.tune_cell71(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                try:
                    True if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)) else (_ for _ in ()).throw(_Branch())
                    cap = None
                except _Branch:
                    pass
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                seen.append(f'{_fetch_zone(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                try:
                    True if rows else (_ for _ in ()).throw(_Branch())
                    opened = self.fold_token([i for i in rows if i not in self.full][:min(3, budget)], self.push_link() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
                except _Branch:
                    pass
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def build_index(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            try:
                True if left else (_ for _ in ()).throw(_Branch())
                return (number, left[:1])
            except _Branch:
                pass
        return (0, [])

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_open_pool(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.push_mark(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _open_pool(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def push_mark(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ''):
                score = 3.0
            else:
                try:
                    True if words == _words(item.name) else (_ for _ in ()).throw(_Branch())
                    score = 2.0
                except _Branch:
                    mine = _roots(item.shelf or '') | _roots(item.name)
                    score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.push_mark(label) == self.items.index(item) + 1)

    def flip_token(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.fold_base52()
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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.cast_edge70(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _snap_stash86(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        _seq55 = list(_snap_stash86(plan.get('ask')))
        _ix56 = 0
        while _ix56 < len(_seq55):
            pick = _seq55[_ix56]
            _ix56 = _ix56 + 1
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _bind_span(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            try:
                True if not label or len(label.split()) > 5 else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        _seq59 = list(lanes)
        _ix60 = 0
        while _ix60 < len(_seq59):
            number = _seq59[_ix60]
            _ix60 = _ix60 + 1
            keys = self.lane_keys(number, table)
            read = max(1, self.flip_token(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                try:
                    True if self.push_mark(label) == number else (_ for _ in ()).throw(_Branch())
                    own[key] = max(own.get(key, 0.0), carried / read)
                except _Branch:
                    pass

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_open_pool(label), _open_pool(k)) or _names(_open_pool(k), _open_pool(label))]
                return max(near) if near else 0.0
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            _seq57 = list(lanes[number])
            _ix58 = 0
            while _ix58 < len(_seq57):
                pick = _seq57[_ix58]
                _ix58 = _ix58 + 1
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
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
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
            try:
                True if count <= TIGHT_MESSAGES and read >= 3 else (_ for _ in ()).throw(_Branch())
                sure = [x for x in lanes[number] if _words(x) in hinted[number] or share_of(x) >= 0.5]
                lanes[number] = sure + [x for x in lanes[number] if x not in sure]
            except _Branch:
                pass
        self.trace.append({'kind': 'plan', 'ask': _wrap_line(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def open_phase(self, entry: Any) -> bool:
        """Whether a stated answer deserves a second reading: it points at another listing, or its value has words that
        no opened listing shows for that property, which is how words around a value creep into it."""
        try:
            True if not isinstance(entry, dict) else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        prop, _, value = str(entry.get('need') or '').partition(':')
        words = set(re.findall('[a-z0-9]+', value.lower()))
        try:
            True if entry.get('points') is True else (_ for _ in ()).throw(_Branch())
            return True
        except _Branch:
            pass
        try:
            True if len(words) < 2 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        seen: set[str] = set()
        for record in self.full.values():
            for key, shown in (_fact_block(record) or {}).items():
                try:
                    True if _names(_open_pool(prop), _open_pool(key)) or _names(_open_pool(key), _open_pool(prop)) else (_ for _ in ()).throw(_Branch())
                    seen |= set(re.findall('[a-z0-9]+', str(shown).lower()))
                except _Branch:
                    pass
        return bool(words - seen)

    def spin_edge30(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.items[number - 1].name}?'

    def step_rail(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.snap_link(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.draw_unit5(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.wrap_core()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def draw_phase(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        _seq61 = list(_snap_stash86(heard.get('needs')))
        _ix62 = 0
        while _ix62 < len(_seq61):
            entry = _seq61[_ix62]
            _ix62 = _ix62 + 1
            try:
                True if not isinstance(entry, dict) or not str(entry.get('need') or '').strip() else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
            number = min(max(1, _bind_span(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            try:
                True if entry.get('points') is True and (number, need) not in self.refs else (_ for _ in ()).throw(_Branch())
                self.refs.append((number, need))
            except _Branch:
                pass
        _seq63 = list(_snap_stash86(heard.get('drop')))
        _ix64 = 0
        while _ix64 < len(_seq63):
            entry = _seq63[_ix64]
            _ix64 = _ix64 + 1
            try:
                True if isinstance(entry, dict) and str(entry.get('property') or '').strip() else (_ for _ in ()).throw(_Branch())
                item = self.items[min(max(1, _bind_span(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
            except _Branch:
                pass
        _seq65 = list(_snap_stash86(heard.get('free')))
        _ix66 = 0
        while _ix66 < len(_seq65):
            entry = _seq65[_ix66]
            _ix66 = _ix66 + 1
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _bind_span(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        try:
            True if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0) else (_ for _ in ()).throw(_Branch())
            self.budget = heard['budget']
        except _Branch:
            pass

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        try:
            True if not steps or self.finished or (not self.push_cell(stage)) else (_ for _ in ()).throw(_Branch())
            return (True, '')
        except _Branch:
            pass
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_fetch_zone(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.snap_stash() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            try:
                True if said else (_ for _ in ()).throw(_Branch())
                self.draw_phase({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            except _Branch:
                pass
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _wrap_line(call, 200), 'steps': _wrap_line(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            try:
                True if told or hold else (_ for _ in ()).throw(_Branch())
                self.version += 1
                return (False, ('Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            except _Branch:
                pass
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _snap_stash86(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_bind_span(n) or 0) in proof for n in _snap_stash86(e.get('lines'))))]
            try:
                True if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2) else (_ for _ in ()).throw(_Branch())
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _wrap_line(unproven, 120)})
                self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            except _Branch:
                pass
            try:
                True if not call or name not in self.schemas or name in ending else (_ for _ in ()).throw(_Branch())
                if not call:
                    self.gate_ok.add((stage, self.wrap_core() + f'#{self.version}'))
                break
            except _Branch:
                pass
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.step_rail([(1, '', text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.snap_link(name, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            seen = reply.get('observation')
            self.drive_cell(seen)
            self.done_log.append(f'step {self.step}: {name} {_fetch_zone(args)} -> {self.check_span(seen)[:300]} (cart then: {self.wrap_core()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_fetch_zone(args)} -> {self.check_span(seen)[:400]}')
        return (True, 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def draw_unit5(self, observation: Any) -> str:
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
            try:
                True if isinstance(node, str) and len(node.split()) >= 2 else (_ for _ in ()).throw(_Branch())
                found.append(node)
            except _Branch:
                try:
                    True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
                    for value in node.values():
                        walk(value, depth + 1)
                except _Branch:
                    try:
                        True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                        for value in node:
                            walk(value, depth + 1)
                    except _Branch:
                        pass
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _fetch_zone(observation)[:400]

    def step_node62(self, observation: Any) -> list[str]:
        lines: list[str] = []
        try:
            True if isinstance(observation, dict) and (not self.wrap_pulse(observation)) else (_ for _ in ()).throw(_Branch())
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.wrap_pulse(value, False, 1)):
                    for inner in value.values():
                        try:
                            True if isinstance(inner, str) and len(inner.split()) >= 6 else (_ for _ in ()).throw(_Branch())
                            lines.append(inner)
                        except _Branch:
                            pass
        except _Branch:
            pass
        return [line for line in lines if line not in self.heard]

    def push_flag(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.snap_stash() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _snap_stash86(got.get('gone')) if isinstance(g, (str, int))]
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
        drops = [e for e in _snap_stash86(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _snap_stash86(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        _seq67 = list(drops)
        _ix68 = 0
        while _ix68 < len(_seq67):
            entry = _seq67[_ix68]
            _ix68 = _ix68 + 1
            item = self.items[min(max(1, _bind_span(entry.get('item')) or 1), len(self.items)) - 1]
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
        for entry in adds:
            item = self.items[min(max(1, _bind_span(entry.get('item')) or 1), len(self.items)) - 1]
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
        try:
            True if changed or summary else (_ for _ in ()).throw(_Branch())
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        except _Branch:
            pass
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
            try:
                True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
                ident = self.wrap_pulse(node, False, depth)
                try:
                    True if ident and _fact_block(node) and (ident not in ids) else (_ for _ in ()).throw(_Branch())
                    ids.append(ident)
                    return
                except _Branch:
                    pass
                for value in node.values():
                    walk(value, depth + 1)
            except _Branch:
                try:
                    True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                    _seq69 = list(node)
                    _ix70 = 0
                    while _ix70 < len(_seq69):
                        value = _seq69[_ix70]
                        _ix70 = _ix70 + 1
                        walk(value, depth + 1)
                except _Branch:
                    pass
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
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_open_pool(prop), _open_pool(k))]
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
            return _fetch_zone(_wrap_line(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _fetch_zone(self.cart_lines)

    def draw_unit(self, ident: str) -> int:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                try:
                    True if isinstance(node, dict) else (_ for _ in ()).throw(_Branch())
                    try:
                        True if ident in [v.strip() for v in node.values() if isinstance(v, str)] else (_ for _ in ()).throw(_Branch())
                        counts = [_bind_span(v) for k, v in node.items() if not isinstance(v, str) and _bind_span(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    except _Branch:
                        pass
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                except _Branch:
                    try:
                        True if isinstance(node, list) else (_ for _ in ()).throw(_Branch())
                        stack += node
                    except _Branch:
                        pass
        except _Branch:
            pass
        return self.cart_lines.get(ident, 1)

    def fold_ring35(self) -> list[str]:
        try:
            True if self.cart_view is not None and self.cart_seen_at >= self.cart_at else (_ for _ in ()).throw(_Branch())
            text = _fetch_zone(self.cart_view)
            return [i for i in self.rows if i in text]
        except _Branch:
            pass
        return list(self.cart_lines)

    def spin_link37(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _fetch_zone(_wrap_line(record, 400))[:3000] if record else '(no record seen)'

    def flip_phase19(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        _seq71 = list(order)
        _ix72 = 0
        while _ix72 < len(_seq71):
            ident = _seq71[_ix72]
            _ix72 = _ix72 + 1
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_fetch_zone(_wrap_line(record, 160))[:900])
            try:
                True if len(lines) >= limit else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        return '\n'.join(lines)

    def shut_stash75(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        _seq73 = list(self.seen_order)
        _ix74 = 0
        while _ix74 < len(_seq73):
            ident = _seq73[_ix74]
            _ix74 = _ix74 + 1
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _lift_heap(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def push_vault56(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        try:
            True if not self.bound else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        index = self.shut_stash75()
        cart = self.fold_ring35()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.push_vault56(main)]
            parts = [i for i in cart if i in self.push_vault56(part)]
            if len(devices) != 1 or not parts:
                continue
            device = index['marks'].get(devices[0]) or {}
            _seq75 = list(parts)
            _ix76 = 0
            while _ix76 < len(_seq75):
                line = _seq75[_ix76]
                _ix76 = _ix76 + 1
                if self.fits_device(line, device, index) is False:
                    good = [i for i in self.push_vault56(part) if self.fits_device(i, device, index) and (not self.sold_out(i)) and (i not in cart)][:4]
                    named = sorted(self.fit_set(line, index))[:4] or [(index['marks'].get(line) or {}).get('own') or 'other models']
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(named)}, not with {devices[0]} ({', '.join(sorted(device.get('codes') or device.get('series') or []))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else ''))
        return faults

    def fits_device(self, ident: str, device: dict[str, Any], index: dict[str, Any]) -> bool | None:
        return _fit_verdict(index['marks'].get(ident) or {}, device, self.fit_set(ident, index))

    def fit_plan(self, short: bool=False) -> str:
        try:
            True if not self.bound else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        index = self.shut_stash75()
        lines = []
        _seq83 = list(sorted({m for _, m in self.bound}))
        _ix84 = 0
        while _ix84 < len(_seq83):
            main = _seq83[_ix84]
            _ix84 = _ix84 + 1
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            _seq81 = list(parts)
            _ix82 = 0
            while _ix82 < len(_seq81):
                p = _seq81[_ix82]
                _ix82 = _ix82 + 1
                _seq79 = list(self.push_vault56(p))
                _ix80 = 0
                while _ix80 < len(_seq79):
                    i = _seq79[_ix80]
                    _ix80 = _ix80 + 1
                    try:
                        True if self.sold_out(i) else (_ for _ in ()).throw(_Branch())
                        continue
                    except _Branch:
                        pass
                    own = (index['marks'].get(i) or {}).get('own') or ''
                    named = self.fit_set(i, index) | ({own.split('-')[0]} if re.match(PART_SERIES + '-', own) else set())
                    _seq77 = list(named)
                    _ix78 = 0
                    while _ix78 < len(_seq77):
                        code = _seq77[_ix78]
                        _ix78 = _ix78 + 1
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {}
            for code, _ in ranked:
                marks = {i: index['marks'].get(i) or {} for i in self.push_vault56(main)}
                devices[code] = [i for i, m in marks.items() if any((_code_names(code, c) for c in m.get('codes') or [])) or code in (m.get('series') or [])][:2]
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            try:
                True if text else (_ for _ in ()).throw(_Branch())
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
            except _Branch:
                pass
        try:
            True if not lines else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code, its series or another name it is sold as, or when the device's text names the part's series; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def tune_line(self) -> list[str]:
        cat_key = self.category_key()
        ids = [i for i in self.fold_ring35() if (self.draw_unit(i) or 0) > 0]
        try:
            True if not ids or not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
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
                have = self.draw_unit(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(self.items):
            _seq85 = list(ids)
            _ix86 = 0
            while _ix86 < len(_seq85):
                i = _seq85[_ix86]
                _ix86 = _ix86 + 1
                try:
                    True if not owner[i] else (_ for _ in ()).throw(_Branch())
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
                except _Branch:
                    pass
        _seq87 = list(ids)
        _ix88 = 0
        while _ix88 < len(_seq87):
            i = _seq87[_ix88]
            _ix88 = _ix88 + 1
            if self.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.price_of(i) for i in ids}
        try:
            True if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())) else (_ for _ in ()).throw(_Branch())
            try:
                True if self.brief.get('budget_scope') == 'each' else (_ for _ in ()).throw(_Branch())
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            except _Branch:
                total = sum((prices[i] * self.draw_unit(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        except _Branch:
            pass
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        try:
            True if not cat_key else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        out = []
        for ident in self.fold_ring35():
            have = self.draw_unit(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.fold_ring35() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
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
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _push_grid(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _push_grid(v)), None)
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
        try:
            True if len(calls) + 3 > self.left() else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        done = []
        _seq89 = list(calls)
        _ix90 = 0
        while _ix90 < len(_seq89):
            args = _seq89[_ix90]
            _ix90 = _ix90 + 1
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            reply = self.snap_link(name, args)
            if reply is None or self.finished:
                break
            self.walk_link19(name, args)
            self.cart_said = _fetch_zone(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(_fetch_zone(args) + ' -> ' + self.check_span(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def cast_sigma75(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.fold_ring35()
        key = self.cast_sigma75(ids)
        try:
            True if key in self.checked or not self.cast_unit(limit) or (not ids) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.snap_stash()
        try:
            True if lead else (_ for _ in ()).throw(_Branch())
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        except _Branch:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.spin_link37(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.flip_phase19(ids, 15)
        got = self.ask_json(prompt, self.cast_unit(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        try:
            True if not got or got.get('ok') is not False else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _snap_stash86(got.get('problems'))][:5]
        better = [str(b) for b in _snap_stash86(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        try:
            True if better else (_ for _ in ()).throw(_Branch())
            text += '. Listings that look better: ' + ', '.join(better)
        except _Branch:
            pass
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.fold_hook(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            _seq91 = list(reversed(self.seen_order))
            _ix92 = 0
            while _ix92 < len(_seq91):
                ident = _seq91[_ix92]
                _ix92 = _ix92 + 1
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                try:
                    True if record and len(records) < 45 else (_ for _ in ()).throw(_Branch())
                    records.append('OPENED ' + _fetch_zone(_wrap_line(record, 300))[:1400])
                except _Branch:
                    pass
            _seq93 = list(reversed(self.seen_order))
            _ix94 = 0
            while _ix94 < len(_seq93):
                ident = _seq93[_ix94]
                _ix94 = _ix94 + 1
                try:
                    True if len(records) >= 60 else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _fetch_zone(_wrap_line(self.rows.get(ident), 80))[:300])
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _wrap_line(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _snap_stash86(got.get('chains')):
                try:
                    True if not isinstance(chain, dict) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _snap_stash86(chain.get('links')) if isinstance(l, dict)]
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
            _seq95 = list(reach[:max(0, min(3, 6 - len(seen)))])
            _ix96 = 0
            while _ix96 < len(_seq95):
                ident = _seq95[_ix96]
                _ix96 = _ix96 + 1
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                try:
                    True if faults else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                reply = self.snap_link(tool, args)
                try:
                    True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.check_span(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
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

    def open_pool(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        _seq97 = list(ids)
        _ix98 = 0
        while _ix98 < len(_seq97):
            ident = _seq97[_ix98]
            _ix98 = _ix98 + 1
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            try:
                True if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)) else (_ for _ in ()).throw(_Branch())
                out.append(ident)
            except _Branch:
                pass
        return out

    def opened_shop(self, about: str) -> list[str]:
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

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.open_pool(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        _seq99 = list(self.opened_shop(about))
        _ix100 = 0
        while _ix100 < len(_seq99):
            ident = _seq99[_ix100]
            _ix100 = _ix100 + 1
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
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.open_pool(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                self.link_searches += 1
                reply = self.snap_link(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.tune_cell71(observation)]
                    known = self.open_pool(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
            except _Branch:
                pass
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _fetch_zone(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.open_pool(about, list(reversed(self.seen_order)))))[:2]

    def drive_delta(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.drive_delta()
        target = self.fold_hook(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.fold_ring35() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        try:
            True if not stale or self.left() < len(stale) + 3 else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            before = dict(self.rows.get(ident) or {})
            reply = self.snap_link(tool, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_fetch_zone(_wrap_line(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.fold_ring35() if i not in self.gone and self.sold_out(i)]
        try:
            True if not lost else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        try:
            True if picks and self.left() <= 6 else (_ for _ in ()).throw(_Branch())
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        except _Branch:
            pass
        try:
            True if picks and self.brief.get('pick') == 'cheapest' else (_ for _ in ()).throw(_Branch())
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        except _Branch:
            pass
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def step_hook65(self) -> str:
        ids = [i for i in self.fold_ring35() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.cast_unit():
            return []
        records = []
        _seq101 = list(reversed(self.seen_order))
        _ix102 = 0
        while _ix102 < len(_seq101):
            ident = _seq101[_ix102]
            _ix102 = _ix102 + 1
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_fetch_zone(_wrap_line(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            try:
                True if len(records) >= 25 else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.snap_stash()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.spin_link37(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.cast_unit(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _snap_stash86((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            try:
                True if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.fold_ring35()) and (ident not in [i for _, i in fits]) else (_ for _ in ()).throw(_Branch())
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
            except _Branch:
                pass
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def check_base(self) -> str:
        try:
            True if self.audited or not self.cast_unit() else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.audited = True
        got = self.snap_rail58()
        self.take_review(got)
        named = self.audit_fits(got)
        missing = [n for n in range(1, len(self.items) + 1) if not named.get(str(n))]
        try:
            True if missing else (_ for _ in ()).throw(_Branch())
            rows = [str(i) for i in _snap_stash86((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            opened = bool(rows) and self.open_rows(rows)
            try:
                True if self.recovery_search(got, missing) or opened else (_ for _ in ()).throw(_Branch())
                got = self.snap_rail58()
                self.take_review(got)
                named = self.audit_fits(got) or named
            except _Branch:
                pass
        except _Branch:
            pass
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def snap_rail58(self) -> dict[str, Any] | None:
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        cat_key = self.category_key()
        opened: list[str] = []
        rows: list[str] = []
        _seq105 = list((True, False))
        _ix106 = 0
        while _ix106 < len(_seq105):
            strict = _seq105[_ix106]
            _ix106 = _ix106 + 1
            _seq103 = list(reversed(self.seen_order))
            _ix104 = 0
            while _ix104 < len(_seq103):
                ident = _seq103[_ix104]
                _ix104 = _ix104 + 1
                label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
                try:
                    True if self.sold_out(ident) or (strict and (cap is not None and (self.price_of(ident) or 0) > cap or (label and (not any((self.owns(item, label) for item in self.items)))))) else (_ for _ in ()).throw(_Branch())
                    continue
                except _Branch:
                    pass
                record = self.full.get(ident) if ident in self.full else None
                try:
                    True if record and len(opened) < 20 else (_ for _ in ()).throw(_Branch())
                    opened.append('OPENED ' + _fetch_zone(_wrap_line(record, 200))[:1000])
                except _Branch:
                    try:
                        True if not record and len(rows) < 15 else (_ for _ in ()).throw(_Branch())
                        rows.append('ROW ' + _fetch_zone(_wrap_line(self.rows.get(ident), 120))[:500])
                    except _Branch:
                        pass
            if opened or rows:
                break
        try:
            True if not opened and (not rows) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.snap_stash()
        answers = [line for line in self.done_log if 'message to the shopper' in line]
        try:
            True if answers else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers)
        except _Branch:
            pass
        tried = [str(v) for args in self.searched for v in args.values() if isinstance(v, str)]
        try:
            True if tried else (_ for _ in ()).throw(_Branch())
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(tried))
        except _Branch:
            pass
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        got = self.ask_json(prompt, self.cast_unit(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        """Apply what the decline review found wrong with the ledger: story set aside, values corrected. A requirement
        the shopper stated in an answer can be corrected but not set aside, and a lookup is left alone."""
        lookups = {need for _, need in self.refs}
        changed = []
        aside: set[int] = set()
        for entry in _snap_stash86((got or {}).get('not_firm')):
            if isinstance(entry, dict):
                number, need = (_bind_span(entry.get('item')) or 0, str(entry.get('need') or '').strip())
                if 1 <= number <= len(self.items) and number not in aside and (need in self.items[number - 1].musts) and (need not in self.items[number - 1].dropped) and (need not in lookups):
                    aside.add(number)
                    self.items[number - 1].dropped.append(need)
                    changed.append(f'set aside {need}')
        _seq107 = list(_snap_stash86((got or {}).get('corrected')))
        _ix108 = 0
        while _ix108 < len(_seq107):
            entry = _seq107[_ix108]
            _ix108 = _ix108 + 1
            try:
                True if isinstance(entry, dict) else (_ for _ in ()).throw(_Branch())
                number, need, value = (_bind_span(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip())
                try:
                    True if 1 <= number <= len(self.items) and need in self.items[number - 1].musts + self.items[number - 1].told and (need not in lookups) and (':' in value) else (_ for _ in ()).throw(_Branch())
                    self.items[number - 1].dropped.append(need)
                    self.items[number - 1].told.append(value)
                    changed.append(f'{need} read as {value}')
                except _Branch:
                    pass
            except _Branch:
                pass
        if changed:
            self.version += 1
            self.trace.append({'kind': 'review', 'changed': changed})
            self.notes.append('a review of the request: ' + '; '.join(changed))

    def fetch_hook(self, number: int) -> str:
        """The category label of a requested item: the one most of its rows show, else the brief's."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()] if cat_key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def list_category(self, number: int, cap: float | None, more: int=0) -> list[str] | None:
        """The rows the store's category listing shows for an item, under a price limit when one is given, with up to
        `more` further pages. None when the store has no such listing or it does not answer."""
        spec = self.browse
        label = self.fetch_hook(number)
        try:
            True if not spec or not label or self.browse_misses >= 2 or (cap is not None and (not spec['price'])) else (_ for _ in ()).throw(_Branch())
            return None
        except _Branch:
            pass
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        faults: list[str] = []
        args = _shape(args, self.schemas.get(spec['tool']), 'args', faults)
        try:
            True if faults or self.finished else (_ for _ in ()).throw(_Branch())
            self.browse_misses += 1
            return None
        except _Branch:
            pass
        reply = self.snap_link(spec['tool'], args)
        if reply is None:
            self.browse_misses += 1
            return None
        seen = reply.get('observation')
        self.observe(spec['tool'], seen)
        found = [i for i, _ in self.tune_cell71(seen)]
        try:
            True if isinstance(seen, dict) and seen.get('error') else (_ for _ in ()).throw(_Branch())
            self.browse_misses += 1
            return None
        except _Branch:
            pass
        for _ in range(more):
            page = self.paging(spec['tool'], seen)
            if not page or page[3] is None or self.finished:
                break
            faults = []
            call = _shape({**args, page[2]: page[3]}, self.schemas.get(spec['tool']), 'args', faults)
            reply = self.snap_link(spec['tool'], call) if not faults else None
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            seen = reply.get('observation')
            self.observe(spec['tool'], seen)
            found += [i for i, _ in self.tune_cell71(seen)]
        return found

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        faults: list[str] = []
        args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
        try:
            True if faults or self.finished else (_ for _ in ()).throw(_Branch())
            return []
        except _Branch:
            pass
        reply = self.snap_link(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get('observation'))
        self.searched = (self.searched + [dict(args)])[-8:]
        return [i for i, _ in self.tune_cell71(reply.get('observation'))]

    def slot_delta(self, number: int, found: list[str], below: float | None=None) -> list[str]:
        """Buyable rows of the item's kind, unread first, most of the requirement's words first, then cheapest."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower())}
        rows = [i for i in dict.fromkeys(found) if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        try:
            True if below is not None else (_ for _ in ()).throw(_Branch())
            rows = [i for i in rows if (self.price_of(i) if self.price_of(i) is not None else below) < below]
        except _Branch:
            pass
        return sorted(rows, key=lambda i: (i in self.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _fetch_zone(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        """Before a decline stands: list each unfitted item's category and run the searches the review proposes, then
        read the most promising new rows, so that the second look at the evidence has something new to judge. Half of
        the steps it may use are kept for reading; the cart and the checkout stay possible."""
        keep = self.push_link() + len(self.items) + 2
        room = min(RECOVERY_STEPS, self.left() - keep)
        try:
            True if room < 3 or self.time_left() < EXPLORE_RESERVE + 25 else (_ for _ in ()).throw(_Branch())
            return False
        except _Branch:
            pass
        before = len(self.full)
        last = self.step + room
        searching = self.step + max(1, room // 2)
        plan = self.search_template()
        queries: list[tuple[int, str]] = []
        _seq109 = list(_snap_stash86((got or {}).get('search')))
        _ix110 = 0
        while _ix110 < len(_seq109):
            entry = _seq109[_ix110]
            _ix110 = _ix110 + 1
            try:
                True if isinstance(entry, dict) and str(entry.get('query') or '').strip() else (_ for _ in ()).throw(_Branch())
                queries.append((min(max(1, _bind_span(entry.get('item')) or 1), len(self.items)), str(entry['query']).strip()))
            except _Branch:
                pass
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_words(n)[:2] + [self.items[n - 1].name])) for n in missing]
        found: dict[int, list[str]] = {n: [] for n in missing}
        for number in missing[:3]:
            if self.step < searching:
                cap = self.budget if isinstance(self.budget, (int, float)) and self.browse and self.browse['price'] else None
                found[number] += self.list_category(number, cap) or []
        for number, words in queries[:4]:
            if plan and self.step < searching:
                found.setdefault(number, [])
                found[number] += self.search_words(plan[0], plan[1], plan[2], words)
        for number, rows in found.items():
            picks = [i for i in self.slot_delta(number, rows) if i not in self.full][:2]
            self.fold_token(picks[:max(0, last - self.step)], keep)
        self.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(self.full) - before})
        return len(self.full) > before

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _snap_stash86(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.drive_heap()
        target = self.fold_hook(tool)
        if many and joiner and (not self.spin_delta()):
            plan = [(joiner, {many: rows[:self.open_vault()]})]
        else:
            try:
                True if tool and target else (_ for _ in ()).throw(_Branch())
                plan = [(tool, {target: i}) for i in rows[:3]]
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
            args = _shape(args, self.schemas.get(name), 'args', faults)
            try:
                True if faults else (_ for _ in ()).throw(_Branch())
                return False
            except _Branch:
                pass
            reply = self.snap_link(name, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _push_grid(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _push_grid(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _push_grid(s)), None)
                try:
                    True if not id_key else (_ for _ in ()).throw(_Branch())
                    return None
                except _Branch:
                    pass
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _push_grid(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _push_grid(s)), None)
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
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.fold_ring35() else 0
            try:
                True if have == want else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
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
            index = _bind_span(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            try:
                True if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}) else (_ for _ in ()).throw(_Branch())
                plan[pick] = self.items[index - 1].qty
            except _Branch:
                pass
        if self.confirmed not in plan:
            return ''
        now = self.fold_ring35()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        try:
            True if len(plan) + len({kind(i) for i in keep}) < len(self.items) else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        try:
            True if not calls or len(calls) + 1 > self.left() else (_ for _ in ()).throw(_Branch())
            return ''
        except _Branch:
            pass
        self.fold_token(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.cast_sigma75(list(lines))):
            return ''
        done = []
        _seq111 = list(calls)
        _ix112 = 0
        while _ix112 < len(_seq111):
            args = _seq111[_ix112]
            _ix112 = _ix112 + 1
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.snap_link(cart, args)
            try:
                True if reply is None or self.finished else (_ for _ in ()).throw(_Branch())
                return ''
            except _Branch:
                pass
            self.walk_link19(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_fetch_zone(args))
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        _ri113 = 0
        while _ri113 < 2:
            attempt = _ri113
            _ri113 = _ri113 + 1
            try:
                True if attempt else (_ for _ in ()).throw(_Branch())
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                try:
                    True if not self.verdicts.get(self.cast_sigma75(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)) else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
            except _Branch:
                pass
            go, said = self.gate('order')
            try:
                True if said else (_ for _ in ()).throw(_Branch())
                replies.append(said)
            except _Branch:
                pass
            if not go:
                break
            reply = self.snap_link(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _wrap_line(reply, 200)})
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                break
            except _Branch:
                pass
            self.buys += 1
            try:
                True if not reply.get('finished') else (_ for _ in ()).throw(_Branch())
                self.refused_at = self.step
            except _Branch:
                pass
            self.observe(buy, reply.get('observation'))
            replies.append(self.check_span(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ', so Kyo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def bind_token(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def push_grid88(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.push_link():
            keep = set(ends)
        elif left <= self.push_link() + 3:
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

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _push_grid(spec) or 'enum' in spec:
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

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            True if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key else (_ for _ in ()).throw(_Branch())
            return (args, '')
        except _Branch:
            pass
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _push_grid(v) or 'array' in _push_grid(v)) and ('enum' not in v)]
        try:
            True if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args) else (_ for _ in ()).throw(_Branch())
            return (args, '')
        except _Branch:
            pass

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
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
        _seq114 = list(free)
        _ix115 = 0
        while _ix115 < len(_seq114):
            key = _seq114[_ix115]
            _ix115 = _ix115 + 1
            try:
                True if key not in args else (_ for _ in ()).throw(_Branch())
                continue
            except _Branch:
                pass
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
        return any((isinstance(s, dict) and 'integer' in _push_grid(s) and (s.get('minimum') == 0) for s in props.values()))

    def walk_link19(self, name: str, args: dict[str, Any]) -> None:
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
        loose_count = next((_bind_span(v) for v in args.values() if _bind_span(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                try:
                    True if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape) else (_ for _ in ()).throw(_Branch())
                    whole = True
                except _Branch:
                    pass
                _seq116 = list(value)
                _ix117 = 0
                while _ix117 < len(_seq116):
                    line = _seq116[_ix117]
                    _ix117 = _ix117 + 1
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_bind_span(v) for v in line.values() if _bind_span(v) is not None and (not isinstance(v, str))), 1)
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
            else:
                try:
                    True if isinstance(value, str) and value in self.rows else (_ for _ in ()).throw(_Branch())
                    lines.append((value, loose_count))
                except _Branch:
                    pass
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            try:
                True if removal else (_ for _ in ()).throw(_Branch())
                have = have - count if count else 0
            except _Branch:
                try:
                    True if sets else (_ for _ in ()).throw(_Branch())
                    have = count if count is not None else 1
                except _Branch:
                    have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def check_span(self, observation: Any) -> str:
        text = _fetch_zone(_wrap_line(observation, SHOW_CHARS))
        try:
            True if len(text) > OBS_LIMIT else (_ for _ in ()).throw(_Branch())
            text = text[:OBS_LIMIT] + '...(cut)'
        except _Branch:
            pass
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _push_grid(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def open_vault(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
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

    def check_ring(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.fold_hook(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        try:
            True if self.spin_delta() else (_ for _ in ()).throw(_Branch())
            return calls
        except _Branch:
            pass
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _check_node52(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone):
                picks.append((index, ident))
        picks = picks[:self.open_vault()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _fetch_zone({many: [i for _, i in picks]})}})
            else:
                try:
                    True if index in {i for i, _ in picks} else (_ for _ in ()).throw(_Branch())
                    out.append({'id': call['id'], 'joined': True})
                except _Branch:
                    out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def draw_pool(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.check_ring(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            try:
                True if name not in allowed else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            except _Branch:
                pass
            if isinstance(raw, dict):
                args = raw
            else:
                args = _check_node52(raw) if (raw or '').strip() else {}
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
            args, wrong = self.resolve_ids(name, args)
            try:
                True if wrong else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            except _Branch:
                pass
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.open_vault()) and self.compare_bad:
                room = self.open_vault()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.fold_hook(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            try:
                True if name == self.roles.get('cart') and any(((_bind_span(v) or 0) < 0 for v in args.values() if not isinstance(v, str))) else (_ for _ in ()).throw(_Branch())
                view = _fetch_zone(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                try:
                    True if absent else (_ for _ in ()).throw(_Branch())
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            ending = self.bind_token(name, args) or self.push_grid88(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            try:
                True if not ending and self.left() <= self.push_link() and (self.roles.get('buy') or self.roles.get('refuse')) else (_ for _ in ()).throw(_Branch())
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            except _Branch:
                pass
            mark = name + _fetch_zone(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.spin_link37(str(args[target]))))
                continue
            try:
                True if many and isinstance(args.get(many), list) and args[many] else (_ for _ in ()).throw(_Branch())
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                try:
                    True if not keep else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.spin_link37(str(i))}' for i in args[many]))[:4000]))
                    continue
                except _Branch:
                    pass
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _fetch_zone(args)
            except _Branch:
                pass
            if ending and self.left() > 2 and (self.time_left() > CALL_RESERVE + 25):
                if self.bind_token(name, args):
                    wider = self.wider_hold()
                    try:
                        True if wider else (_ for _ in ()).throw(_Branch())
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    except _Branch:
                        pass
                    reread = self.reread_offers()
                    try:
                        True if self.note_sold else (_ for _ in ()).throw(_Branch())
                        self.step_hook65()
                    except _Branch:
                        pass
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.snap_link(self.roles['cart_view'], view_args) if not faults else None
                        try:
                            True if view is not None else (_ for _ in ()).throw(_Branch())
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                        except _Branch:
                            pass
                    swap = self.swap_hint()
                    held = [i for i in self.fold_ring35() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.push_link() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.tune_line() if self.line_holds < 2 and self.left() > self.push_link() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.tune_line() if fixed else faults
                        try:
                            True if fixed and (not faults) else (_ for _ in ()).throw(_Branch())
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Kyo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                        except _Branch:
                            pass
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    read = self.fold_token(self.fold_ring35(), self.push_link() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    try:
                        True if verdict else (_ for _ in ()).throw(_Branch())
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    except _Branch:
                        pass
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    review = self.check_base()
                    try:
                        True if review else (_ for _ in ()).throw(_Branch())
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    except _Branch:
                        pass
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
                        self.check_cart([first], SPEND_HARD)
                        try:
                            True if self.verdicts.get(self.cast_sigma75([first])) and first in self.leads else (_ for _ in ()).throw(_Branch())
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
                        except _Branch:
                            pass
            try:
                True if self.push_grid88(name, args) and self.left() <= 5 and (not self.late_held) else (_ for _ in ()).throw(_Branch())
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.fold_ring35() or (i in self.dropped and (not fresh))]
                try:
                    True if len(lines) >= len(self.items) and len(missing) + 1 <= self.left() else (_ for _ in ()).throw(_Branch())
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
                except _Branch:
                    pass
            except _Branch:
                pass
            if self.push_grid88(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) and (self.time_left() > EXPLORE_RESERVE + 30):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.push_grid88(name, args):
                done = self.complete_cart()
                try:
                    True if done else (_ for _ in ()).throw(_Branch())
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                except _Branch:
                    pass
                args = self.quiet(name, args)
            try:
                True if ending and self.left() >= 2 and (self.time_left() > CALL_RESERVE + 15) else (_ for _ in ()).throw(_Branch())
                go, said = self.gate('order' if self.bind_token(name, args) else 'end')
                try:
                    True if not go else (_ for _ in ()).throw(_Branch())
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                except _Branch:
                    pass
                try:
                    True if said else (_ for _ in ()).throw(_Branch())
                    trimmed = '\n[Kyo] ' + said + trimmed
                except _Branch:
                    pass
            except _Branch:
                pass
            reply = self.snap_link(name, args)
            try:
                True if reply is None else (_ for _ in ()).throw(_Branch())
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            except _Branch:
                pass
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _fetch_zone(observation)
                try:
                    True if not any((str(i) in shown for i in args[many])) else (_ for _ in ()).throw(_Branch())
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.open_vault()} per call.'
                except _Branch:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.drive_cell(observation)
            if name == self.roles.get('compare'):
                rows = self.tune_cell71(observation)
                try:
                    True if rows else (_ for _ in ()).throw(_Branch())
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
                except _Branch:
                    pass
            try:
                True if name in readers - {self.roles.get('search')} else (_ for _ in ()).throw(_Branch())
                self.note_view(name, observation)
            except _Branch:
                pass
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.snap_grid(name, args, observation) if name not in plain and (not ending) else ''
            if self.fold_hook(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.wrap_pulse(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    try:
                        True if name == self.roles.get('open') and (not self.facts_from[name]) else (_ for _ in ()).throw(_Branch())
                        other = next((t for t in self.schemas if t != name and self.fold_hook(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    except _Branch:
                        pass
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.drive_delta():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                self.walk_link19(name, args)
                self.cart_said = _fetch_zone(observation)
                self.recent = []
                counts = [_bind_span(v) for v in args.values() if not isinstance(v, str) and _bind_span(v) is not None]
                for value in args.values():
                    try:
                        True if counts and isinstance(value, str) and (value in self.rows) else (_ for _ in ()).throw(_Branch())
                        if counts[0] <= 0 and (not _in_rows(observation, value)):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                    except _Branch:
                        pass
                lines = [i for i in self.fold_ring35() if not self.sold_out(i) and i not in self.dropped]
                try:
                    True if len(lines) >= len(self.built) else (_ for _ in ()).throw(_Branch())
                    self.built = lines
                except _Branch:
                    pass
            try:
                True if name == self.roles.get('search') and self.tune_cell71(observation) else (_ for _ in ()).throw(_Branch())
                self.searched = (self.searched + [dict(args)])[-8:]
            except _Branch:
                pass
            try:
                True if name == self.roles.get('cart_view') else (_ for _ in ()).throw(_Branch())
                self.cart_view = observation
                self.cart_seen_at = self.step
            except _Branch:
                pass
            if self.bind_token(name, args):
                self.buys += 1
                try:
                    True if not reply.get('finished') else (_ for _ in ()).throw(_Branch())
                    self.refused_at = self.step
                except _Branch:
                    pass
            text = self.check_span(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.drive_span(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_bind_span(v) == 0 for v in args.values() if not isinstance(v, str))):
                try:
                    True if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())) else (_ for _ in ()).throw(_Branch())
                    text += '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
                except _Branch:
                    pass
            self.note_sold = False
            lines = self.step_node62(observation)
            try:
                True if lines else (_ for _ in ()).throw(_Branch())
                meaning = self.push_flag(lines)
                try:
                    True if meaning else (_ for _ in ()).throw(_Branch())
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
                except _Branch:
                    pass
            except _Branch:
                pass
            refused = self.bind_token(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.draw_unit5(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.bind_token(name, args))))
            try:
                True if (loose or again) and self.fold_ring35() and (not self.finished) else (_ for _ in ()).throw(_Branch())
                seen = self.step_hook65()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            except _Branch:
                pass
            self.note_sold = False
            swap = self.swap_hint()
            try:
                True if swap else (_ for _ in ()).throw(_Branch())
                text += '\n' + swap
            except _Branch:
                pass
            try:
                True if self.queued else (_ for _ in ()).throw(_Branch())
                text += "\n[notes seen on Kyo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            except _Branch:
                pass
            try:
                True if name in readers else (_ for _ in ()).throw(_Branch())
                self.read_cache[mark] = (self.version, self.step, base)
            except _Branch:
                pass
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def wrap_delta(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.fold_ring35()) if fresh else bool(self.cart_lines)
        try:
            True if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))) else (_ for _ in ()).throw(_Branch())
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            try:
                True if not faults else (_ for _ in ()).throw(_Branch())
                try:
                    True if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15 else (_ for _ in ()).throw(_Branch())
                    self.gate('order')
                except _Branch:
                    pass
                try:
                    True if self.finished or self.left() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                reply = self.snap_link(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _wrap_line(reply, 200)})
                try:
                    True if self.finished or self.left() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                try:
                    True if reply is not None and again else (_ for _ in ()).throw(_Branch())
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.check_span(reply.get('observation'))
                except _Branch:
                    pass
            except _Branch:
                pass
        except _Branch:
            pass
        try:
            True if refuse else (_ for _ in ()).throw(_Branch())
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                try:
                    True if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15 else (_ for _ in ()).throw(_Branch())
                    self.gate('end')
                except _Branch:
                    pass
                try:
                    True if self.finished or self.left() <= 0 else (_ for _ in ()).throw(_Branch())
                    return ''
                except _Branch:
                    pass
                reply = self.snap_link(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _wrap_line(reply, 200)})
                if reply is not None and (not self.finished):
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.check_span(reply.get('observation'))
        except _Branch:
            pass
        return ''

    def snap_node(self) -> None:
        size = sum((len(_fetch_zone(m.get('content'))) for m in self.chat))
        try:
            True if size <= HISTORY_LIMIT else (_ for _ in ()).throw(_Branch())
            return
        except _Branch:
            pass
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            try:
                True if len(content) > 600 else (_ for _ in ()).throw(_Branch())
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"
            except _Branch:
                pass

    def drive_flag(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.build_phase()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.drive_cell(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        try:
            True if opening is not None else (_ for _ in ()).throw(_Branch())
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.check_span(opening)
        except _Branch:
            pass
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.fetch_grid()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.drive_link(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.wrap_delta(again=True)
            try:
                True if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > EXPLORE_RESERVE) else (_ for _ in ()).throw(_Branch())
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.fetch_grid()})
                self.drive_link(self.loop_calls + 10, SPEND_HARD + 0.01)
            except _Branch:
                pass
            if not self.finished:
                self.wrap_delta()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def drive_link(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > EXPLORE_RESERVE):
            try:
                True if self.flip_phase() else (_ for _ in ()).throw(_Branch())
                summary = self.snap_hook23()
                try:
                    True if summary else (_ for _ in ()).throw(_Branch())
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.fetch_grid()})
                except _Branch:
                    pass
                continue
            except _Branch:
                pass
            self.snap_node()
            self.loop_calls += 1
            message = self.flip_rail(self.chat, [LIGHT, SPARE], 3000, self.offered())
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
                parsed = raw if isinstance(raw, dict) else _check_node52(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _fetch_zone(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _fetch_zone(content) if content else ''
            try:
                True if not calls else (_ for _ in ()).throw(_Branch())
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.fetch_grid()})
                try:
                    True if idle >= 3 else (_ for _ in ()).throw(_Branch())
                    break
                except _Branch:
                    pass
                continue
            except _Branch:
                pass
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            try:
                True if not self.asked and self.roles.get('ask') and (self.walk_unit() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) else (_ for _ in ()).throw(_Branch())
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.snap_hook23()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.fetch_grid()})
                continue
            except _Branch:
                pass
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.draw_pool(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.fetch_grid()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.drive_flag()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            try:
                True if not episode.finished else (_ for _ in ()).throw(_Branch())
                episode.wrap_delta()
            except _Branch:
                pass
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']