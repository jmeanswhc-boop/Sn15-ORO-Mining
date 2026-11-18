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
LIST_CAP = 8              
HEARD_ONE = (0.92, 0.88, 0.58, 0.36, 0.28, 0.16, 0.12, 0.1)
HEARD_MANY = (0.8, 0.78, 0.66, 0.66, 0.64, 0.6, 0.6, 0.6)
NEEDS_ONE = 1.0           
NEEDS_EACH = 0.5           
BELIEF_CAP = 0.8           # no property is ever taken as near-certain before the shopper says so
GAIN_FLOOR = 0.02          # a message expected to reveal less than this is not sent
SWEEP_OPENS = 6
CLOCK_SOFT = 225.0
CLOCK_HARD = 262.0
CLOCK_LLM_END = 280.0
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
CARVE_SEARCHES = 6         # extra searches Keystone runs itself to reach listings a capped category search cannot show
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Keystone, buying on behalf of one shopper in an online store, and you act only through the store\'s tools. An episode scores 1 or 0. It scores 1 when the order holds every requested item at its quantity, each meeting every requirement still in force at checkout, within the budget; or, when the store really has nothing that meets everything, when the episode ends with the no-purchase tool the way the request asks.\n\n1. THE STORE\n- Each tool call costs one step and steps run out; thinking is free. Every store names its tools, arguments and results its own way: read each description and schema, and use the exact argument names and allowed values.\n- Searches can be narrow: a category filter, a keyword field that may only read the first words, a few rows per page, a token or offset for the next page. Compare the reported total with the rows you have, and page on with the exact token or offset when the answer depends on seeing them all. Paging can stop early; then the rest is reachable only through other search words. Short, value-first queries work best (the wanted value and the product noun). A filtered search with no rows usually has a wrong filter value or parameter: retry with a label exactly as rows show it, or with another filter.\n- A search row is a summary. Stores split a listing differently: one view shows the attributes, another only the offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Keystone has learned it. Judge each property requirement from those attributes, so read them for every listing you might buy; one call that shows several listings saves steps when it shows what you need. Do not reopen a listing that has not changed.\n- Keystone\'s interviewer talks to the shopper; you never message them yourself. Its findings arrive in the conversation and in the ledger.\n- The request may ask for steps along the way (an approval, a message, a check). The ledger lists them. Keystone runs the ones due just before the order or the ending when you call checkout or the ending tool, so never wait for them and never decline because of them; do any others with the store\'s tools (product questions still go through the interviewer).\n\n2. WHEN A LISTING FITS\n- Kind: the listing\'s category label must match the requested item as a whole; a neighbouring label does not count.\n- Properties: a requirement is met when the listing\'s attribute for that property states the value. Key spelling never matters (spaces, underscores, camel case, prefixes) and one element of a list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value containing the wanted words counts ("deep teal" for teal, "extra roomy fit" for roomy); different words do not ("teal green" is not "sea green", and a code or abbreviation is not the word it stands for). Convert units before comparing numbers; limits include the edge; a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (in any spelling or a plain synonym) decides: if it says something else, the listing fails, whatever the title or other keys say. If several attributes name the property, one stating the value is enough. A key for a different property, even a related one (a handle colour is not the body colour), decides nothing. After opened listings Keystone adds [evidence] lines with the keys that match each requirement.\n- With no attribute for the property, the wanted value stated under another attribute (say, a list of styles that includes it) or in the title or description counts; a number of another kind does not (a weight is not a capacity), nor does a bare figure whose text does not say what it measures.\n- A category label that itself states a required value meets that requirement.\n- Some requirements point at another listing instead of giving a value ("the same X as the one sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve them first: find that listing (search its category, brand, shop or words), open its page in full with a single-listing or side-by-side tool (a row is not enough), and when that page sends you on to a further listing for the property, find and open that one too; the last page gives the value. The order counts only when every page of the chain was opened. When the wish is anything but that other value, it excludes only listings sharing that exact value, and it is never a reason to end without buying.\n- A page may state a rule that links one value to another; when a requirement uses that link, read the other value by following its chain and apply the rule as the store words it. A size written as a count times a unit size counts as their product. When a listing you are about to buy refers you to another page about itself, look at that page first.\n- Parts and devices: when one item must fit another or a device the shopper owns, choose them together from the listings\' own compatibility words. Keystone reads those words for you: the [fit atlas] lines show which device candidates the seen parts fit, counting every name a device is sold under, code families such as 12xx and the models a part excludes; trust them over your own reading. Search does not read compatibility words, so a search for a model code finds nothing: search or browse the part\'s kind by product words and pick from the rows the atlas marks. When no part that meets its own requirements fits the device you chose, switch to another device that meets its requirements and that such a part fits; never end without buying while that switch is untried.\n- Exclusions and closed lists are binding. Wishes ("ideally", "would be nice") only break ties. Never buy a line the store shows as sold out, and the total (price times quantity, over all lines) may not pass the budget; equal is fine.\n\n3. CHOOSING\n- Cheapest (lowest price, smallest total): the order must hold the lowest-priced listing among ALL that fit. Find the whole fitting set (page through it: rows often come in no price order, and Keystone reads the next pages of a short result itself), check candidates from the cheapest up and take the first that fully fits. If a search takes a maximum price, list the category below your current best pick so only cheaper candidates remain.\n- Most or least of a measure: compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right; do not spend steps hunting for better.\n- Put each item in the cart as soon as you have opened a listing that meets all its requirements (for cheapest, most or least: once the fitting set is compared), then move on. Stores often sell out a carted line soon after, and an early cart leaves steps to replace it.\n- Several different items need one cart line each, at the requested quantity.\n\n4. CHANGES WHILE YOU WORK\n- The shopper or the store may change the task: a new budget, a requirement added or dropped, a new price, a line sold out. Such notes appear inside tool results, often as a refused checkout, whichever tool shows them; the ledger keeps the requirements in force. A refusal does not end the episode: apply every change, fix the cart and check out again.\n- Re-read each chosen listing shortly before checkout, since price and stock can change without a note.\n\n5. FINISHING\n- Set the cart to exactly the chosen lines (remove anything else), look at the cart once, then call checkout on its own.\n- End without buying only after a real search shows that no listing meets everything. Being short of steps proves nothing of the kind: then check out the best cart you have. End the way the request asks, and when a remark is needed keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'Turn a shopper\'s request and the store\'s tool list into a work order for a buying assistant. Return ONE compact JSON object (no indentation, nothing else).\n\nHOW TO READ THE REQUEST\n- Items: one entry per distinct product to buy. A product named in one phrase joined by "and" (one "I want ..." with one quantity, often one store department) is ONE item; split only when each product gets its own quantity or its own requirements.\n- Requirements ("musts") are firm conditions on a product, each written "property: constraint" with the exact value, limit, exclusion or closed list, under the request\'s own word for the property ("for a canal-side picnic" gives "occasion: canal-side", "a 9-inch blade" gives "blade length: 9 inch"); never swap in a neighbouring property.\n- Price, budget, stock and quantity are never musts. Story (who uses it, where it goes, what went wrong before, the event behind it) is context, not a must, unless it states a value or limit outright. Words that are part of the product\'s name belong to "name", not to musts.\n- A use the request makes a property of the product itself ("jackets for office wear", "a dress made for dancing", "the occasion must be garden party") IS a must on that property, unless the store\'s category label already says it. An event, person or setting the product is bought for, and how it should feel or look in use, stay context.\n- Whatever the shopper says they will check or verify is a firm must however it is worded, even when it names a use, an event or a kind ("they are for a rowing day, I will check" gives "occasion: rowing day"; "it must be a cloche, I will look at the label" gives "kind: cloche").\n- A must whose value is read from another listing is ONE must holding that listing\'s whole description ("handle colour: unlike the handle colour of the jug Belvaro sells with a cork stopper"); the words describing that other listing (its shop, its kind, "with <property> <value>") stay inside it and are never musts of this product.\n\nKEYS\n"items": [{"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copied exactly), else "", "quantity": units of that one listing (1 when not stated; a pair or set sold as one listing is 1), "musts": [...]}]\n"budget": the most the shopper will pay in total, as a number in the request\'s unit, or null. "budget_unit": that unit in words, or null. "budget_scope": "total" | "each" | null.\n"in_stock_only": true unless the request clearly accepts items that cannot be bought now.\n"pick": "cheapest" for the lowest price or smallest total, "most" or "least" for the extreme of a measure, else "any" (remarks about tight money are not a pick rule). "pick_measure": the measure for most or least, else "".\n"messages": how many messages the shopper allows (integer) or null. "steps": how many steps, moves or tool calls are allowed (integer) or null.\n"lookups": the musts (copied exactly) whose value is read first from another listing, else [].\n"rules": every instruction about how to work or how to end, each in at most 12 words. "wishes": things the shopper would merely like. "ask_first": things the request says to check with the shopper before choosing ([] when none).\n"process": steps the request requires besides finding and buying (approval, telling or showing the shopper something, checking with them, reading something first, using a certain tool), each {"step": at most 15 words, "when": "choose" | "order" (right before checkout) | "end" (just before ending without buying), "messages": shopper messages it takes (0 when none), "reply": true when it needs the shopper\'s answer before going on}; [] when none. Requirements, budget, stock, the message and step limits and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest alternatives when nothing fits, false when it asks for them, null when silent.\n"fits": [{"item": item number, "with": item number}] for each item the request says must fit, suit or work with ANOTHER requested item; [] when none.\n"fit_codes": [{"item": item number, "code": the model code or model name exactly as written}] for each item that must fit a device the shopper already owns and names (not a requested item); [] when none.\n"roles": the exact tool for each job, or null: "search", "open" (shows one listing\'s attributes; of two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (one listing\'s current price and stock, when that is a separate tool), "compare" (several listings in full), "ask" (messages the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without buying).\n"buy_args": fixed arguments that make "buy" check out (for a tool with a mode argument, that mode), else {}. "refuse_args": fixed arguments that make "refuse" end without buying, else {}.\n'
PLAN = 'Help plan a short interview with a shopper. Besides what the request says, the shopper holds a few firm wishes they never wrote down, each about one property of one requested product. A wish comes out only when a question names its property, and only a few questions can be sent, so the order of properties matters.\n\nYou get the request, the requested items with what the request already fixes for each, and the attribute keys seen on opened listings of each item (with how many listings carry each key and sample values). A wish may also be about a property no listed key names.\n\nFor every item, list the properties worth asking about, most promising first, and rate each:\n- "hinted": the request itself asks to check it with the shopper (ASK FIRST), or its story points at it without giving the value (someone whose taste decides, how it should look or where it goes, who will use it, what goes inside it, something the shopper says they cannot judge);\n- "likely": a property buyers of this kind of product usually choose between, carried by most of the item\'s listings;\n- "possible": anything else worth a place.\nLeave out properties the request already fixes for that item (a key whose values only repeat a required value is NOT fixed), a second name for a property already listed, keys whose values list several purchase options at once, identifiers, model names, warranty and delivery. A key marked [not the stated ...] is a different property from that requirement. Pack sizes and counts are ordinary properties.\nName each property plainly with all the words of its key ("strap colour" for strapColour, never just "colour"); no values, no brackets, one property per entry.\nUp to COUNT per item. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "rate": "hinted" | "likely" | "possible", "hint": "the request words it comes from, at most 8, or empty"}]}'
HEAR = 'You read a shopper\'s answers to a buying assistant. Each answer either states a firm wish, says the shopper does not mind, says they did not follow the message, or asks for fewer things per message.\n\nRules for wishes:\n1. A wish is what the shopper insists on. Phrases that refuse to give something up ("not up for trade", "won\'t budge on", "I can bend on most things but not") insist on that value. Write "not X" only when the shopper rejects X.\n2. Name the wish by the property the answer names (it may differ from the property asked about) and copy the value exactly as said, nouns and units included ("at least 3 shelves", "oval", "not wicker"). Several answers about one property are one wish.\n3. When the answer gives the value by comparison with another listing (like, unlike or close to the value of some shop\'s product, which it picks out by a few of its properties), that is one wish with "points": true, whose value is the answer\'s own words from the comparison to the end of that description, unchanged. The properties that describe the other product are not wishes.\n4. Homely or roundabout phrasing that still names a value states that value.\n5. "asked": the property named in the question that this wish answers, copied from the question; null when none fits.\n\nOther outcomes:\n- "free": the property the shopper said they do not mind, only when the question named a single property or the answer says which one.\n- "crowded": numbers of the questions whose answer complains about too many things at once or asks for one question at a time (not answers that only say they did not follow).\n- "puzzled": numbers of the questions whose answer only says they did not follow or understand.\n- "drop": requirements the shopper now gives up. "budget": a new total limit as a number, or null.\n\nReturn ONE JSON object and nothing else: {"needs": [{"item": item number, "need": "property: constraint", "points": true or false, "asked": "..." or null}], "drop": [{"item": item number, "property": "..."}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers], "puzzled": [question numbers], "budget": number or null}'
NOTE = 'A note appeared while a buying assistant was working; it may come from the shopper or from the store. Classify it and list every change it makes. A requirement that is replaced counts as a drop of the old one plus an add of the new one.\nReturn ONE JSON object and nothing else: {"kind": "change" (budget, requirements, a price, availability, or a refused order) | "error" (a call was malformed or not accepted) | "other", "budget": the new total limit as a number in the request\'s unit or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or unavailable (named or not), "price_moved": true when it says a price changed, "summary": "one short sentence"}'
READING = 'judge each requirement from the attribute whose key names the same property (any spelling or plain synonym; one list element is enough; synonyms, unit conversion and values containing the wanted words, such as "deep teal" for teal, all count; limits include the edge); a key for another property, even a related one (a handle colour is not the body colour), decides nothing; with no such attribute, the wanted value stated under another attribute, in the title or in the description counts, but a bare figure that does not say what it measures does not; a category label that states the value meets it, and the label must match the item; being unlike another listing\'s value holds unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
CHECK = 'You review a cart just before checkout and decide from the records whether it is exactly right.\nHow to read the records: ' + READING + ' When several attributes name the property, one stating the value is enough. A number of another kind proves nothing (a weight is not a capacity). Where the records state a rule linking one value to another, apply it; a size written as a count times a unit size is their product. The newest record of a listing wins.\nWhat the cart needs: one separate line per requested item at its quantity, nothing else, nothing shown out of stock, and a total (price times quantity over all lines) within the budget. When the pick rule is cheapest, most or least, look at the other listings shown as well: one that clearly fits everything and beats a cart line on that rule makes the cart wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Say ok=false only for a problem the records clearly show, or a requirement no record addresses.'
AUDIT = 'A buying assistant believes nothing in the store fits and wants to end without buying. Before it does, check the listing records against the requirements. How to read the records: ' + READING + '\nRecords marked ROW were seen only as search rows: their attributes are unread, so a property the row does not show can neither pass nor fail them.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show a fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'Some requirements in a shopper\'s request take their value from another listing the request points at. The words that pick out that other listing (its kind, its shop, the properties after "with ...") are sometimes misread as requirements on the product being bought. Using the request\'s own wording, say which of the OTHER REQUIREMENTS below only describe that other listing.\nReturn ONE JSON object and nothing else: {"describe_other": [those requirements, copied exactly; empty when all are real requirements on the product]}'
TRAIL = 'A buying assistant is about to check out. Some requirements take their value from another listing, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when every listing along each such chain was opened in full. Follow each chain from the requirement\'s own description, link by link, through the pages\' words, using the records below (OPENED: the page was opened; ROW: only a search row was seen).\nA record is a link only when it matches the whole description of that link: shop or brand, kind of product AND every property value the description gives (a "10-piece multipack" rules out the same shop\'s single 10-piece box). When no record matches all of it, the link is not found yet, even if the shop sells other listings of that kind; give search words for it (the shop\'s name and the description\'s values). List each chain\'s links in order and stop at the first link that is not OPENED or not found, since its words are still unknown.\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record is that link, else "", "about": how the requirement or the previous page describes it, "find": search words for a link not found}], "done": true when the last link is an OPENED page stating the property\'s value itself (also when the requirement asks for a value other than or unlike it), else false, "value": that value as the page states it, or ""}]}'
GATE = 'A buying assistant is about to STAGE, and the shopper\'s request asks for the steps listed below on the way. Using the numbered RECORD of what the assistant has done, decide for each step due before this STAGE whether it is done for the cart as it stands now. A step is done only when a RECORD line shows it done; for a step about the cart or the order, that line must come after the last cart change and its reply must grant it. Questions about product properties never do a step.\nIf a step is still missing, give the single store call that does it next (a tool from TOOLS with its arguments). Messages to the shopper go through the messaging tool: one short plain sentence or question about that step only, naming no product property or value the step does not need. Give no call when every due step is done, or when the messages or steps left cannot do it.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [RECORD line numbers showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A line in the buying assistant's cart has just sold out. Among the other records shown, find listings that could replace it. How to read the records: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number, as the record shows it} for each listing that appears to meet every requirement of the sold-out item, at most 6], "reason": "one sentence"}'
SHAPE = 'Below is a result from a store tool. Name its parts with the exact keys used by the result and by the tool\'s parameters. Return ONE JSON object and nothing else: {"rows": key of the list of listings, "count": key of the total number of matches (not the number shown) or null, "next": key of the value that asks for the next page or null, "next_arg": the tool parameter that takes that value or null, "price": key of a row\'s price or null, "stock": key, inside a row, saying whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
WIDEN_EACH = 6

PROOF_SKIP = '\\d|\\b(?:same as|other than|unlike|listing|page|shown|within|at most|at least|under|over|below|above|less|more|between|up to|maximum|minimum)\\b'
PROOF_NOT = '^\\s*(?:not|no|non|never|except|excluding|exclude|without|anything but)\\b[\\s:-]*'
PROOF_SCORE = {'pass': 4, 'weak': 2, 'conflict': 1, 'unknown': 1, 'differs': 0}
PROOF_LINK = {'a', 'an', 'the', 'of', 'in', 'with', 'and', 'to', 'for', 'on', 'it', 'is'}
PROOF_SHOW = 6
PROOF_LEARNS = 6
PROOF_ROOM = 12
PROOF_LINES = 90
PROOF_READ = 'A shopper set a requirement on a product. Below are numbered attribute lines (key = value) from listings of that kind in one store. For each line decide two things. Does its key name the requirement\'s property (the same property under any spelling or a plain synonym, such as "material" for "made of"; a related but different property does not count)? Does its value meet the requirement (case, singular or plural and plain synonyms are fine, one list element is enough, a value containing the wanted words counts, units are converted before numbers are compared, limits include the edge, a range must lie wholly inside a limit, an exclusion is met by any other value; an abbreviation or code is not the word it stands for)?\nReturn ONE JSON object and nothing else: {"keys": [keys, copied exactly, that name the requirement\'s property], "yes": [numbers of the lines, under any key, whose value meets the requirement]}'

def _proof_words(text: Any) -> frozenset[str]:
    words = re.findall('[a-z0-9]+', str(text or '').lower())
    return frozenset(w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in words if w not in PROOF_LINK)

def _proof_alts(value: Any) -> list[frozenset[str]]:
    parts = re.split('\\s+or\\s+|\\s*/\\s*|\\s*,\\s*', str(value or ''))
    return [w for w in (_proof_words(p) for p in parts) if w]

def _proof_holds(alts: list[frozenset[str]], text: Any) -> bool:
    words = _proof_words(text)
    return any(alt <= words for alt in alts)


ATLAS_CODE = re.compile('(?<![A-Za-z0-9])([A-Z][A-Za-z]{0,7})([- ]?)(\\d{1,6})((?:[xX]{2,4}|[*?#]{1,4})|[A-Za-z]{0,3})(?![A-Za-z0-9])')
ATLAS_CLAUSE = re.compile('(?<![a-z])(?:(in|not\\s+|never\\s+)?(?:compatib\\w*(?:\\s+(?:with|models?|devices?|for))?|fits|fitting|fit\\s+for|works?\\s+with|for\\s+use\\s+with|suitable\\s+for|designed\\s+for|made\\s+for))\\b\\s*[:=]?\\s*', re.I)
ATLAS_IDENT = re.compile('(?<![a-z])(?:model(?:\\s*(?:no\\.?|number|#|code|name))?|(?:also\\s+)?sold\\s+as|(?:also\\s+)?known\\s+as|aka|a\\.k\\.a\\.)\\s*[:#=]?\\s*', re.I)
ATLAS_EXCEPT = re.compile('(?<![a-z])(?:except(?:\\s+for)?|excluding|exclude[sd]?|but\\s+not|other\\s+than|apart\\s+from|not\\s+(?:for|with|the))(?![a-z])', re.I)
ATLAS_ALL = re.compile('(?<![a-z])(?:every|all|any|each)(?![a-z])', re.I)
ATLAS_END = re.compile('[.;|\\n]\\s|[;|\\n]|\\.$')
CART_PLAN = 'A shopping agent is about to end without buying, but here an order with one listing for each requested item is the only finish that can earn anything. From the listing records seen, choose for EVERY requested item the one listing most likely to meet its requirements: the right kind of product first, then the requirements, then the compatibility lines, then a price that keeps the total inside the budget. Use only ids shown in the records, never one marked SOLD OUT. Return ONE JSON object and nothing else: {"cart": [{"item": item number, "id": listing id}], "reason": "one sentence"}'
ATLAS_READS = 3      # model reads of listings the code could not parse, per episode
ATLAS_BATCH = 12     # listings per read
ATLAS_CHARS = 420    # text per listing in a read
ATLAS_PROMPT = 'Store listings below may state which devices a product fits, or which model a device is. For each numbered listing, return only what its own text states. Copy each code or model name as the text writes it. Write a family of codes (a series, a range, every model starting with something) as the shared start followed by one x for each free digit when the text shows how many digits there are, otherwise as the shared start followed by *. Put models the text says it does not fit, or excludes from a family, under "not". Use empty lists when the text says nothing of the kind; never guess from the kind of product or from words of its title. Return ONE JSON object and nothing else: {"listings": [{"n": listing number, "is": [codes or model names this product itself is or is also sold as], "fits": [codes or families it fits], "not": [codes or families it does not fit]}]}'


def _atlas_tokens(text: str, every: bool=False) -> list[tuple[str, Any]]:
    """Codes in a stretch of text: ('code', key) for a model code, ('family', regex) for a code with wildcards."""
    out: list[tuple[str, Any]] = []
    for hit in ATLAS_CODE.finditer(text):
        letters, _, digits, tail = hit.groups()
        tail = tail or ''
        if not tail and re.match('\\s*(?:series|range|family|line|lineup|generation)\\b', text[hit.end():], re.I):
            out.append(('family', re.compile('^' + re.escape(letters.lower() + digits) + '\\d*[a-z]{0,3}$')))
            continue
        wild = bool(re.fullmatch('[xX]{2,4}|[*?#]{1,4}', tail)) or (every and re.fullmatch('[xX]', tail) is not None)
        if wild:
            if tail[0] in '*':
                body = '\\d+'
            else:
                body = '\\d{%d}' % len(tail)
            out.append(('family', re.compile('^' + re.escape(letters.lower() + digits) + body + '[a-z]{0,3}$')))
        else:
            if len(digits) < 2:
                continue
            out.append(('code', letters.lower() + digits + tail.lower()))
    return out


def _atlas_end(text: str, start: int) -> int:
    hit = ATLAS_END.search(text, start)
    return hit.start() if hit else len(text)


def _atlas_claim() -> dict[str, Any]:
    return {'codes': set(), 'families': [], 'not': set(), 'not_families': [], 'only_not': True}


def _atlas_merge(one: dict[str, Any] | None, two: dict[str, Any] | None) -> dict[str, Any] | None:
    if not one or not two:
        return one or two
    return {'codes': one['codes'] | two['codes'], 'families': one['families'] + two['families'], 'not': one['not'] | two['not'], 'not_families': one['not_families'] + two['not_families'], 'only_not': one['only_not'] and two['only_not']}


def _atlas_read(text: str, notes: dict[str, dict[str, Any]] | None=None) -> tuple[set[str], dict[str, Any] | None]:
    """(identity codes, claim) of one listing's text. A claim holds the codes and families it fits and those it leaves out.
    A clause about another listing ("part X also fits ...") goes into notes under that listing's code instead."""
    text = str(text or '')
    claim: dict[str, Any] | None = None
    spans: list[tuple[int, int]] = []
    for hit in ATLAS_CLAUSE.finditer(text):
        end = _atlas_end(text, hit.end())
        body = text[hit.end():end]
        tokens_all = _atlas_tokens(body, bool(ATLAS_ALL.search(body)))
        if not tokens_all:
            continue
        spans.append((hit.start(), end))
        other = re.search('(?<![A-Za-z0-9])([A-Z][A-Za-z]{0,7}[- ]?\\d{2,6}[A-Za-z]{0,3})\\s+(?:also|too|as well)\\s+$', text[max(0, hit.start() - 40):hit.start()])
        if other:
            if notes is not None:
                key = re.sub('[^a-z0-9]', '', other.group(1).lower())
                note = notes.setdefault(key, _atlas_claim())
                for kind, value in tokens_all:
                    note['only_not'] = False
                    (note['codes'].add(value) if kind == 'code' else note['families'].append(value))
            continue
        claim = claim or _atlas_claim()
        negated = bool(hit.group(1))
        cut = ATLAS_EXCEPT.search(body)
        head, rest = (body[:cut.start()], body[cut.end():]) if cut else (body, '')
        every = bool(ATLAS_ALL.search(body))
        for kind, value in _atlas_tokens(head, every):
            if negated:
                (claim['not'].add(value) if kind == 'code' else claim['not_families'].append(value))
            else:
                claim['only_not'] = False
                (claim['codes'].add(value) if kind == 'code' else claim['families'].append(value))
        for kind, value in _atlas_tokens(rest, every):
            (claim['not'].add(value) if kind == 'code' else claim['not_families'].append(value))
    blank = list(text)
    for start, end in spans:
        for i in range(start, end):
            blank[i] = ' '
    plain = ''.join(blank)
    own: set[str] = set()
    for hit in ATLAS_IDENT.finditer(plain):
        # Only the code right after the cue names the listing, and a list of codes joined by commas or "or" after it.
        body = plain[hit.end():hit.end() + 120]
        while True:
            first = ATLAS_CODE.match(body)
            if not first:
                break
            for kind, value in _atlas_tokens(first.group(0)):
                if kind == 'code':
                    own.add(value)
            joint = re.match('\\s*(?:,|/|&|\\bor\\b|\\band\\b)\\s*', body[first.end():])
            if not joint:
                break
            body = body[first.end() + joint.end():]
    return own, claim


class FitAtlas:
    """Identities and fit claims of every listing seen, with model aliases joined across listings."""

    def __init__(self) -> None:
        self.own: dict[str, set[str]] = {}
        self.claims: dict[str, dict[str, Any]] = {}
        self.parent: dict[str, str] = {}
        self.texts: dict[str, int] = {}
        self.parts: dict[str, set[str]] = {}
        self.notes: dict[str, dict[str, Any]] = {}
        self.read_own: dict[str, set[str]] = {}            # what a model read in a listing the code could not parse
        self.read_claims: dict[str, dict[str, Any]] = {}
        self.cued: set[str] = set()                         # listings with fit or model words but no code the parser could take

    def _root(self, code: str) -> str:
        self.parent.setdefault(code, code)
        while self.parent[code] != code:
            self.parent[code] = self.parent[self.parent[code]]
            code = self.parent[code]
        return code

    def _join(self, one: str, two: str) -> None:
        a, b = (self._root(one), self._root(two))
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def learn(self, ident: str, text: str) -> None:
        size = len(text or '')
        if self.texts.get(ident, -1) >= size:
            return
        self.texts[ident] = size
        own, claim = _atlas_read(text, self.notes)
        short = bool(claim) and any(len(re.sub('[^0-9]', '', c)) <= 2 for c in claim['codes'])
        if (ATLAS_CLAUSE.search(str(text or '')) and (claim is None or short)) or (ATLAS_IDENT.search(str(text or '')) and not own):
            self.cued.add(ident)
        else:
            self.cued.discard(ident)
        own = own | self.read_own.get(ident, set())
        claim = _atlas_merge(claim, self.read_claims.get(ident))
        # A listing's own part or catalogue code, written in brackets after the word "part", is what notes elsewhere name.
        parts = {re.sub('[^a-z0-9]', '', h.lower()) for h in re.findall('[\\[(]\\s*part\\s+([A-Z][A-Za-z]{0,7}[- ]?\\d{2,6}[A-Za-z]{0,3})\\s*[\\])]', str(text or ''))}
        if parts:
            self.parts[ident] = parts
        if claim is not None:
            self.claims[ident] = claim
        if own:
            self.own[ident] = own
            self.rejoin()

    def learn_read(self, ident: str, own: list[Any], fits: list[Any], not_: list[Any]) -> None:
        """Add what a model read in one listing: its own names, and the codes or families it fits or does not fit."""

        def parse(values: list[Any]) -> list[tuple[str, Any]]:
            out: list[tuple[str, Any]] = []
            for value in values:
                text = str(value or '').strip()
                found = _atlas_tokens(text, True)
                if not found and re.search('\\d', text) and len(text) <= 30:
                    found = [('code', re.sub('[^a-z0-9]', '', text.lower()))]
                out += found
            return out
        names = {v for k, v in parse(own) if k == 'code'}
        claim = _atlas_claim()
        for kind, value in parse(fits):
            claim['only_not'] = False
            (claim['codes'].add(value) if kind == 'code' else claim['families'].append(value))
        for kind, value in parse(not_):
            (claim['not'].add(value) if kind == 'code' else claim['not_families'].append(value))
        if names:
            self.read_own[ident] = names
            self.own[ident] = self.own.get(ident, set()) | names
            self.rejoin()
        if claim['codes'] or claim['families'] or claim['not'] or claim['not_families']:
            self.read_claims[ident] = claim
            self.claims[ident] = _atlas_merge(self.claims.get(ident), claim)
        self.cued.discard(ident)

    def rejoin(self) -> None:
        """Aliases from every listing's names; a code that several listings carry with different partners names no one device."""
        carriers: dict[str, set[frozenset[str]]] = {}
        for codes in self.own.values():
            for code in codes:
                carriers.setdefault(code, set()).add(frozenset(codes))
        self.parent = {}
        for codes in self.own.values():
            solid = sorted(c for c in codes if len(carriers.get(c, ())) <= 2)
            for code in solid:
                self._root(code)
            for code in solid[1:]:
                self._join(solid[0], code)

    def aliases(self, codes: set[str]) -> set[str]:
        roots = {self._root(c) for c in codes}
        return {c for c in self.parent if self._root(c) in roots} | set(codes)

    def known(self) -> set[str]:
        return {c for codes in self.own.values() for c in codes}

    def target(self, ident: str) -> set[str]:
        return self.aliases(self.own.get(ident, set())) if self.own.get(ident) else set()

    def claim_of(self, ident: str) -> dict[str, Any] | None:
        claim = self.claims.get(ident)
        for code in self.parts.get(ident, ()):
            claim = _atlas_merge(claim, self.notes.get(code))
        return claim

    def verdict(self, ident: str, target: set[str]) -> bool | None:
        """True when the listing's own words say it fits a device with these codes, False when they say it does not
        (or name only other devices of this store), None when it says nothing about fit."""
        claim = self.claim_of(ident)
        if not claim or not target:
            return None
        codes = self.aliases(set(target))
        if codes & claim['not'] or any(f.match(c) for f in claim['not_families'] for c in codes):
            return False
        if codes & claim['codes'] or any(f.match(c) for f in claim['families'] for c in codes):
            return True
        if claim['only_not']:
            return None
        # A named code that is the start of one of the device's codes may be a family written another way: unknown.
        if any(c != n and c.startswith(n) for n in claim['codes'] for c in codes):
            return None
        # The claim names devices but not this one: it decides only when it speaks of the same kind of codes.
        prefixes = {re.match('[a-z]+', c).group(0) for c in codes if re.match('[a-z]+', c)}
        named = claim['codes'] | claim['not']
        universe = self.known()
        if named & universe or any(re.match('[a-z]+', c) and re.match('[a-z]+', c).group(0) in prefixes for c in named):
            return False
        if any(f.match(c) for f in claim['families'] + claim['not_families'] for c in universe):
            return False
        if any(f.pattern.startswith('^' + re.escape(p)) for f in claim['families'] for p in prefixes):
            return False
        # Same shape of code (as many digits as one of the device's names): a list of other devices of this store.
        shapes = {len(re.sub('[^0-9]', '', c)) for c in codes}
        if any(len(re.sub('[^0-9]', '', c)) in shapes for c in named):
            return False
        return None

    def describe(self, codes: set[str]) -> str:
        """A device's codes as the store writes them, joined into one name."""
        return ' = '.join(sorted(self.aliases(codes)))


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

def _same_property(a: str, b: str) -> bool:
    one, two = (_stem(a), _stem(b))
    return _names(one, two) or _names(two, one)

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

class Keystone:

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
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.born = time.monotonic()
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
        self.alias_cache: dict[int, tuple[Any, dict[str, set[str]]]] = {}
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.atlas = FitAtlas()            # compatibility read by code from every record seen
        self.fit_codes: dict[int, set[str]] = {}   # item -> codes of a device the shopper already has
        self.atlas_said = ''
        self.no_styles = {'puzzled': 0, 'plain': 0}   # how this shopper says "no": puzzled replies or plain shrugs
        self.planned = False                         # the last-resort cart plan was tried
        self.shelf_map: dict[str, int] = {}           # category label -> item, learned when names share no words
        self.atlas_reads = 0                # model reads of listings the parser could not read
        self.atlas_read: set[str] = set()   # listings already given to such a read
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.proof_holds = 0
        self.proof_learned: dict[str, dict[str, set[Any]]] = {}
        self.proof_asks = 0
        self.proof_pressed = False
        self.whole_said: set[str] = set()
        self.proof_shown = ''
        self.single_file = False
        self.capped = False
        self.ever_carted: set[str] = set()
        self.cap_said: set[str] = set()
        self.page_said: set[str] = set()
        self.store_rules: list[str] = []
        self.page_sentences: list[str] = []
        self.committed = False    # the best-known cart was already tried instead of a decline
        self.short_filled = False  # a cart short of an item was filled before a forced order
        self.swept = False

    def loop_cap(self) -> int:
        """Model turns follow the task's own step limit; the spend and the clock still bound them."""
        return max(LOOP_CALLS, self.max_steps)

    def call_cap(self) -> int:
        return max(ALL_CALLS, self.loop_cap() + ALL_CALLS - LOOP_CALLS)

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.born

    def late(self) -> bool:
        return self.elapsed() >= CLOCK_SOFT

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

    def step_plan(self) -> str:
        """With several items, steps run out on one of them: show which item has been left behind."""
        if len(self.items) < 2 or self.cart_touched or self.step * 3 < self.max_steps:
            return ''
        reads = [sum(1 for i in self.full if self.proof_owner(i) == n) for n in range(1, len(self.items) + 1)]
        behind = [n for n, count in enumerate(reads, 1) if count <= 1]
        if not behind:
            return ''
        names = ', '.join(self.items[n - 1].name for n in behind)
        return f' Listings read per item: {reads}. Read {names} next, about {max(1, (self.left() - self.tail() - len(self.items)) // len(self.items))} steps each: every item needs a pick before the steps run out.'

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap() or self.elapsed() >= CLOCK_HARD:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or self.late() or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
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
        room = CLOCK_LLM_END - self.elapsed()
        _llm.timeout = max(5, min(50, int(room)))
        result = _llm.post_verbose('/inference/chat/completions', body)
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
            if self.broke or self.no_credit or self.calls >= self.call_cap():
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
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for entry in _items(brief.get('fit_codes')):
            if isinstance(entry, dict):
                number = became.get(_count(entry.get('item')) or 0)
                text = str(entry.get('code') or '').strip()
                codes = {v for k, v in _atlas_tokens(text) if k == 'code'} or ({re.sub('[^a-z0-9]', '', text.lower())} if re.search('\\d', text) else set())
                if number and codes:
                    self.fit_codes.setdefault(number, set()).update(codes)
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
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0), 'reply': entry.get('reply')})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages'] and p.get('reply') is not False]
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
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
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
            when = {'choose': 'before choosing', 'order': 'at the order (Keystone does it when you check out)', 'end': 'at an ending without a purchase (Keystone does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Keystone] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        text += self.step_plan()
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        if (self.bound or self.fit_codes) and self.buys == 0:
            self.atlas_assist()
        plan = self.fit_plan() if (self.bound or self.fit_codes) and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        wide = self.whole_labels()
        if wide:
            text += '\n' + wide
        table = self.proof_table()
        if table:
            text += '\n' + table
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
            return f"\n[Keystone] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
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
        rest = f"\n[Keystone] {count} listings matched and Keystone has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        if value is None and texts:
            rest = self.cap_note(name, call, seen)
        return '\n[Keystone read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

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
        return f'\n[Keystone] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Keystone] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def rule_sentences(self) -> list[str]:
        """Sentences seen on pages that name the property of a requirement pointing at another listing, each returned once."""
        props = [_roots(need.partition(':')[0]) for _, need in self.refs]
        out: list[str] = []
        for sentence in self.page_sentences:
            if sentence not in self.page_said and any((prop and prop <= _roots(sentence) for prop in props)):
                self.page_said.add(sentence)
                self.store_rules.append(sentence)
                out.append(sentence)
        return out

    def page_notes(self, observation: Any) -> str:
        """What a page says that is easy to pass over: a sentence naming the property of a requirement that points at another listing, and a multiplied size worked out. Each is said once."""
        texts: list[str] = []

        def walk(node: Any, depth: int=0) -> None:
            if isinstance(node, str):
                texts.append(node)
            elif depth < 4 and isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif depth < 4 and isinstance(node, list):
                for value in node[:30]:
                    walk(value, depth + 1)
        walk(observation)
        notes: list[str] = []
        for text in texts:
            for sentence in re.split('(?<=[.!?])\\s+', text):
                if 4 <= len(sentence.split()) <= 60 and sentence not in self.page_sentences and len(self.page_sentences) < 400:
                    self.page_sentences.append(sentence)
        notes += [f'store rule: {sentence}' for sentence in self.rule_sentences()]
        for text in texts:
            # A small whole count times one size, followed by any short unit word. Chained figures are dimensions, and
            # the note says "possible" because two figures can also be a shape: the model decides from the page.
            for found in re.finditer('(?<![\\d.])(\\d{1,2})\\s*[x\u00d7]\\s*(\\d+(?:\\.\\d+)?)\\s*([a-z]{1,5})\\b', text, re.I):
                count, size, unit = found.group(1), found.group(2), found.group(3).lower()
                if unit == 'x' or re.search('\\d\\s*[x\u00d7]\\s*$', text[:found.start()]):
                    continue
                total = float(count) * float(size)
                note = f'possible total: {count} x {size} {unit} = {total:g} {unit}'
                if note not in self.page_said:
                    self.page_said.add(note)
                    notes.append(note)
        return '\n[Keystone] ' + ' | '.join(notes[:6]) if notes else ''

    def cap_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Say so when a search cannot reach its reported total: the tool's page limit is below it, or a later page ended early. A first page without a next value proves nothing, since some stores number their pages."""
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(param)
        top = spec.get('maximum') if isinstance(spec, dict) else None
        given = args.get(param)
        given = given if isinstance(given, int) and not isinstance(given, bool) else 0
        if isinstance(top, int) and not isinstance(top, bool) and 0 < top < shown:
            reach = top * shown
            how = f'{param} stops at {top}'
        elif value is None and given > 0:
            # A value below a page's length is a page number; anything else is a row offset.
            reach = given * shown if given < shown else given + shown
            how = 'paging ended there'
        else:
            return ''
        mark = f'{name}#{count}'
        if reach >= count or mark in self.cap_said:
            return ''
        self.cap_said.add(mark)
        self.capped = True
        self.trace.append({'kind': 'capped', 'tool': name, 'count': count, 'reach': reach})
        return f'\n[Keystone] This search can show at most {reach} of its {count} matches ({how}). Only other search words can reach the other {count - reach}.'

    def carve_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """How to run a word search: as the agent last did, else from the tool's schema."""
        plan = self.search_template()
        if plan:
            return plan
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        if not tool or not named:
            return None
        keep: dict[str, Any] = {}
        for key, spec in props.items():
            options = spec.get('enum') if isinstance(spec, dict) else None
            if isinstance(options, list) and options:
                word = next((o for o in options if re.search('word|text|key|search|query', str(o), re.I)), None)
                if word is not None:
                    keep[key] = word
                elif key in (schema.get('required') or []):
                    keep[key] = options[0]
        return (tool, keep, named[0])

    def carve_words(self, number: int) -> list[str]:
        """Search words most likely to reach listings not yet seen: the item's names one by one, then what cheap titles of its kind have in common."""
        item = self.items[number - 1]
        cat_key = self.category_key()

        def single(phrase: str) -> str:
            words = phrase.split()
            last = words[-1] if words else ''
            last = last[:-3] + 'f' if last.endswith('lves') else last[:-3] + 'y' if last.endswith('ies') else last[:-2] if re.search('(?:ch|sh|ss|x)es$', last) else last[:-1] if len(last) > 3 and last.endswith('s') and not last.endswith('ss') else last
            return ' '.join(words[:-1] + [last])
        names: list[str] = []
        plurals: list[str] = []
        for phrase in re.split(r'\s+(?:and|or)\s+|[&,/;]', _words(item.name) + ',' + _words(item.shelf or '')):
            phrase = phrase.strip()
            if phrase:
                names.append(single(phrase))
                plurals.append(phrase)
        rows = [(i, r) for i, r in self.rows.items() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        rows.sort(key=lambda p: self.price_of(p[0]) if self.price_of(p[0]) is not None else float('inf'))
        cheap = rows[:max(8, len(rows) // 3)]
        counts: dict[str, int] = {}
        for _, row in cheap:
            title = next((str(v) for k, v in row.items() if isinstance(v, str) and re.search('title|name|headline', k, re.I)), '')
            others = ' '.join((str(v).lower() for k, v in row.items() if isinstance(v, str) and len(str(v)) < 40 and not re.search('title|name|headline|description', k, re.I)))
            for word in set(re.findall('[a-z]{4,}', title.lower())):
                if word not in others and word not in PROOF_LINK:
                    counts[word] = counts.get(word, 0) + 1
        tokens = sorted(counts, key=lambda w: (-counts[w], w))
        said = {_words(str(v)) for args in self.searched for v in args.values() if isinstance(v, str)}
        out: list[str] = []
        for word in names + tokens + plurals:
            if word not in out and word not in said:
                out.append(word)
        return out

    def carve_search(self, numbers: list[int], spare: int, leads: dict[int, list[str]] | None=None) -> list[str]:
        """Run word searches for these items in turn, leaving spare steps, until they stop finding new rows."""
        plan = self.carve_plan()
        if not plan:
            return []
        tool, base, field = plan
        queues = {n: list(dict.fromkeys((leads or {}).get(n, []) + self.carve_words(n))) for n in numbers}
        ran: list[str] = []
        quiet = {n: 0 for n in numbers}
        while len(ran) < CARVE_SEARCHES and self.left() - self.tail() > spare and not self.finished and not self.late():
            live = [n for n in numbers if queues[n] and quiet[n] < 3]
            if not live:
                break
            for number in live:
                if len(ran) >= CARVE_SEARCHES or self.left() - self.tail() <= spare:
                    break
                word = queues[number].pop(0)
                faults: list[str] = []
                call = _shape({**base, field: word}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                known = set(self.rows)
                reply = self.act(tool, call)
                if reply is None or self.finished:
                    return ran
                self.observe(tool, reply.get('observation'))
                ran.append(word)
                quiet[number] = 0 if set(self.rows) - known else quiet[number] + 1
        return ran

    def carve_sweep(self) -> str:
        """The pick is the cheapest and a category search shows only part of the set: search under other words, then read the cheapest rows that undercut the cart."""
        cat_key = self.category_key()
        lines: dict[int, float] = {}
        for ident in self.cart_ids():
            label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
            owner = self.owner_of(label)
            price = self.price_of(ident)
            if owner and price is not None:
                lines[owner] = max(lines.get(owner, 0.0), price)
        if not lines:
            return ''
        before = set(self.rows)
        ran = self.carve_search(list(lines), 8)
        fresh = set(self.rows) - before
        self.trace.append({'kind': 'carve', 'words': ran, 'new_rows': len(fresh)})
        cart = self.cart_ids()
        under: list[str] = []
        for number, price in lines.items():
            pool = [i for i, r in self.rows.items() if i not in cart and i not in self.full and self.buyable(i) and self.owns(self.items[number - 1], str(r.get(cat_key) or '')) and (self.price_of(i) if self.price_of(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            under += pool[:CHEAPER_OPENS]
        read = self.fill_facts(under[:CHEAPER_OPENS * len(lines)], self.tail() + 4)
        if not ran:
            return ''
        said = f"Keystone searched under other words ({', '.join(ran)}) and saw {len(fresh)} new rows. "
        if read:
            said += 'These undercut the cart; their attributes:\n' + '\n'.join(read) + '\nSwap in the cheapest that meets every requirement. '
        else:
            said += 'None undercuts the cart unread. '
        return said

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        carved = self.carve_sweep()
        if carved:
            return f"Not run yet. The pick is the cheapest, and searches here do not show every matching listing. {carved}Then check out." + (f" Cart lines: {'; '.join(lines)}." if lines else '')
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

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
            texts.append(f'{ident}: ' + self.show(seen)[:1500] + self.page_notes(seen))
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
        """Belief interview. Every (item, property) carries the chance that the shopper holds an unwritten wish on it.
        Each message is built to reveal the most: the item and the ordered properties with the largest sum of
        chance x how often a shopper answers that place in a list. A reply moves the chances: a revealed wish settles
        the property it answered, a reply without one lowers each named property by how likely it was to be heard."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        beliefs = self.belief_priors()
        found = {n: 0 for n in range(1, len(self.items) + 1)}
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        history: set[tuple[int, tuple[str, ...]]] = set()
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (not self.late()):
            plan = self.belief_message(beliefs, history)
            if not plan:
                break
            number, labels, gain = plan
            batch = self.send([(number, labels[0], self.belief_text(number, labels))])
            if not batch:
                break
            sent += batch
            history.add((number, tuple(labels)))
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if _items(heard.get('needs')) and self.judges():
                heard = self.hear(batch, (HEAVY, LIGHT)) or heard
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] += 1
            crowded = 1 in [_count(x) for x in _items(heard.get('crowded'))]
            puzzled = 1 in [_count(x) for x in _items(heard.get('puzzled'))]
            self.trace.append({'kind': 'belief_ask', 'item': number, 'labels': labels, 'gain': round(gain, 3), 'fresh': fresh, 'crowded': crowded, 'puzzled': puzzled})
            if crowded and not fresh:
                # This shopper takes one thing at a time: nothing was tested, so the chances stay as they were.
                self.single_file = True
                continue
            for entry in _items(heard.get('free')):
                if isinstance(entry, dict) and entry.get('property'):
                    n = min(max(1, _count(entry.get('item')) or number), len(self.items))
                    for key in [k for k in beliefs if k[0] == n and self.belief_same(n, k[1], str(entry['property']))]:
                        beliefs[key] = 0.0
            if fresh:
                self.belief_settle(number, labels, heard, beliefs)
            else:
                self.no_styles['puzzled' if puzzled else 'plain'] += 1
                self.belief_clear(number, labels, beliefs, puzzled and self.no_styles['plain'] > 0)
        return self.interview_report(sent, told, found)

    def heard_at(self, place: int) -> float:
        table = HEARD_ONE if len(self.items) == 1 else HEARD_MANY
        return table[min(place, len(table) - 1)]

    def belief_priors(self) -> dict[tuple[int, str], float]:
        """Starting chances. A model ranks and rates the properties; the store's attribute keys add the properties it left
        out. Weight falls with rank, rises with the rating and with how many of the item's listings carry the key; each
        item's weights are scaled to the number of unwritten wishes such an item usually holds."""
        table = self.key_table()
        count = min(16, max(4, self.msgs_free() * LIST_CAP))
        fixed = [f'item {n}: {item.name} - fixed by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(self.items, 1)]
        listing, last = ([], None)
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one]' if self.catch_all(name) else ''))
        prompt = PLAN.replace('COUNT', str(count)) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(fixed)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to check these with the shopper):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no opened listings yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        weights: dict[tuple[int, str], float] = {}
        ranks: dict[int, int] = {}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5 or self.belief_fixed(number, label):
                continue
            if any(k[0] == number and self.belief_same(number, k[1], label) for k in weights):
                continue
            rank = ranks.get(number, 0)
            ranks[number] = rank + 1
            rate = {'hinted': 2.5, 'likely': 1.2, 'possible': 0.7}.get(str(pick.get('rate') or '').strip().lower(), 1.0)
            if str(pick.get('hint') or '').strip() and rate < 2.5:
                rate = max(rate, 1.8)
            weights[(number, self.belief_spelled(number, label, table))] = rate / (1.0 + 0.6 * rank)
        for number in range(1, len(self.items) + 1):
            read = max(1, self.records_of_item(number))
            share = {key: carried / read for lab, key, carried, _ in table if self.owner_of(lab) == number}
            for j, key in enumerate(self.lane_keys(number, table)):
                if self.belief_fixed(number, key):
                    continue
                mine = next((k for k in weights if k[0] == number and self.belief_same(number, k[1], key)), None)
                lift = 0.6 + 0.8 * min(1.0, share.get(key, 0.0))
                if mine:
                    weights[mine] *= lift
                else:
                    weights[(number, key)] = (0.25 / (1.0 + 0.3 * j)) * lift * (0.3 if self.catch_all(key) else 1.0)
        beliefs: dict[tuple[int, str], float] = {}
        mass = NEEDS_ONE if len(self.items) == 1 else NEEDS_EACH
        for number in range(1, len(self.items) + 1):
            mine = {k: w for k, w in weights.items() if k[0] == number}
            total = sum(mine.values())
            for key, w in mine.items():
                beliefs[key] = min(BELIEF_CAP, mass * w / total) if total else 0.0
        self.trace.append({'kind': 'belief_priors', 'top': {n: sorted(((round(p, 2), k[1]) for k, p in beliefs.items() if k[0] == n), reverse=True)[:6] for n in range(1, len(self.items) + 1)}})
        return beliefs

    def belief_spelled(self, number: int, label: str, table: list[tuple[str, str, int, list[str]]]) -> str:
        """The store's own key for a label the model wrote in other spelling (plural, British -our)."""
        plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
        return next((key for lab, key, _, _ in table if self.owner_of(lab) == number and plain(key) == plain(label)), label)

    def belief_fixed(self, number: int, label: str) -> bool:
        """The request or an earlier answer already fixes this property, or the key is not a property at all."""
        item = self.items[number - 1]
        if re.search(NOT_PROPERTY, _words(label)):
            return True
        props = [m.split(':', 1)[0] for m in item.musts + item.told if m not in item.dropped]
        return any(_names(_stem(prop), _stem(label)) for prop in props)

    def belief_same(self, number: int, one: str, two: str) -> bool:
        a, b = (_stem(one), _stem(two))
        return _words(one) == _words(two) or _names(a, b) or _names(b, a) or _words(two) in self.key_aliases(number).get(_words(one), set())

    def belief_message(self, beliefs: dict[tuple[int, str], float], history: set[tuple[int, tuple[str, ...]]]) -> tuple[int, list[str], float] | None:
        """The next message: for each item, its most likely open properties in order (one when the shopper takes one
        thing at a time); the item whose list promises the most is asked. A list already sent unchanged is shortened
        from the front, since the same message draws the same reply."""
        cap = 1 if self.single_file else LIST_CAP
        best = None
        for number in range(1, len(self.items) + 1):
            ranked = sorted(((p, label) for (n, label), p in beliefs.items() if n == number and p > 0.005), key=lambda x: -x[0])
            chosen: list[tuple[float, str]] = []
            for p, label in ranked:
                if len(chosen) >= cap:
                    break
                if self.belief_fixed(number, label) or any(self.belief_same(number, label, other) for _, other in chosen):
                    continue
                chosen.append((p, label))
            while chosen and (number, tuple(l for _, l in chosen)) in history:
                chosen = chosen[1:]
            if not chosen:
                continue
            gain = sum(p * self.heard_at(i) for i, (p, _) in enumerate(chosen))
            if best is None or gain > best[2]:
                best = (number, [l for _, l in chosen], gain)
        if best is None or best[2] < GAIN_FLOOR:
            return None
        return best

    def belief_text(self, number: int, labels: list[str]) -> str:
        name = self.items[number - 1].name
        if len(labels) == 1:
            return self.question(number, labels[0], self.samples(number, labels[0]) or None)
        names = ', '.join(labels[:-1]) + f' or {labels[-1]}'
        return f'For the {name}, which {names} would you like?'

    def belief_clear(self, number: int, labels: list[str], beliefs: dict[tuple[int, str], float], weak: bool) -> None:
        """No wish came back: each named property keeps only the chance that its wish went unheard at its place
        (Bayes: p(1-h) / (1-ph)). A puzzled reply from a shopper who also gives plain shrugs counts half."""
        for place, label in enumerate(labels):
            key = (number, label)
            p = beliefs.get(key, 0.0)
            h = self.heard_at(place) * (0.5 if weak else 1.0)
            beliefs[key] = p * (1 - h) / max(1e-06, 1 - p * h)

    def belief_settle(self, number: int, labels: list[str], heard: dict[str, Any], beliefs: dict[tuple[int, str], float]) -> None:
        """A wish came back. The property it answers is settled: the one the reading says was asked, else the named one
        sharing the wish's words, else the one whose seen values hold the wish's value. Only one wish comes back per
        message, so the other names keep most of their chance."""
        values: dict[str, set[str]] = {}
        for _, key, _, seen in self.key_table():
            values.setdefault(key, set()).update((w for v in seen for w in _roots(v)))
        settled: set[str] = set()
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or ':' not in str(entry.get('need') or ''):
                continue
            owner = min(max(1, _count(entry.get('item')) or number), len(self.items))
            prop, _, value = str(entry['need']).partition(':')
            said = str(entry.get('asked') or '').strip()
            pick = None
            if owner == number:
                pick = next((l for l in labels if said and self.belief_same(number, l, said)), None)
                pick = pick or next((l for l in labels if _stem(l) & _stem(prop)), None)
                pick = pick or next((l for l in labels if _roots(value) and _roots(value) <= values.get(_words(l), set())), None)
            for key in [k for k in beliefs if k[0] == owner and (k[1] == pick or self.belief_same(owner, k[1], prop))]:
                beliefs[key] = 0.0
                settled.add(key[1])
            beliefs.setdefault((owner, prop.strip()), 0.0)
        if not settled and labels:
            beliefs[(number, labels[0])] = beliefs.get((number, labels[0]), 0.0) * 0.3
        for label in labels:
            if label not in settled and beliefs.get((number, label), 0.0) > 0:
                beliefs[(number, label)] *= 0.8
        self.trace.append({'kind': 'belief_settled', 'item': number, 'done': sorted(settled)})

    def key_aliases(self, number: int) -> dict[str, set[str]]:
        """Keys that state one property: the same value on nearly every opened listing carrying both (cached until
        another listing is opened)."""
        mark = (number, len(self.full))
        if self.alias_cache.get(number, (None, None))[0] == mark:
            return self.alias_cache[number][1]
        out = self.key_aliases_now(number)
        self.alias_cache[number] = (mark, out)
        return out

    def key_aliases_now(self, number: int) -> dict[str, set[str]]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        both: dict[tuple[str, str], int] = {}
        same: dict[tuple[str, str], int] = {}
        words: dict[tuple[str, str], int] = {}
        for record in self.full.values():
            if not cat_key or not self.owns(item, str(record.get(cat_key) or '')):
                continue
            block = {_words(k): _words(str(v)) for k, v in (_fact_block(record) or {}).items() if str(v).strip()}
            keys = sorted(block)
            for i, one in enumerate(keys):
                for other in keys[i + 1:]:
                    pair = (one, other)
                    both[pair] = both.get(pair, 0) + 1
                    if block[one] == block[other]:
                        same[pair] = same.get(pair, 0) + 1
                        if re.search('[a-z]{3}', block[one]):
                            words[pair] = words.get(pair, 0) + 1
        out: dict[str, set[str]] = {}
        for pair, count in both.items():
            # A bare number can match by chance ("15 cm" as both depth and width), so it takes more records.
            if same.get(pair, 0) >= 0.8 * count and (count >= 3 or (count >= 2 and words.get(pair, 0) >= 2)):
                out.setdefault(pair[0], set()).add(pair[1])
                out.setdefault(pair[1], set()).add(pair[0])
        return out

    def interview_report(self, sent: list[tuple[int, str, str]], told: list[str], found: dict[int, int]) -> str:
        found_items = [n for n in found if found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'asked: {q}\nreply: {a}' for _, q, a in sent]
        summary = f"[interview] Keystone sent {len(sent)} message(s) to the shopper (one step each):\n" + '\n'.join(lines)
        summary += '\nWishes the shopper revealed (now firm requirements): ' + ('; '.join(told) if told else 'none')
        summary += '. Every pick must meet them; search again with the new values where the current candidates do not show them.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These wishes take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full and every listing that page sends you to; the last page gives the value, and the order counts only once all of them are opened.'
            self.rule_sentences()
            if self.store_rules:
                summary += ' Store pages state: ' + ' '.join(self.store_rules[:3]) + ' Where that links this value to another, apply the rule.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
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
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
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
        return "[Keystone searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def catch_all(self, key: str) -> bool:
        """A key whose values mostly list three or more things at once."""
        name = _words(key)
        values = [str(v) for record in self.full.values() for k, v in (_fact_block(record) or {}).items() if _words(k) == name]
        return bool(values) and sum((1 for v in values if len(re.split('[,;/|]', v)) >= 3)) * 2 >= len(values)

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
        if _words(label) in self.shelf_map:
            return self.shelf_map[_words(label)]
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
                live = [m for m in item.musts + item.told if m not in item.dropped]
                hit = [m for m in live if _words(m.split(':')[0]) == gone] or [m for m in live if _same_property(m.split(':')[0], gone)]
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
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about {("item " + str(n)) if n else "the items it names"}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

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
                return (False, ('Before going on, Keystone did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Keystone: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
        return (True, 'Before going on, Keystone did the steps the request asks for:\n' + '\n'.join(said) if said else '')

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
                hit = ([m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])] or [m for m in live if _same_property(m.split(':')[0], target.split(':')[0])])[:1]
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
        return "lines by Keystone's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def record_text(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        parts = [str(v) for v in record.values() if isinstance(v, str)]
        for block in record.values():
            if isinstance(block, dict):
                parts += [f'{_words(k)}: {v}' for k, v in block.items() if isinstance(v, (str, int, float, list))]
            elif isinstance(block, list):
                parts += [str(v) for v in block if isinstance(v, str)]
        return '. '.join(parts)

    def atlas_sync(self) -> FitAtlas:
        """Feed every record seen into the fit atlas (a record is re-read only when it grew)."""
        for ident in self.seen_order:
            self.atlas.learn(ident, self.record_text(ident))
        return self.atlas

    def atlas_gaps(self) -> list[str]:
        """Listings whose fit words the code could not read, on items where fit decides the order: a part kind where
        most seen listings carry no readable claim, a device kind where most carry no readable model name, and any
        listing of those kinds that uses fit or model wording without a code the parser could take."""
        atlas = self.atlas_sync()
        parts = sorted({p for p, _ in self.bound} | set(self.fit_codes))
        mains = sorted({m for _, m in self.bound})
        gaps: list[str] = []
        for number in parts + mains:
            ids = [i for i in self.item_ids(number) if not self.sold_out(i) and i not in self.atlas_read]
            every = [i for i in self.item_ids(number) if not self.sold_out(i)]
            if number in parts:
                missing = [i for i in ids if not atlas.claim_of(i)]
                thin = len(every) >= 3 and sum(1 for i in every if atlas.claim_of(i)) * 2 < len(every)
            else:
                missing = [i for i in ids if not atlas.own.get(i)]
                thin = len(every) >= 2 and sum(1 for i in every if atlas.own.get(i)) * 2 < len(every)
            gaps += [i for i in (missing if thin else [i for i in missing if i in atlas.cued]) if i not in gaps]
        return gaps

    def atlas_assist(self, now: bool=False) -> int:
        """When the code could not read fit words that decide the order, one light model reads those listings and the
        atlas judges what it returns by the same rules. Few reads per episode, none late or without budget."""
        if not (self.bound or self.fit_codes) or self.atlas_reads >= ATLAS_READS or self.late() or self.finished:
            return 0
        if self.no_credit or self.broke or self.cost > SPEND_HARD:
            return 0
        gaps = self.atlas_gaps()
        if not gaps or (len(gaps) < 3 and not now):
            return 0
        batch = gaps[:ATLAS_BATCH]
        self.atlas_reads += 1
        self.atlas_read.update(batch)
        lines = [f'{n}. ' + self.record_text(i)[:ATLAS_CHARS] for n, i in enumerate(batch, 1)]
        got = self.ask_json(ATLAS_PROMPT + '\n\nLISTINGS:\n' + '\n'.join(lines), [LIGHT], 2500) or {}
        added = 0
        for entry in _items(got.get('listings')):
            if not isinstance(entry, dict):
                continue
            n = _count(entry.get('n'))
            if not n or not 1 <= n <= len(batch):
                continue
            own, fits, not_ = (_items(entry.get('is')), _items(entry.get('fits')), _items(entry.get('not')))
            if own or fits or not_:
                self.atlas.learn_read(batch[n - 1], own, fits, not_)
                added += 1
        self.trace.append({'kind': 'atlas_read', 'asked': len(batch), 'added': added})
        return added

    def fit_targets(self, number: int) -> list[tuple[set[str], str]]:
        """What a requested item must fit: a device the shopper names, and the cart's line of the item it is bound to."""
        atlas = self.atlas_sync()
        out: list[tuple[set[str], str]] = []
        if self.fit_codes.get(number):
            codes = atlas.aliases(set(self.fit_codes[number]))
            out.append((codes, 'the shopper\'s ' + atlas.describe(codes)))
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        for part, main in self.bound:
            if part != number:
                continue
            devices = [i for i in cart if i in self.item_ids(main) and atlas.own.get(i)]
            if len(devices) == 1:
                out.append((atlas.target(devices[0]), f'{devices[0]} ({atlas.describe(atlas.own[devices[0]])})'))
        return out

    def fit_verdict(self, ident: str, targets: list[tuple[set[str], str]]) -> bool | None:
        verdicts = [self.atlas.verdict(ident, codes) for codes, _ in targets]
        if False in verdicts:
            return False
        if verdicts and all(v is True for v in verdicts):
            return True
        return None

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        """Cart lines whose own words say they do not fit the device they must fit, with seen listings that do."""
        numbers = sorted({p for p, _ in self.bound} | set(self.fit_codes))
        if not numbers:
            return []
        self.atlas_assist(now=True)
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        faults = []
        for number in numbers:
            targets = self.fit_targets(number)
            if not targets:
                continue
            for line in [i for i in cart if i in self.item_ids(number)]:
                if self.fit_verdict(line, targets) is not False:
                    continue
                good = [i for i in self.item_ids(number) if i not in cart and not self.sold_out(i) and self.buyable(i) and self.fit_verdict(i, targets) is True]
                good.sort(key=lambda i: self.proof_price(i) if self.proof_price(i) is not None else float('inf'))
                want = ' and '.join(name for _, name in targets)
                faults.append(f"{line} ({self.items[number - 1].name}) does not fit {want} by its own compatibility words" + (f"; seen listings of that kind that fit: {', '.join(good[:4])}" if good else '; no seen listing of that kind fits it yet: search or browse that kind by product words (model codes are not searchable) and Keystone marks the rows that fit' + (', or switch the device' if any(p == number for p, _ in self.bound) else '')))
        return faults

    def fit_plan(self, short: bool=False) -> str:
        """The atlas as the buyer should see it: which device candidates the seen parts fit, and which parts fit a named device."""
        numbers = sorted({p for p, _ in self.bound} | set(self.fit_codes))
        if not numbers:
            return ''
        atlas = self.atlas_sync()

        def price(i: str) -> str:
            value = self.proof_price(i)
            return f' {value:g}' if isinstance(value, (int, float)) else ''

        def fitting(number: int, codes: set[str]) -> tuple[list[str], int]:
            ids = [i for i in self.item_ids(number) if not self.sold_out(i) and self.buyable(i)]
            good = [i for i in ids if atlas.verdict(i, codes) is True]
            good.sort(key=lambda i: self.proof_price(i) if self.proof_price(i) is not None else float('inf'))
            return good, sum(1 for i in ids if atlas.verdict(i, codes) is False)
        lines = []
        for number in sorted(self.fit_codes):
            codes = atlas.aliases(set(self.fit_codes[number]))
            good, bad = fitting(number, codes)
            lines.append(f"item {number} ({self.items[number - 1].name}) must fit the shopper's {atlas.describe(codes)}: seen rows that fit: " + (', '.join(i + price(i) for i in good[:6]) or 'none yet') + f'; seen rows that do not: {bad}')
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            devices = [i for i in self.item_ids(main) if atlas.own.get(i) and not self.sold_out(i) and self.buyable(i)]
            ranked = []
            for d in devices:
                codes = atlas.target(d)
                per = {p: fitting(p, codes)[0] for p in parts}
                ranked.append((-sum(1 for v in per.values() if v), -sum(len(v) for v in per.values()), d, per))
            ranked.sort(key=lambda r: r[:2])
            shown = []
            for _, _, d, per in ranked[:4 if not short else 2]:
                shown.append(f"{d}{price(d)} ({atlas.describe(atlas.own[d])}): " + '; '.join(f"{self.items[p - 1].name} " + (', '.join(i + price(i) for i in per[p][:3]) or 'none seen') for p in parts))
            if shown:
                lines.append(f"{self.items[main - 1].name} candidates and the seen parts that fit each: " + ' | '.join(shown))
            elif devices or any(atlas.claims.get(i) for p in parts for i in self.item_ids(p)):
                lines.append(f"no {self.items[main - 1].name} seen yet states its model code: open or list that kind to read the codes")
        if not lines:
            return ''
        text = ' || '.join(lines)
        if short:
            return text[:700]
        return "[fit atlas: Keystone reads model codes, other names a device is sold as, code families (12xx) and exclusions from every listing's own words; a search for a code finds nothing, so search or browse by product words and pick from the rows marked here] " + text[:1400]

    def whole_label(self, number: int, label: str) -> str:
        """A seen category label that covers every word of the item's name when this label does not; '' otherwise."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        wanted = _roots(item.name)
        if item.shelf.strip() or not cat_key or len(wanted) < 2:
            return ''
        have = len(_roots(label) & wanted)
        seen = {str(r.get(cat_key) or '') for r in self.rows.values()}
        better = [l for l in seen if l and wanted <= _roots(l)]
        return min(better, key=len) if better and have < len(wanted) else ''

    def whole_labels(self) -> str:
        """Says once, as soon as it is seen, which category label covers the whole of what was asked."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        said = []
        for number, item in enumerate(self.items, 1):
            for label in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}):
                whole = self.whole_label(number, label)
                if label and whole and whole not in self.whole_said:
                    self.whole_said.add(whole)
                    said.append(f"[category] {whole!r} is the category for the whole of what was asked ({item.name}); {label!r} is a neighbouring one, so buy from {whole!r}: list it with the category search and open its listings.")
        return '\n'.join(said)

    def proof_owner(self, ident: str) -> int:
        """Which requested item a listing belongs to, by its category label; 0 when it belongs to none."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        if not label:
            return 1 if len(self.items) == 1 else 0
        owner = self.owner_of(label)
        if owner or len(self.items) != 1:
            return owner
        seen = {str(r.get(cat_key) or '') for r in self.rows.values()}
        return 0 if any(self.owner_of(other) for other in seen) else 1

    def proof_carriers(self, number: int, prop: str, alts: list[frozenset[str]]) -> set[str]:
        """Keys of a neighbouring property that this store uses to state the wanted value on other listings."""
        words = {w for w in _stem(prop) if len(w) >= 4}
        out: set[str] = set()
        for ident, record in self.full.items():
            if self.proof_owner(ident) != number:
                continue
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if _names(_stem(prop), _stem(key)) or not any(w in name.replace(' ', '') for w in words):
                    continue
                if _proof_holds(alts, value):
                    out.add(name)
        return out

    def proof_against(self, number: int, prop: str, alts: list[frozenset[str]], block: dict[str, Any]) -> str | None:
        """A key that states the wanted value on other listings of this item and something else on this one."""
        for carrier in self.proof_carriers(number, prop, alts):
            against = next((k for k, v in block.items() if _words(k) == carrier and not _proof_holds(alts, v)), None)
            if against:
                return against
        return None

    def proof_need(self, ident: str, number: int, need: str) -> tuple[str, str]:
        """One requirement on one opened listing: (status, the store's words that show it). Only the store's own record is read."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        prop, sep, value = need.partition(':')
        prop, value = prop.strip(), value.strip() if sep else prop.strip()
        if need in [m for _, m in self.refs] or self.described(number, need) or (need not in self.proof_learned and re.search(PROOF_SKIP, value, re.I)):
            return ('skip', '')
        learned = self.proof_learned.get(need)
        if learned:
            keys = {k: v for k, v in block.items() if _words(k) in learned['keys'] or _names(_stem(prop), _stem(k))}
            good = [k for k, v in keys.items() if (_words(k), str(v)) in learned['yes']]
            if good:
                against = None if re.match(PROOF_NOT, value, re.I) else self.proof_against(number, prop, _proof_alts(value), block)
                if against:
                    return ('conflict', f'{good[0]}={str(block[good[0]])[:40]} but {against}={str(block[against])[:40]}')
                return ('pass', f'{good[0]}={str(block[good[0]])[:50]}')
            if keys:
                first = next(iter(keys))
                return ('differs', f'{first}={str(keys[first])[:40]}')
            other = next((k for k, v in block.items() if (_words(k), str(v)) in learned['yes']), None)
            if other:
                return ('weak', f'{other}={str(block[other])[:40]}')
        named = {k: v for k, v in block.items() if _names(_stem(prop), _stem(k))}
        negative = bool(re.match(PROOF_NOT, value, re.I))
        alts = _proof_alts(re.sub(PROOF_NOT, '', value, flags=re.I))
        if not alts:
            return ('skip', '')
        cat_key = self.category_key()
        label = str(record.get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        if not negative and label and _proof_holds(alts, label):
            return ('pass', f'category {label[:40]}')
        if negative:
            hit = next((k for k, v in named.items() if _proof_holds(alts, v)), None)
            if hit:
                return ('differs', f'{hit}={str(named[hit])[:40]}')
            return ('pass', f'{next(iter(named))}={str(next(iter(named.values())))[:40]}') if named else ('weak', 'no attribute names it')
        hit = next((k for k, v in named.items() if _proof_holds(alts, v)), None)
        if hit:
            against = self.proof_against(number, prop, alts, block)
            if against:
                return ('conflict', f'{hit}={str(named[hit])[:40]} but {against}={str(block[against])[:40]}')
            return ('pass', f'{hit}={str(named[hit])[:50]}')
        if named:
            first = next(iter(named))
            return ('differs', f'{first}={str(named[first])[:40]}')
        other = next((k for k, v in block.items() if _proof_holds(alts, v)), None)
        if other:
            return ('weak', f'{other}={str(block[other])[:40]}')
        text = ' '.join(str(v) for k, v in record.items() if isinstance(v, str))
        return ('weak', 'title or description') if _proof_holds(alts, text) else ('unknown', 'no attribute states it')

    def proof_lines(self, number: int) -> list[tuple[str, str]]:
        """Distinct attribute lines on opened listings of an item, those on most listings first."""
        per: dict[tuple[str, str], int] = {}
        carried: dict[str, int] = {}
        for ident, record in self.full.items():
            if self.proof_owner(ident) != number:
                continue
            for key, value in (_fact_block(record) or {}).items():
                if re.search(NOT_PROPERTY, _words(key)):
                    continue
                per[(str(key), str(value))] = per.get((str(key), str(value)), 0) + 1
                carried[_words(key)] = carried.get(_words(key), 0) + 1
        return sorted(per, key=lambda line: (-carried[_words(line[0])], -per[line]))

    def proof_learn(self) -> None:
        """Have the light model read each requirement against the store's own attribute lines, once per new set of lines."""
        for number, need in self.needs_now():
            if self.proof_asks >= PROOF_LEARNS or not self.judges() or self.finished:
                return
            if need in [m for _, m in self.refs]:
                continue
            lines = self.proof_lines(number)[:PROOF_LINES]
            known = self.proof_learned.get(need) or {'keys': set(), 'yes': set(), 'seen': set()}
            if len(lines) < 2 or all((_words(k), v) in known['seen'] for k, v in lines):
                continue
            self.proof_asks += 1
            shown = '\n'.join(f'{i}. {k} = {v[:70]}' for i, (k, v) in enumerate(lines, 1))
            got = self.ask_json(PROOF_READ + f'\n\nREQUIREMENT: {need}\n\nLINES:\n{shown}', [LIGHT, HEAVY], 1500)
            if not isinstance(got, dict) or not isinstance(got.get('yes'), list):
                continue
            yes = {(_words(lines[n - 1][0]), lines[n - 1][1]) for n in (_count(x) for x in got['yes']) if n and 1 <= n <= len(lines)}
            names = {_words(k) for k in _items(got.get('keys')) if isinstance(k, str)}
            self.proof_learned[need] = {'keys': known['keys'] | names, 'yes': known['yes'] | yes, 'seen': known['seen'] | {(_words(k), v) for k, v in lines}}
            self.trace.append({'kind': 'proof_learn', 'need': need[:80], 'keys': sorted(names)[:6], 'yes': len(yes), 'lines': len(lines)})

    def proof_read(self, ident: str, number: int) -> tuple[int, list[str], int]:
        """All requirements of an item on a listing: (lowest score, one note per requirement, how many could be judged)."""
        worst, notes, judged = (4, [], 0)
        for owner, need in self.needs_now():
            if owner != number:
                continue
            status, shown = self.proof_need(ident, number, need)
            if status == 'skip':
                continue
            judged += 1
            worst = min(worst, PROOF_SCORE[status])
            notes.append(f"{need.split(':', 1)[0].strip()} {status}" + (f' ({shown})' if shown else ''))
        return (worst, notes, judged)

    def proof_room(self, number: int, skip: Any=()) -> float | None:
        """How much one unit-line of this item may cost and keep the order inside the budget."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return None
        if self.brief.get('budget_scope') == 'each':
            return float(self.budget)
        taken = sum((self.proof_price(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids() if i not in skip and self.proof_owner(i) != number)
        return float(self.budget) - taken

    def proof_price(self, ident: str) -> float | None:
        """The listing's price, also before the result shape has named the price key."""
        price = self.price_of(ident)
        if price is not None:
            return price
        record = self.rows.get(ident) or {}
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and re.search('price|cost|amount', _words(key)):
                return float(value)
        return None

    def proof_rows(self, number: int, skip: Any=()) -> list[dict[str, Any]]:
        """Opened, buyable listings of an item inside the budget: strongest evidence first, then cheapest."""
        room = self.proof_room(number, skip)
        qty = self.items[number - 1].qty
        cat_key = self.category_key()
        rows = []
        for ident in self.full:
            if ident in skip or not self.buyable(ident) or self.proof_owner(ident) != number:
                continue
            if cat_key and self.whole_label(number, str((self.rows.get(ident) or {}).get(cat_key) or '')):
                continue
            price = self.proof_price(ident)
            if room is not None and price is not None and price * qty > room + 1e-09:
                continue
            score, notes, judged = self.proof_read(ident, number)
            rows.append({'id': ident, 'price': price, 'score': score, 'notes': notes, 'judged': judged})
        rows.sort(key=lambda r: (-r['score'], r['price'] if r['price'] is not None else float('inf')))
        return rows

    def proof_table(self) -> str:
        """The table the agent reads, shown again only when it has changed."""
        lines = []
        for number, item in enumerate(self.items, 1):
            rows = [r for r in self.proof_rows(number) if r['judged']][:PROOF_SHOW]
            if rows:
                lines.append(f'item {number} ({item.name}):')
                lines += [f"  {r['id']} price {r['price']}: {'; '.join(r['notes'])[:170]}" for r in rows]
        text = '\n'.join(lines)
        if not text or text == self.proof_shown:
            return ''
        self.proof_shown = text
        return "[proof table: what the store's own attributes say on opened listings; pass = an attribute naming the property states the value, conflict = another attribute, which states this value on other listings, says otherwise here, weak = stated only elsewhere, differs = the named attribute says something else]\n" + text

    def proof_faults(self) -> list[str]:
        """Better opened listings for the cart's lines, by the store's own attributes. Empty when the cart is the best seen."""
        self.proof_learn()
        lines = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        pick = self.brief.get('pick')
        faults = []
        if pick in ('most', 'least'):
            return faults
        for ident in lines:
            number = self.proof_owner(ident)
            if not number or ident not in self.full:
                continue
            score, notes, judged = self.proof_read(ident, number)
            if not judged:
                continue
            passing = [r for r in self.proof_rows(number, lines) if r['score'] == 4 and r['judged']]
            price = self.proof_price(ident)
            chosen = None
            if pick == 'cheapest':
                cheaper = [r for r in passing if r['price'] is not None and price is not None and r['price'] < price]
                chosen = min(cheaper, key=lambda r: r['price']) if cheaper else (passing and score <= 1 and min(passing, key=lambda r: r['price'] if r['price'] is not None else float('inf'))) or None
            elif score < 4 and passing:
                chosen = passing[0]
            if chosen:
                said = '; '.join(chosen['notes'])[:220]
                faults.append(f"{chosen['id']} (price {chosen['price']}) states every requirement: {said}. The cart's {ident} (price {price}): {'; '.join(notes)[:220]}. Take {ident} out and add {chosen['id']} at the requested quantity")
        return faults

    def proof_thin(self) -> str:
        """A cart line none of whose requirements' own attributes state the value, with no opened listing better."""
        self.proof_learn()
        if self.brief.get('pick') in ('most', 'least'):
            return ''
        said = []
        for ident in [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i in self.full]:
            number = self.proof_owner(ident)
            if not number:
                continue
            score, notes, judged = self.proof_read(ident, number)
            if judged and score < 4 and not [r for r in self.proof_rows(number) if r['score'] == 4 and r['judged']]:
                said.append(f"{ident}: {'; '.join(notes)[:200]}")
        return "No opened listing has an attribute naming the requirement that states it. " + ' | '.join(said) + '.' if said else ''

    def proof_leads(self) -> dict[str, list[str]]:
        """Before a decline: opened listings that state every requirement of every item. Empty unless each item has one."""
        self.proof_learn()
        named: dict[str, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            rows = [r['id'] for r in self.proof_rows(number) if r['score'] == 4 and r['judged']]
            if not rows:
                return {}
            named[str(number)] = rows[:4]
        return named

    def proof_order(self, lost: list[str], picks: list[str]) -> list[str]:
        """Sold-out replacements, strongest evidence first; listings the table passes join even when the model missed them."""
        self.proof_learn()
        numbers = {self.proof_owner(i) for i in lost} - {0}
        if len(numbers) != 1:
            return picks
        number = numbers.pop()
        rows = self.proof_rows(number, set(lost) | set(self.cart_ids()))
        cat_key = self.category_key()
        if cat_key:
            picks = [i for i in picks if not self.whole_label(number, str((self.rows.get(i) or {}).get(cat_key) or ''))]
        scores = {r['id']: r['score'] for r in rows if r['judged']}
        passing = [r['id'] for r in rows if r['score'] == 4 and r['judged']]
        merged = list(dict.fromkeys(passing + picks)) if self.brief.get('pick') != 'cheapest' else list(dict.fromkeys(picks + passing))
        keep = [i for i in merged if scores.get(i, 1) >= 2]
        ordered = keep or merged
        if self.brief.get('pick') == 'cheapest':
            return sorted(ordered, key=lambda i: (scores.get(i, 1) < 4, self.proof_price(i) if self.proof_price(i) is not None else float('inf')))
        return sorted(ordered, key=lambda i: -scores.get(i, 1))

    def line_faults(self) -> list[str]:
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
            for i in lines:
                whole = self.whole_label(n, label[i])
                if whole:
                    faults.append(f"{i} is in the category {label[i]!r}, a neighbouring one; {whole!r} is the category for the whole of what was asked ({item.name}): list that category and pick from it")
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
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
        fitted = self.fit_lines(ids)
        if fitted:
            prompt += '\n\nCOMPATIBILITY READ BY CODE (model codes, the other names a device is sold as, code families and exclusions; this decides whether a part fits):\n' + fitted
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
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

    def fit_lines(self, ids: list[str]) -> str:
        """For a planned or held cart: each part's fit to what it must fit, as the atlas reads it."""
        out = []
        for number in sorted({p for p, _ in self.bound} | set(self.fit_codes)):
            targets = self.fit_targets(number)
            planned = [i for i in ids if i in self.item_ids(number)]
            if not targets and planned:
                # A planned cart is not in the store's cart yet: judge against the planned device.
                devices = [i for p, m in self.bound if p == number for i in ids if i in self.item_ids(m) and self.atlas.own.get(i)]
                targets = [(self.atlas.target(d), f'{d} ({self.atlas.describe(self.atlas.own[d])})') for d in devices[:1]]
            for line in planned:
                said = self.fit_verdict(line, targets)
                if targets and said is not None:
                    out.append(f"{line} {'FITS' if said else 'DOES NOT FIT'} {' and '.join(name for _, name in targets)}")
        return '\n'.join(out)

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
            text += '[Keystone opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
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
        if self.brief.get('pick') == 'cheapest' and self.left() >= 10:
            self.pages_read = 0
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Keystone] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
            return self.proof_order(lost, [])[:4]
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return self.proof_order(lost, [])[:4]
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return self.proof_order(lost, ranked)[:4]

    def sweep(self) -> str:
        """Before a decline: read the cheapest unread listings of each item that are inside the budget, once."""
        if self.swept or self.left() < self.tail() + 8 or self.late():
            return ''
        self.swept = True
        if self.capped:
            # The searches so far could not reach all their matches, so "nothing fits" is not shown yet: look further before judging.
            numbers = list(range(1, len(self.items) + 1))
            leads = {n: [' '.join(self.value_words(n) + [self.items[n - 1].name])] for n in numbers if self.value_words(n)}
            before = set(self.rows)
            ran = self.carve_search(numbers, 8 + SWEEP_OPENS, leads)
            self.trace.append({'kind': 'carve', 'words': ran, 'new_rows': len(set(self.rows) - before), 'why': 'decline'})
        share = max(2, SWEEP_OPENS // max(1, len(self.items)))
        picks: list[str] = []
        for number in range(1, len(self.items) + 1):
            room = self.proof_room(number)
            qty = self.items[number - 1].qty
            pool = [i for i in self.item_ids(number) if i not in self.full and self.buyable(i)]
            pool = [i for i in pool if room is None or self.proof_price(i) is None or self.proof_price(i) * qty <= room + 1e-09]
            pool.sort(key=lambda i: self.proof_price(i) if self.proof_price(i) is not None else float('inf'))
            picks += pool[:share]
        read = self.fill_facts(picks[:SWEEP_OPENS], self.tail() + 5)
        self.trace.append({'kind': 'sweep', 'read': len(read)})
        return '[Keystone read these unread listings inside the budget before ending]\n' + '\n'.join(read) if read else ''

    def exhausted(self) -> bool:
        """Steps or time ran out, so an unread or unresolved listing says nothing about whether the store has a fit."""
        return self.late() or self.left() <= self.tail() + len(self.items) + 2

    def commit_due(self) -> bool:
        """Whether a decline should become an order. With several items a correct decline is rare and an order often passes; with one item when steps or time ran out, or when a listing already chosen sold out, since that shows nothing about the rest of the store. A sold-out line does not count as a cart."""
        if self.committed or not self.roles.get('cart') or not self.roles.get('buy'):
            return False
        if any((not self.sold_out(i) for i in self.cart_lines)):
            return False
        lost = any((self.sold_out(i) for i in self.ever_carted | set(self.cart_lines)))
        return len(self.items) > 1 or lost or self.exhausted()

    def commit_near(self) -> bool:
        """Only the steps to fill the cart and check out remain: an empty cart gets the best known of each item, a cart
        short of an item gets that item's best known listing."""
        if not (self.roles.get('cart') and self.roles.get('buy')):
            return False
        if not self.cart_touched:
            return not self.committed and self.left() <= self.tail() + len(self.items) + 1
        if self.short_filled or len(self.items) < 2:
            return False
        short = self.short_items()
        if not short or self.left() > self.tail() + len(short) + 1:
            return False
        self.fill_short()
        return True

    def short_items(self) -> list[int]:
        """Items without a line in a cart that holds fewer lines than items. An item no seen listing belongs to is not
        counted, since its listings cannot be told apart from another item's."""
        ids = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        if not ids or len(ids) >= len(self.items):
            return []
        held = {self.proof_owner(i) for i in ids}
        return [n for n in range(1, len(self.items) + 1) if n not in held and self.item_ids(n)]

    def fill_short(self) -> bool:
        """Before a forced order: add the best known listing of each item the cart lacks, inside the budget. A cart
        missing an item fails whatever else it holds, so a best guess can only help."""
        self.short_filled = True
        self.proof_learn()
        lines = {i: self.held_units(i) or 1 for i in self.cart_ids() if not self.sold_out(i)}
        added = {}
        for number in self.short_items():
            room = self.proof_room(number)
            qty = self.items[number - 1].qty
            for row in self.best_options(number):
                if row['id'] in lines:
                    continue
                price = row['price']
                if room is not None and price is not None and price * qty > room + 1e-09:
                    continue
                added[row['id']] = qty
                break
        if not added:
            return False
        return self.put_in_cart({**lines, **added}, 'fill_short')

    def best_options(self, number: int) -> list[dict[str, Any]]:
        """What the store offers an item, best first: opened listings by evidence, a listing that states another value left out, then unread listings by price."""
        rows = [r for r in self.proof_rows(number) if r['score'] > 0 and not self.sold_out(r['id'])]
        room = self.proof_room(number)
        qty = self.items[number - 1].qty
        read = {r['id'] for r in self.proof_rows(number)}
        unread = []
        for ident in self.item_ids(number):
            if ident in read or ident in self.full or not self.buyable(ident):
                continue
            price = self.proof_price(ident)
            if room is None or price is None or price * qty <= room + 1e-09:
                unread.append({'id': ident, 'price': price, 'score': 0.5, 'notes': [], 'judged': 0})
        unread.sort(key=lambda r: r['price'] if r['price'] is not None else float('inf'))
        found = rows + unread
        targets = self.fit_targets(number) if (self.fit_codes.get(number) or any(p == number for p, _ in self.bound)) else []
        if targets:
            # A listing whose own words say it fits comes first; one that says it does not fit goes last.
            marks = {r['id']: self.fit_verdict(r['id'], targets) for r in found}
            found = [r for r in found if marks[r['id']] is True] + [r for r in found if marks[r['id']] is None] + [r for r in found if marks[r['id']] is False]
        carted = [i for p, m in self.bound if m == number for i in self.cart_ids() if i in self.item_ids(p) and not self.sold_out(i)]
        if carted:
            # A device for parts already in the cart: those the parts say they fit first, those they say they do not last.
            atlas = self.atlas_sync()

            def rank(row: dict[str, Any]) -> int:
                said = [atlas.verdict(i, atlas.target(row['id'])) for i in carted]
                return 2 if False in said else 0 if said and all(v is True for v in said) else 1
            found = sorted(found, key=rank)
        return found

    def joint_options(self, options: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        """Choose each device together with the parts bound to it: the device candidate whose seen parts cover the most
        bound items (fits stated by the parts' own words first), then its own evidence. The device is then locked and each
        part list keeps only the parts that fit it (or say nothing), so a later budget swap cannot break the fit."""
        if not self.bound:
            return options
        self.atlas_assist(now=True)
        atlas = self.atlas_sync()
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            best = None
            for rank, device in enumerate(options[main - 1][:12]):
                codes = atlas.target(device['id'])
                if not codes:
                    continue
                kept: dict[int, list[dict[str, Any]]] = {}
                sure = 0
                for p in parts:
                    named = self.fit_codes.get(p)
                    marks = [(atlas.verdict(r['id'], codes), r) for r in options[p - 1]]
                    good = [r for v, r in marks if v is True]
                    open_ = [r for v, r in marks if v is None]
                    if named:
                        own = atlas.aliases(set(named))
                        good = [r for r in good if atlas.verdict(r['id'], own) is not False]
                        open_ = [r for r in open_ if atlas.verdict(r['id'], own) is not False]
                    kept[p] = good + open_
                    sure += bool(good)
                key = (sum(1 for v in kept.values() if v), sure, device['score'], -rank)
                if best is None or key > best[0]:
                    best = (key, device, kept)
            if best is None or best[0][1] == 0:
                continue
            key, device, kept = best
            twins = [r for r in options[main - 1] if r['id'] != device['id'] and atlas.target(r['id']) == atlas.target(device['id'])]
            options[main - 1] = [device] + twins
            for p, rows in kept.items():
                if rows:
                    options[p - 1] = rows
            self.trace.append({'kind': 'joint_fit', 'device': device['id'], 'cover': key[0], 'sure': key[1], 'parts': {p: [r['id'] for r in rows[:3]] for p, rows in kept.items()}})
        return options

    def best_lines(self) -> dict[str, int]:
        """One listing per item that together stay inside the budget. Empty when an item has none, so a real dead end still declines."""
        self.proof_learn()
        options = self.joint_options([self.best_options(n) for n in range(1, len(self.items) + 1)])
        if not options or not all(options):
            return {}
        picks = [row[0] for row in options]
        limited = isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each'
        while limited:
            total = sum((p['price'] or 0.0) * self.items[n].qty for n, p in enumerate(picks))
            if total <= float(self.budget) + 1e-09:
                break
            swaps = []
            for n, row in enumerate(options):
                current = picks[n]['price'] or 0.0
                cheaper = [r for r in row if r['price'] is not None and r['price'] < current]
                if cheaper:
                    best = max(cheaper, key=lambda r: (r['score'], -r['price']))
                    swaps.append((best['score'] - picks[n]['score'], (current - best['price']) * self.items[n].qty, n, best))
            if not swaps:
                return {}
            _, _, n, best = max(swaps, key=lambda x: (x[0], x[1]))
            picks[n] = best
        lines: dict[str, int] = {}
        for n, pick in enumerate(picks):
            lines[pick['id']] = lines.get(pick['id'], 0) + self.items[n].qty
        return lines

    def put_in_cart(self, lines: dict[str, int], kind: str) -> bool:
        cart = self.roles.get('cart')
        calls = self.cart_calls(lines, [i for i in self.cart_lines if i not in lines and self.sold_out(i)])
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        self.trace.append({'kind': kind, 'lines': lines})
        return True

    def commit_cart(self) -> bool:
        lines = self.best_lines() if not self.finished else {}
        if not lines and not self.finished and len(self.items) > 1:
            lines = self.plan_cart()
        return bool(lines) and self.put_in_cart(lines, 'commit_cart')

    def plan_cart(self) -> dict[str, int]:
        """Last resort before a multi-item decline (race 15: every one of 59 such declines failed, 34 of them with an
        empty cart that the code could not fill because an item matched no seen category by its words). A model proposes
        one seen listing per item; code keeps the proposal only when every item gets an in-stock listing, the total fits
        the budget and no part contradicts the device it must fit. The categories it used are learned for the item."""
        if self.planned or self.late() or not self.judges(SPEND_HARD):
            return {}
        self.planned = True
        cat_key = self.category_key()
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident) or not self.buyable(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            records.append(f'{ident}: ' + _dump(_clip(record, 140))[:700])
            if len(records) >= 40:
                break
        if not records:
            return {}
        prompt = CART_PLAN + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        plan = self.fit_plan()
        if plan:
            prompt += '\n\nCOMPATIBILITY READ BY CODE:\n' + plan
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(SPEND_HARD), 2000) or {}
        picks: dict[int, str] = {}
        for entry in _items(got.get('cart')):
            if not isinstance(entry, dict):
                continue
            number, ident = (_count(entry.get('item')), str(entry.get('id') or '').strip())
            if number and 1 <= number <= len(self.items) and ident in self.rows and not self.sold_out(ident) and self.buyable(ident):
                picks.setdefault(number, ident)
        self.trace.append({'kind': 'plan_cart', 'picks': picks, 'reason': str(got.get('reason') or '')[:200]})
        if len(picks) < len(self.items) or len(set(picks.values())) < len(picks):
            return {}
        # The model's choice of category for an item is kept, so later code finds that item's listings.
        if cat_key:
            for number, ident in picks.items():
                label = str((self.rows.get(ident) or {}).get(cat_key) or '')
                if label and not self.owner_of(label):
                    self.shelf_map[_words(label)] = number
        atlas = self.atlas_sync()
        for part, main in self.bound:
            codes = atlas.target(picks[main])
            if codes and atlas.verdict(picks[part], codes) is False:
                label = str((self.rows.get(picks[part]) or {}).get(cat_key) or '') if cat_key else ''
                swap = [i for i in self.seen_order if i not in picks.values() and not self.sold_out(i) and self.buyable(i) and label and str((self.rows.get(i) or {}).get(cat_key) or '') == label and atlas.verdict(i, codes) is True]
                if not swap:
                    return {}
                picks[part] = min(swap, key=lambda i: self.proof_price(i) if self.proof_price(i) is not None else float('inf'))
        limited = isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each'
        total = sum((self.proof_price(i) or 0.0) * self.items[n - 1].qty for n, i in picks.items())
        if limited and total > float(self.budget) + 1e-09:
            return {}
        lines: dict[str, int] = {}
        for number, ident in picks.items():
            lines[ident] = lines.get(ident, 0) + self.items[number - 1].qty
        return lines

    def rescue_cart(self) -> bool:
        """The episode must end with an empty cart: fill it with proven fits, or else with the best known of each item."""
        if not self.roles.get('cart') or self.cart_lines or self.finished:
            return False
        named = self.proof_leads()
        if named:
            return self.put_in_cart({ids[0]: self.items[int(number) - 1].qty for number, ids in named.items()}, 'rescue_cart')
        if self.committed:
            return False
        self.committed = True
        return self.commit_cart()

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
        self.atlas_assist(now=True)
        plan = self.fit_plan()
        if plan:
            prompt += '\n\nCOMPATIBILITY READ BY CODE (decides whether a part fits a device; a part fits only the devices it names or whose code family it names, never an excluded one):\n' + plan
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 4000)
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
            index = _count(number)
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
        return 'Not run (Keystone acted instead): the independent check confirmed ' + self.confirmed + ', so Keystone set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.ever_carted |= set(self.cart_lines)
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
                have = have - count if count else 0
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
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Keystone opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Keystone's interviewer handles every message to the shopper."))
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
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Keystone] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
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
                    trimmed += f"\n[Keystone] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
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
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). By its own compatibility words, a part in the cart does not fit what it must fit: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device it must fit), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Keystone set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    thin = self.proof_thin() if not self.proof_pressed and self.left() > PROOF_ROOM else ''
                    if thin:
                        self.proof_pressed = True
                        self.trace.append({'kind': 'proof_thin', 'said': thin[:300]})
                        out.append((cid, 'Not run yet (no step used). ' + thin + f' {self.left()} steps are left: open more listings of this kind first, searching with the wanted value, then check out.\nCart now: ' + self.cart_text()))
                        continue
                    better = self.proof_faults() if self.proof_holds < 2 and self.left() > self.tail() + 3 else []
                    if better:
                        self.proof_holds += 1
                        self.trace.append({'kind': 'proof_hold', 'faults': [f[:300] for f in better[:3]]})
                        out.append((cid, "Not run yet (no step used). The store's own attributes show a better fit than the cart: " + ' | '.join(better[:3]) + '. Then check out.\nCart now: ' + self.cart_text()))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Keystone read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Keystone re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    swept = self.sweep()
                    if swept:
                        out.append((cid, 'Not run yet (no step used). ' + swept + '\nJudge these against the requirements: buy what fits; end without a purchase only if none does.'))
                        continue
                    review = self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if not self.leads:
                        named = self.proof_leads()
                        if named:
                            self.lead_items = named
                            self.leads = [i for ids in named.values() for i in ids]
                            self.trace.append({'kind': 'proof_leads', 'leads': list(self.leads)})
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        owner = self.proof_owner(first)
                        scored = self.proof_read(first, owner) if owner and first in self.full else (2, [], 0)
                        if scored[2] and scored[0] <= 1:
                            self.trace.append({'kind': 'proof_unconfirmed', 'id': first, 'notes': scored[1][:4]})
                            self.leads = [i for i in self.leads if i != first]
                        elif self.verdicts.get(self.check_key([first])) and first in self.leads:
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
            if self.is_refuse(name, args) and self.commit_due():
                self.committed = True
                if self.commit_cart():
                    turned = self.finish(again=True)
                    if self.finished or self.broke:
                        out.append((cid, 'Keystone checked out the best-known cart instead of ending without a purchase.'))
                        continue
                    out.append((cid, 'Not run (no step used). Keystone carted the best-known listing of each item instead of ending without a purchase. ' + (turned or '') + '\nCart now: ' + self.cart_text() + '\nFix what the store says, then check out.'))
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
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Keystone] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Keystone] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation) + self.cap_note(name, args, observation)
            text += self.page_notes(observation) if not ending else ''
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Keystone] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Keystone re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Keystone's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and not self.cart_touched:
            self.rescue_cart()
            if self.finished or self.left() <= 0:
                return ''
        if buy and len(self.items) >= 2 and not self.short_filled and self.short_items() and self.left() >= len(self.short_items()) + 2:
            self.fill_short()
            if self.finished or self.left() <= 0:
                return ''
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
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

    def order_instead(self) -> bool:
        """The shopper refused an ending without a purchase. The refusal itself was read (a new budget, say), so with the
        steps to cart and check out, order the best known listings rather than ending the same way again."""
        buy = self.roles.get('buy')
        if self.finished or not buy or not self.roles.get('cart') or self.left() < len(self.items) + 2:
            return False
        if not [i for i in self.cart_ids() if not self.sold_out(i)]:
            lines = self.best_lines()
            if not lines or not self.put_in_cart(lines, 'order_after_refusal'):
                return False
        if self.finished or self.left() <= 0:
            return False
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return False
        if self.left() >= 2:
            self.gate('order')
        if self.finished or self.left() <= 0:
            return False
        reply = self.act(buy, args)
        self.trace.append({'kind': 'order_after_refusal', 'reply': _clip(reply, 200)})
        if reply is not None and not self.finished:
            self.observe(buy, reply.get('observation'))
        return self.finished

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
        self.work(self.loop_cap(), SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned.startswith('ending without') and self.order_instead():
                turned = ''
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Keystone] Keystone tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (not self.late()):
            if self.commit_near():
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
                if self.no_credit or self.calls >= self.call_cap() or self.misses > 2:
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
        episode = Keystone(problem_data)
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
