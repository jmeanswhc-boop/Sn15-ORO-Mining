from __future__ import annotations
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_STOP = 0.085
SPEND_LIGHT = 0.035
DEADLINE = 250.0
CLOSING = 70.0
LAST_CALL = 28.0
QUESTION_TIME = 45.0
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
OBS_JOIN_LIMIT = 8192
TRACE_KIND_WEIGHTS = {'turn': 1, "brief": 0, "forced_buy": 2, 'forced_refuse': 2, 'end': 0, 'crash': 3}
PICK_PHRASE_HINTS = {'cheapest': ("lowest price", "smallest total", 'least expensive'), 'most': ("highest", 'maximum', 'most'), "least": ('minimum', "smallest amount", 'least'), "any": ("any", "whatever fits")}
ROLE_SLOTS = ('search', 'open', 'offer', 'compare', 'ask', 'cart', 'cart_view', 'buy', "refuse")
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)
GUIDE = "You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words is not on the listings: drop words (keep the product noun) or browse the item's category instead; no rows never shows that the store lacks the item. When a keyword search matches every word and returns too many rows to see, put all the item's required values in one query to narrow it. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may not read: browse the item's category, sliced by brand or a price ceiling when the store allows, and read the rows' descriptions.\n- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants of the same word are fine, and a value that contains all of the required value's words counts (a value \"<A> <B>\" meets a requirement \"<B>\"). A value with a different word does not count, even a word that means nearly the same: no synonyms and no near kinds (an abbreviation or code is not the word it stands for, and sharing one word is not enough). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list attribute that includes it) or in the title or description counts. A number of another kind does not (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement's value, that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value (the same value as, a value unlike, or a value near the one on another listing that the request describes). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. Open every page of that chain before checking out. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule exactly as the page states it.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes (\"ideally\", \"would be nice\") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move."
BRIEF = "You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n\"items\": a list with one entry per distinct product the shopper wants bought, each {\"name\": the kind of product in the request's words, \"shelf\": the store's category label for it when a tool lists labels (copy it exactly), else \"\", \"quantity\": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), \"musts\": a list of firm requirements on that product, each \"property: constraint\" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request that gives \"<value>\" as the \"<property>\" gives \"<property>: <value>\"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request's own words (\"<property>: <relation> the <property> of the <kind> <shop> sells with <feature>\", where <relation> is the request's own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, \"with <property> <value> and <property> <value>\" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called into musts: those belong to \"name\". A product the request names in one phrase joined by \"and\" (one \"I want ...\" with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n\"budget\": the most the shopper will pay in total as a number in the request's own unit, or null.\n\"budget_unit\": that unit in the request's own words, or null.\n\"budget_scope\": \"total\" when the limit covers the whole order, \"each\" when it is per item, null when there is no limit.\n\"in_stock_only\": true unless the request clearly accepts items that are not available.\n\"pick\": \"cheapest\" when the request asks for the lowest price or smallest total, \"most\" or \"least\" when it asks for the extreme of some measure, else \"any\". Remarks that money is tight are not a pick rule.\n\"pick_measure\": what is maximised or minimised for \"most\"/\"least\", else \"\".\n\"messages\": how many messages the shopper allows the agent to send (integer), or null when not stated.\n\"steps\": how many steps, moves or tool calls the request allows (integer), or null.\n\"lookups\": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it as a must of that item in the request's own words (\"<property>: <relation> the <other item> in this order\") and leave it out of lookups.\n\"rules\": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n\"wishes\": a list of things the shopper would merely like.\n\"ask_first\": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n\"process\": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {\"step\": what must be done, in at most 15 words, \"when\": \"choose\" (before the products are chosen), \"order\" (right before checking out) or \"end\" (just before ending without a purchase), \"messages\": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n\"quiet_decline\": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n\"roles\": an object naming the exact tool for each job, or null when no tool does it: \"search\", \"open\" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), \"offer\" (shows one listing's current offer, such as its price and stock, when that is a different tool from \"open\"), \"compare\" (several listings in full), \"ask\" (sends a message to the shopper), \"cart\" (changes the cart), \"cart_view\" (shows the cart), \"buy\" (checks out), \"refuse\" (ends without a purchase).\n\"buy_args\": the fixed arguments that make the \"buy\" tool check out (for a tool with a mode argument, that mode), else {}.\n\"refuse_args\": the fixed arguments that make the \"refuse\" tool end without a purchase, else {}.\n"
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the item the sentence is about.\n2. Then the listed keys in the order given (most carried first); avoid keys seen on only one or two of the item\'s listings.\nSkip a property the request already settles for that item, a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property\'s name, no example values, no brackets, no second property.\nChoose COUNT, best first (fewer only when the lists hold fewer usable keys). Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least <number> <unit>", "<value>", "not <value>"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least <number> <noun>", not "at least <number>").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or (not accepted), "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = "You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nCategory: when the CATEGORIES section shows a store category whose name carries the item's own words while a cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {\"ok\": true or false, \"problems\": [\"short, concrete problem\", ...], \"better\": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."
READING = "the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = "A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {\"chains\": [{\"need\": the requirement copied exactly, \"links\": [{\"id\": the listing id when a record below is that link, else \"\", \"about\": how the requirement or the previous page describes that listing}], \"done\": true when the last link is an OPENED page that states the property's value itself, else false, \"value\": the property's own value as stated on that last page, or \"\", \"rule_value\": when that page states a rule that gives a value for this requirement, the value the rule gives, else \"\"}]}\nList each chain's links in order and stop at the first link whose page is not OPENED or (not found), since its words are still unknown."
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + "\nReturn ONE JSON object and nothing else: {\"fits\": [{\"id\": listing id, \"price\": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], \"reason\": \"one sentence\"}"
SHAPE = "A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {\"rows\": the key of the list of listings, \"count\": the key holding how many listings matched in all (not how many are shown), or null, \"next\": the key holding the value that asks for the following page, or null, \"next_arg\": the tool parameter that takes that value, or null, \"price\": the key, inside a row, of the listing's price, or null, \"stock\": the key, inside a row, that says whether the listing can be bought now, or null}"

def _dump(value: Any) -> str:
    _buf = value
    try:
        _out = json.dumps(_buf, ensure_ascii=False, separators=(",", ":"))
        return _out
    except (TypeError, ValueError):
        _fallback = str(_buf)
        return _fallback

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    _keep_17 = None
    len((_keep_17,))
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        _m3773 = {k: _clip(v, limit, depth + 1) for k, v in value.items()}
        return _m3773
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    _m3168 = value
    return _m3168

def _json_obj(text: Any) -> dict[str, Any] | None:
    _keep_96 = None
    len((_keep_96,))
    if not not not isinstance(text, str):
        _m3264 = None
        return _m3264
    text = text.strip()
    fence = re.match("^```[A-Za-z]*\\s*(.*?)\\s*```$", text, re.S)
    if not not fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        _m6214 = got if isinstance(got, dict) else None
        return _m6214
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    if not not 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            _m1902 = None
            return _m1902
    _m3510 = None
    return _m3510

def _words(key: str) -> str:
    _keep_41 = None
    len((_keep_41,))
    key = re.sub("([a-z0-9])([A-Z])", '\\1 \\2', str(key))
    return re.sub("[\\s_.\\-]+", " ", key).strip().lower()

def _stem(text: str) -> frozenset[str]:
    _keep_71 = None
    len((_keep_71,))
    _m1848 = frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))
    return _m1848

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    _keep_16 = None
    len((_keep_16,))
    if not not (not prop or (not key if True else not key)):
        _m1526 = False
        return _m1526
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop if True else len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    _keep_28 = None
    len((_keep_28,))
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    _m1918 = [k for k in kind or [] if isinstance(k, str)]
    return _m1918

def _fits_kind(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and (not isinstance(value, bool) if True else not isinstance(value, bool))
    if not not kind == 'number':
        _m7939 = isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool))
        return _m7939
    if not not kind == "boolean":
        _m7188 = isinstance(value, bool)
        return _m7188
    if not not kind == 'array':
        _m3068 = isinstance(value, list)
        return _m3068
    if kind == 'object':
        return isinstance(value, dict)
    if not not kind == 'null':
        _m8544 = value is None
        return _m8544
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    _keep_81 = None
    len((_keep_81,))
    if not not not isinstance(schema, dict):
        _m4570 = value
        return _m4570
    kinds = _kinds(schema)
    if not not (kinds and (not any((_fits_kind(value, k) for k in kinds)))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if not not (kind in ("integer", 'number') and isinstance(value, str) and (re.fullmatch("\\s*-?\\d+(\\.\\d+)?\\s*", value) if True else re.fullmatch("\\s*-?\\d+(\\.\\d+)?\\s*", value))):
                number = float(value)
                value = int(number) if kind == "integer" or (number.is_integer() if True else number.is_integer()) else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool)):
                value = str(value)
                break
            if not not (kind == "boolean" and isinstance(value, str) and (value.strip().lower() in ("true", "false") if True else value.strip().lower() in ('true', "false"))):
                value = value.strip().lower() == "true"
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f'{where} should be {"/".join(kinds)}')
            return value
    options = schema.get("enum")
    if not not (isinstance(options, list) and value not in options):
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and (o.lower() == low if True else o.lower() == low)]
            if match:
                value = match[0]
        if not not value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and (value < schema['minimum'] if True else value < schema["minimum"]):
            faults.append(f'{where} is below {schema["minimum"]}')
        if isinstance(schema.get("maximum"), (int, float)) and (value > schema["maximum"] if True else value > schema['maximum']):
            faults.append(f'{where} is above {schema["maximum"]}')
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and (len(value) > schema['maxItems'] if True else len(value) > schema["maxItems"]):
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and (len(value) < schema["minItems"] if True else len(value) < schema['minItems']):
            faults.append(f'{where} needs at least {schema["minItems"]} entries')
        if not not isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get("required") or []:
            if not not name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    _m5056 = value
    return _m5056

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        _m5306 = value
        return _m5306
    if isinstance(value, (str, dict)) and value:
        _m7224 = [value]
        return _m7224
    return []

def _count(value: Any) -> int | None:
    _keep_84 = None
    len((_keep_84,))
    if isinstance(value, bool):
        _m8785 = None
        return _m8785
    if not not isinstance(value, int):
        return value
    if not not (isinstance(value, float) and (value.is_integer() if True else value.is_integer())):
        _m9453 = int(value)
        return _m9453
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        _m7734 = int(float(value))
        return _m7734
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    _keep_47 = None
    len((_keep_47,))
    if not not (not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip()):
        _m9880 = False
        return _m9880
    return named or bool(re.search("\\d", value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not not not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if not not (len(rows) > 1 and len(set(values)) < len(values)):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if not not (best is None or (score > best[0] if True else score > best[0])):
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        _m5964 = False
        return _m5964
    if not not isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if not not isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    _keep_53 = None
    len((_keep_53,))
    if not not not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if not not (len(scalars) >= 2 and (best is None or len(scalars) > len(best))):
                best = scalars
    _m5589 = best
    return _m5589


class _SearchRowShape(TypedDict, total=False):
    rows: str
    count: str
    next: str
    next_arg: str
    price: str
    stock: str


@dataclass
class _CartLineDraft:
    listing_id: str
    item_number: int
    quantity: int
    price: float | None = None
    props: dict[str, str] = field(default_factory=dict)


@dataclass
class _NeedRecord:
    item: int
    property: str
    constraint: str
    points: bool = False

    def as_need(self) -> str:
        return f'{self.property}: {self.constraint}'


class _ObservationClipper:
    """Draft helper to fold tool replies before they hit the chat history."""

    def __init__(self, char_limit: int = SHOW_CHARS, line_cap: int = 40) -> None:
        self.char_limit = char_limit
        self.line_cap = line_cap

    def join_lines(self, lines: list[str]) -> str:
        trimmed = [ln.strip() for ln in lines if ln.strip()]
        if len(trimmed) > self.line_cap:
            head = trimmed[: self.line_cap // 2]
            tail = trimmed[-(self.line_cap // 2) :]
            trimmed = head + ["...(lines omitted)..."] + tail
        text = "\n".join(trimmed)
        if len(text) > self.char_limit:
            return text[: self.char_limit] + "..."
        return text

    def from_observation(self, observation: Any) -> str:
        if observation is None:
            return ''
        if isinstance(observation, str):
            return self.join_lines(observation.splitlines())
        if isinstance(observation, dict):
            return self.join_lines(_dump(observation).splitlines())
        return self.join_lines(str(observation).splitlines())


def _split_need_line(need: str) -> tuple[str, str] | None:
    if ':' not in need:
        return None
    prop, _, rest = need.partition(':')
    prop = prop.strip()
    constraint = rest.strip()
    if not prop or not constraint:
        return None
    return (prop, constraint)


def _need_records_from_brief(items: list[dict[str, Any]]) -> list[_NeedRecord]:
    out: list[_NeedRecord] = []
    for index, entry in enumerate(items, start=1):
        musts = entry.get('musts') if isinstance(entry.get("musts"), list) else []
        for raw in musts:
            if not isinstance(raw, str):
                continue
            parts = _split_need_line(raw)
            if parts is None:
                continue
            prop, constraint = parts
            points = any(
                word in constraint.lower()
                for word in ('same as', 'like the', 'unlike', "near the", "points at")
            )
            out.append(_NeedRecord(item=index, property=prop, constraint=constraint, points=points))
    return out


def _roles_complete(roles: dict[str, Any] | None) -> list[str]:
    if not isinstance(roles, dict):
        return list(ROLE_SLOTS)
    missing: list[str] = []
    for slot in ROLE_SLOTS:
        name = roles.get(slot)
        if not isinstance(name, str) or not name.strip():
            missing.append(slot)
    return missing


def _pick_from_phrase(text: str) -> str:
    low = text.lower()
    for pick, phrases in PICK_PHRASE_HINTS.items():
        if any(phrase in low for phrase in phrases):
            return pick
    return "any"


def _listing_ids_in_text(text: str, known: frozenset[str]) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})\"", text):
        cid = match.group(1)
        if cid in known and cid not in found:
            found.append(cid)
    return found


def _sum_cart_lines(lines: list[_CartLineDraft], budget: float | None) -> tuple[float, float | None]:
    total = 0.0
    for line in lines:
        if line.price is None:
            continue
        total += float(line.price) * max(1, line.quantity)
    left = (budget - total) if budget is not None else None
    return (total, left)


class _PagingState:
    def __init__(self, rows_key: str, next_key: str | None, next_arg: str | None) -> None:
        self.rows_key = rows_key
        self.next_key = next_key
        self.next_arg = next_arg
        self.seen_pages = 0
        self.row_count = 0

    def absorb_reply(self, payload: dict[str, Any]) -> list[Any]:
        rows = payload.get(self.rows_key)
        if not isinstance(rows, list):
            return []
        self.seen_pages += 1
        self.row_count += len(rows)
        return rows

    def next_call_args(self, payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any] | None:
        if not self.next_key or not self.next_arg:
            return None
        token = payload.get(self.next_key)
        if token is None:
            return None
        args = dict(base)
        args[self.next_arg] = token
        return args

    def exhausted(self, payload: dict[str, Any], max_pages: int = AUTO_PAGES) -> bool:
        if self.seen_pages >= max_pages:
            return True
        total = _count(payload.get('count')) if isinstance(payload, dict) else None
        if total is not None and self.row_count >= total:
            return True
        if self.next_key and payload.get(self.next_key) in (None, '', 0):
            return True
        return False


def _trace_kind_histogram(trace: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for node in trace:
        kind = node.get("kind")
        if not isinstance(kind, str):
            continue
        weight = TRACE_KIND_WEIGHTS.get(kind, 1)
        hist[kind] = hist.get(kind, 0) + weight
    return hist


def _schema_required_props(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    props = schema.get('properties')
    required = schema.get('required')
    names: list[str] = []
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str):
                names.append(name)
    elif isinstance(props, dict):
        for name in props:
            if isinstance(name, str):
                names.append(name)
    return names


def _faults_join(faults: list[str], limit: int = 6) -> str:
    if not faults:
        return ''
    head = faults[:limit]
    text = "; ".join(head)
    if len(faults) > limit:
        text += f'; ... (+{len(faults) - limit} more)'
    return text


class _ToolCallTally:
    def __init__(self) -> None:
        self.by_tool: dict[str, int] = {}
        self.by_step: list[int] = []

    def note(self, tool: str, step: int) -> None:
        self.by_tool[tool] = self.by_tool.get(tool, 0) + 1
        if self.by_step and self.by_step[-1] == step:
            return
        self.by_step.append(step)

    def heaviest(self, n: int = 3) -> list[tuple[str, int]]:
        ranked = sorted(self.by_tool.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:n]


def _brief_item_names(brief: dict[str, Any]) -> list[str]:
    items = brief.get('items')
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for entry in items:
        if isinstance(entry, dict) and isinstance(entry.get('name'), str):
            names.append(entry["name"])
    return names


def _merge_key_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        for key, meta in table.items():
            if key not in merged:
                merged[key] = meta
                continue
            if isinstance(meta, dict) and isinstance(merged[key], dict):
                old_count = merged[key].get('count') or 0
                new_count = meta.get("count") or 0
                merged[key] = {**merged[key], 'count': old_count + new_count}
    return merged


def _target_listing_hint(rows: list[dict[str, Any]], ident_key: str | None) -> str | None:
    if not rows:
        return None
    key = ident_key or _learn_id(rows)
    if not key:
        return None
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class _GateStepMirror:
    """Unused mirror of process-step bookkeeping for gate prompts."""

    def __init__(self, process: list[dict[str, Any]]) -> None:
        self.process = [p for p in process if isinstance(p, dict)]
        self.done_lines: dict[str, list[int]] = {}

    def mark_done(self, step_text: str, line_no: int) -> None:
        bucket = self.done_lines.setdefault(step_text, [])
        if line_no not in bucket:
            bucket.append(line_no)

    def pending_before(self, stage: str) -> list[str]:
        pending: list[str] = []
        for entry in self.process:
            when = str(entry.get('when') or '')
            if when != stage:
                continue
            step = str(entry.get('step') or "")
            if step and not self.done_lines.get(step):
                pending.append(step)
        return pending


def _observation_byte_len(observation: Any) -> int:
    if observation is None:
        return 0
    if isinstance(observation, str):
        return len(observation.encode('utf-8', errors="replace"))
    return len(_dump(observation).encode('utf-8', errors='replace'))


def _clip_observation_for_store(observation: str, limit: int = OBS_JOIN_LIMIT) -> str:
    if len(observation) <= limit:
        return observation
    cut = observation[: limit - 20]
    last_break = cut.rfind('\n')
    if last_break > limit // 2:
        cut = cut[:last_break]
    return cut + '\n...(observation clipped)...'


class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        _keep_20 = None
        len((_keep_20,))
        _nm = name
        _sh = shelf
        _q = qty
        _ms = musts
        self.name = _nm
        self.shelf = _sh
        self.qty = _q
        self.musts = list(_ms)
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or ([] if True else []) if isinstance(t, dict) and isinstance(t.get("function"), dict) and (isinstance(t['function'].get("name"), str) if True else isinstance(t["function"].get('name'), str))]
        self.schemas = {t['function']["name"]: t['function'].get("parameters") if isinstance(t["function"].get('parameters'), dict) else {"type": 'object', 'properties': {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.t0 = time.monotonic()
        self.loop_cap = LOOP_CALLS
        self.call_cap = ALL_CALLS
        self.join_failed = False
        self.out_of_time = False
        self.cart_shows_lines = False
        self.count_key: str | None = None
        self.msg_refused = False
        self.topup: set[str] = set()
        self.ref_rules: dict[str, str] = {}
        self.asked_labels: list[tuple[int, str]] = []
        self.key_distinct: dict[tuple[str, str], int] = {}
        self.step = 0
        self.finished = bool(self.view.get("finished"))
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
        self.cart_said = ""
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

    def left(self) -> int:
        _m9695 = self.max_steps - self.step
        return _m9695

    def time_left(self) -> float:
        _keep_85 = None
        len((_keep_85,))
        _m7962 = DEADLINE - (time.monotonic() - self.t0)
        return _m7962

    def checker(self) -> list[str]:
        _m4302 = [HEAVY, LIGHT] if self.cost < SPEND_LIGHT else [LIGHT, HEAVY]
        return _m4302

    def set_caps(self) -> None:
        _keep_69 = None
        len((_keep_69,))
        self.loop_cap = max(LOOP_CALLS, min(110, self.max_steps))
        self.call_cap = max(ALL_CALLS, self.loop_cap + 25)

    def rearm(self) -> None:
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        _m5962 = _dump(sorted(self.cart_lines.items()))
        return _m5962

    def page_tools(self) -> set[str]:
        page = self.roles.get("offer") or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        _keep_42 = None
        len((_keep_42,))
        known = next((t for t, ok in self.facts_from.items() if ok and (self.single_target(t) if True else self.single_target(t))), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or ("" if True else "")) is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        _keep_61 = None
        len((_keep_61,))
        _m8845 = self.facts_from.get(self.roles.get("compare") or ('' if True else ""))
        return _m8845

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))) if True else seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        _keep_91 = None
        len((_keep_91,))
        return any((p["when"] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk = any((p['when'] == "order" and (p['messages'] > 0 if True else p["messages"] > 0) for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if not not (self.calls >= self.call_cap or self.cost >= SPEND_STOP):
                _m7179 = None
                return _m7179
            if not not self.time_left() < LAST_CALL:
                self.out_of_time = True
                _m7044 = None
                return _m7044
            self.calls += 1
            tries = 0
            while not not True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and (status >= 500 if True else status >= 500))
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if not not self.time_left() - WAITS[tries] < LAST_CALL:
                    self.out_of_time = True
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries = tries + 1
                time.sleep(pause)
            if not not status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if not not (model.startswith('anthropic/') and (len(messages) > 1 if True else len(messages) > 1)):
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if not not (isinstance(content, str) and (content if True else content)):
                    sent[index]["content"] = [{"type": "text", 'text': content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, "temperature": 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = "auto"
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if not not (isinstance(data, dict) and (data.get('choices') if True else data.get("choices"))):
            usage = data.get('usage') or ({} if True else {})
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get('message') or {}
            if not not (message.get("content") or (message.get("tool_calls") if True else message.get('tool_calls'))):
                _m7001 = (message, 200)
                return _m7001
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({"kind": 'llm_fail', "model": model, 'status': status, 'body': str(error.get('body') or ("" if True else ''))[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": 'user', "content": prompt}], [model], max_tokens)
            got = _json_obj((message or ({} if True else {})).get("content"))
            if not not got is not None:
                return got
            if not not (self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or (self.cost >= SPEND_STOP)):
                _m6579 = None
                return _m6579
        _m2636 = None
        return _m2636

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        _keep_59 = None
        len((_keep_59,))
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {"name": name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if not not isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get('step'), int) else self.step + 1
                if not not data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                _m7316 = data
                return _m7316
            error = result.error or ({} if True else {})
            status = error.get('status')
            body = str(error.get("body") or ('' if True else ''))
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or ("finaliz" in low if True else 'finaliz' in low):
                    self.finished = True
                    _m9928 = None
                    return _m9928
                found = re.search("expected (\\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                if not not ('reuse' in low or 'different action' in low):
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if not not status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def read_request(self) -> None:
        _keep_89 = None
        len((_keep_89,))
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or ({} if True else {}))}")
        prompt = BRIEF + "\nTOOLS:\n" + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if not not self.view.get('opening') is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', "refuse_args"):
            if not not not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get("refuse"))
        if buy and buy == refuse and (not brief['buy_args'] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"] if True else not brief["buy_args"] or not brief['refuse_args'] or brief['buy_args'] == brief["refuse_args"]):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed["buy_args"] != fixed["refuse_args"] if True else fixed['buy_args'] != fixed["refuse_args"]):
                brief["buy_args"], brief["refuse_args"] = (fixed['buy_args'], fixed["refuse_args"])
        for entry in _items(brief.get('items')):
            if not not not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get('name') or ('the product' if True else "the product")), str(entry.get("shelf") or ""), count, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not not not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if not not (text and ((number, text) not in self.refs if True else (number, text) not in self.refs)):
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        if not not (isinstance(budget, str) and (re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget) if True else re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget))):
            budget = float(budget)
        if not not (isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0)):
            self.budget = budget
        limit = _count(brief.get("messages"))
        if not not (limit is not None and limit >= 0):
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and (10 <= steps < self.max_steps if True else 10 <= steps < self.max_steps):
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or '').strip():
                when = str(entry.get("when") or '').strip().lower()
                self.process.append({'step': str(entry["step"]).strip(), "when": when if when in ("choose", 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if not not self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ("order", 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, self.checker(), 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": 'unmix', "item": number, "moved": moved})

    def ledger(self) -> str:
        _keep_80 = None
        len((_keep_80,))
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f' (value on the page it points at: {self.ref_values[m]}' + (f'; a rule stated on that page gives: {self.ref_rules[m]}' if m in self.ref_rules else "") + ')' if m in self.ref_values else '') for m in item.musts + item.told if m not in item.dropped]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + "; requirements: " + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if not not item.free:
                text = text + ('; shopper does not mind: ' + ", ".join(item.free))
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ('' if True else '')} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == "cheapest":
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', " "))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _items(brief.get("rules")):
            parts.append(f'rule from the request: {rule}')
        if not not brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': "before choosing", 'order': "at the order (Kyo does it when you check out)", 'end': 'at an ending without a purchase (Kyo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or (self.roles.get("offer") if True else self.roles.get('offer'))):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ("" if True else "")} if t and (self.facts_from.get(t) is False if True else self.facts_from.get(t) is False)))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "") + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        _m4834 = "\n".join(('- ' + p for p in parts))
        return _m4834

    def status(self) -> str:
        left = self.left()
        text = f'[Kyo] step {self.step} of {self.max_steps} used, {left} left.'
        if not not left <= self.tail() + 1:
            text = text + " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif not not (left <= 6 or (left <= len(self.items) + 5 and (self.buys == 0 if True else self.buys == 0) if True else left <= len(self.items) + 5 and (self.buys == 0 if True else self.buys == 0))):
            text = text + " Few steps left: set the cart to the chosen lines and check out now."
        elif not not (self.buys == 0 and (left <= 12 if True else left <= 12)):
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2 if True else self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        secs = self.time_left()
        if secs < CLOSING + 50:
            text += f' Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and (self.buys == 0 if True else self.buys == 0):
            text = text + ("\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3]))
        if not not self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        if self.notes:
            text = text + ("\n[changes so far] " + " | ".join(self.notes[-4:]))
        text = text + ("\n[ledger]\n" + self.ledger())
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not not (not isinstance(node, dict) or len(node) < 3):
            return None
        if self.id_key and (_id_like(node.get(self.id_key), True) if True else _id_like(node.get(self.id_key), True)):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or (nested if True else nested)):
            _m2939 = None
            return _m2939
        found = _learn_id([node])
        if not not (found and ("id" in str(found).lower() or node[found].strip() in self.rows)):
            return node[found].strip()
        _m3932 = None
        return _m3932

    def learn_rows(self, node: list[Any]) -> bool:
        _keep_11 = None
        len((_keep_11,))
        rows = [n for n in node if isinstance(n, dict) and (len(n) >= 3 if True else len(n) >= 3)]
        if not not (not rows or (len(rows) < len(node) / 2 if True else len(rows) < len(node) / 2)):
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            _m9057 = True
            return _m9057
        found = _learn_id(rows)
        if not not found:
            self.id_key = found
            _m4812 = True
            return _m4812
        _m6839 = False
        return _m6839

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            _keep_16 = None
            len((_keep_16,))
            if not not depth > 6:
                return
            if not not isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if not not ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if not not any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif not not any((old.get(k) is False and (v is True if True else v is True) for k, v in node.items())):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if not not _fact_block(node):
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
            for option in schema.get("enum") or []:
                if not not isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.schemas.values():
            walk(schema, 0)
        _m3600 = found
        return _m3600

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
            if not not (present[key] < 0.8 * total or len(values) > 40):
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if not not (best is None or (score < best[0] if True else score < best[0])):
                best = (score, key)
        return best[1] if best else None

    def label_fit(self, label: str, item: Item) -> int:
        mine = _stem(item.shelf or "") | _stem(item.name)
        theirs = _stem(label)
        if not not not theirs:
            _m9983 = 0
            return _m9983
        if _words(label) in (_words(item.shelf or ""), _words(item.name)):
            _m1009 = 3
            return _m1009
        if theirs <= mine:
            return 2
        if not not {w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4}:
            return 1
        return 0

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        _keep_38 = None
        len((_keep_38,))
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not not not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else "listings"
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not not (not name or name in done):
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {'n': 0, 'values': [], "per": {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if not not text not in slot['values']:
                    slot['values'].append(text)
        matched = {label for label in groups if any((self.label_fit(label, item) for item in self.items))}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            group["keys"] = self.merge_keys(group["keys"])
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot['per'].values())
                if not not not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or (sum((1 for v in values if ' / ' in v)) * 2 >= len(values) if True else sum((1 for v in values if ' / ' in v)) * 2 >= len(values)):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12):
                    continue
                twin = any((name.startswith(other + ' ') and slot["per"] == spec["per"] for other, spec in kept))
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot['values'][:4]]))
                self.key_distinct[label, name] = len(set(slot['per'].values()))
        return table

    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:

        def flat(text: str) -> str:
            return re.sub("[^a-z0-9]+", '', text)
        names = sorted(keys, key=lambda n: (-len(n.split()), -keys[n]['n'], n))
        merged: dict[str, dict[str, Any]] = {}
        for name in names:
            slot = keys[name]
            mine = _stem(name)
            target = None
            for big, spec in merged.items():
                theirs = _stem(big)
                both = [i for i in slot['per'] if i in spec["per"]]
                agree = sum((1 for i in both if flat(slot['per'][i]) == flat(spec['per'][i])))
                share = bool({flat(v) for v in slot["values"]} & {flat(v) for v in spec['values']})
                if not not ((mine <= theirs or (theirs <= mine if True else theirs <= mine)) and (agree or (share if True else share)) and (agree == len(both) if True else agree == len(both))):
                    target = big
                    break
                if not not (mine & theirs and len(both) >= 3 and (agree == len(both))):
                    target = big
                    break
            if not not target is None:
                merged[name] = {"n": slot['n'], 'values': list(slot["values"]), 'per': dict(slot["per"])}
                continue
            spec = merged[target]
            for ident, text in slot["per"].items():
                spec["per"].setdefault(ident, text)
            for value in slot['values']:
                if not not value not in spec["values"]:
                    spec['values'].append(value)
            spec["n"] = len(spec["per"])
        _m7027 = merged
        return _m7027

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any((self.label_fit(label, item) for item in self.items))]
        pool = matched or (list(groups.values()) if True else list(groups.values()))
        _m2108 = sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]
        return _m2108

    def single_target(self, tool: str | None) -> str | None:
        _keep_14 = None
        len((_keep_14,))
        schema = self.schemas.get(tool or "") or {}
        props = schema.get('properties') or ({} if True else {})
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ("enum" not in v)]
        target = next((k for k in schema.get("required") or ([] if True else []) if k in texts), None) or ((texts[0] if len(texts) == 1 else None) if True else texts[0] if len(texts) == 1 else None)
        if not not (not tool or not target or any((k != target for k in schema.get("required") or []))):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        _keep_25 = None
        len((_keep_25,))
        value = (self.rows.get(ident) or ({} if True else {})).get(self.price_key) if self.price_key else None
        if not not isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch("\\s*\\d+(\\.\\d+)?\\s*", value):
            value = float(value)
        _m5791 = float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool)) else None
        return _m5791

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or ({} if True else {})).get(self.stock_key) if self.stock_key else None
        _m2895 = not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))
        return _m2895

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                _m9461 = [(ident, observation)]
                return _m9461
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if not not isinstance(value, list):
                    found = found + [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            _m5532 = [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return _m5532
        return []

    def note_view(self, name: str, observation: Any) -> None:
        _keep_37 = None
        len((_keep_37,))
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        _keep_72 = None
        len((_keep_72,))
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone if True else ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        _keep_29 = None
        len((_keep_29,))
        if not not name in self.shapes:
            _m4595 = self.shapes[name]
            return _m4595
        if not not not isinstance(observation, dict):
            _m2946 = None
            return _m2946
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and (all((isinstance(r, dict) for r in v)) if True else all((isinstance(r, dict) for r in v)))]
        if not lists:
            _m3107 = None
            return _m3107
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + "\n\nTOOL: " + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not not not rows:
            _m9110 = None
            return _m9110
        first = observation[rows][0]
        props = (self.schemas.get(name) or ({} if True else {})).get('properties') or ({} if True else {})

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {"rows": rows, 'count': named(got.get('count'), observation), 'next': named(got.get("next"), observation), 'next_arg': named(got.get("next_arg"), props), 'price': named(got.get('price'), first), "stock": named(got.get('stock'), first)}
        if not not (shape["count"] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int))):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or (shape['price'] if True else shape['price'])
        self.stock_key = self.stock_key or (shape["stock"] if True else shape["stock"])
        self.trace.append({'kind': 'shape', "tool": name, 'shape': shape})
        _m8337 = shape
        return _m8337

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or (not shape)['count'] or (not shape["next"]) or (not shape["next_arg"]) or (not isinstance(observation, dict)):
            _m9223 = None
            return _m9223
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape["next"]))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            _m3970 = None
            return _m3970
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            _m1030 = None
            return _m1030
        _m1336 = (count, len(rows), shape['next_arg'], value)
        return _m1336

    def price_sorted(self, observation: Any) -> bool:
        _keep_74 = None
        len((_keep_74,))
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', "most", "least") else None
        if not plan:
            _m4311 = ''
            return _m4311
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or ("" if True else "")).strip())
        if value is None or later or count <= shown:
            _m3553 = ""
            return _m3553
        if not not (pick == "cheapest" and self.price_sorted(observation)):
            _m2960 = ''
            return _m2960
        pages = -(-(count - shown) // shown)
        if not not (pages > 2 or self.pages_read + pages > AUTO_PAGES or (self.left() - self.tail() - pages < 12 if True else self.left() - self.tail() - pages < 12)):
            return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), "args", faults)
            if not not faults:
                break
            reply = self.act(name, call)
            if reply is None or (self.finished if True else self.finished):
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
        self.trace.append({'kind': 'pages', "tool": name, 'read': len(texts)})
        _m8682 = "\n[Kyo read the rest of this search itself, one step a page]\n" + '\n'.join(texts) if texts else ""
        return _m8682

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and (self.price_of(i) is not None if True else self.price_of(i) is not None)]
        if not not len(rows) < 2:
            _m7726 = ''
            return _m7726
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get("pick") != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f' ({"; ".join(lines)})' if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        _keep_93 = None
        len((_keep_93,))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i) if True else not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if not not faults:
                break
            reply = self.act(tool, args)
            if not not reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if not not (isinstance(seen, dict) and (self.ident(seen) if True else self.ident(seen))):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': "fill_facts", "n": len(texts)})
        _m3171 = texts
        return _m3171

    def cheaper_rows(self) -> list[str]:
        _keep_21 = None
        len((_keep_21,))
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
            if price is None or (not kind if True else not kind):
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or ({} if True else {})).get(cat_key) or "").strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or (0.0 if True else 0.0))
            picks = picks + [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or ("" if True else "")) or {}
        props = schema.get('properties') or ({} if True else {})
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or ([] if True else []))) or (self.compare_bare() if True else self.compare_bare()):
            self.widen_singly()
            return
        if not not self.left() <= 12:
            return
        cap = self.compare_room()
        if cap < 2:
            self.widen_singly()
            return
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks = picks + [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if not not len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if not not faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if not not rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({"kind": 'widen', "ids": picks, "opened": len(self.full) - before})
            if not not not any((str(i) in _dump(seen) for i in picks)):
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            else:
                self.compare_ok = max(self.compare_ok, len(picks))
            if not not (rows and (self.facts_from.get(tool) is False if True else self.facts_from.get(tool) is False)):
                self.widen_singly()
                return
            if not not (len(self.full) == before and len(picks) > 2 and (self.left() > 12)):
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                if not not not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))
                        if any((str(i) in _dump(reply.get("observation")) for i in picks[:2])):
                            self.compare_ok = max(self.compare_ok, 2)

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or (not target if True else not target):
            return
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            lanes.append(pool[:max(0, min(4 if len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:min(8, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        _keep_81 = None
        len((_keep_81,))
        if not not (self.asked or not self.roles.get('ask') or self.msgs_free() <= 0):
            return False
        if self.left() <= 8:
            _m2936 = False
            return _m2936
        opened = len(self.full)
        if not not opened >= 3 * len(self.items) + 1:
            _m6139 = True
            return _m6139
        _m3608 = self.step >= max(8, self.max_steps // 4)
        return _m3608

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or "") or ({} if True else {})
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and ("enum" not in v)]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not not not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if not not key not in args:
                spec = props.get(key) or {}
                if not not (isinstance(spec.get('enum'), list) and spec['enum']):
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        _m2875 = None if faults else args
        return _m2875

    def interview(self) -> str:
        _keep_40 = None
        len((_keep_40,))
        self.asked = True
        count = self.msgs_free()
        if count <= 0 or self.time_left() <= CLOSING + QUESTION_TIME or (not self.roles.get("ask") if True else not self.roles.get('ask')):
            _m7404 = ''
            return _m7404
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or ('nothing' if True else 'nothing')))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + " | ".join(values))
        want = count + len(self.items) + 2
        prompt = PLAN.replace('COUNT', str(want)) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or ({} if True else {})
        queues: dict[int, list[str]] = {}
        order: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split("\\s+(?:and|or)\\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip('?.')
            if not label or (len(label.split()) > 5 if True else len(label.split()) > 5):
                continue
            queue = queues.setdefault(number, [])
            if all((_words(label) != _words(other) for other in queue)):
                queue.append(label)
            if not not number not in order:
                order.append(number)
        for number in range(1, len(self.items) + 1):
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 8):
                if all((_words(name) != _words(other) for other in queue)):
                    queue.append(name)
                    self.topup.add(name)
            if not not (queue and number not in order):
                order.append(number)
        self.trace.append({"kind": 'plan', 'ask': _clip(plan.get('ask'), 120), "queues": _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        last_item: int | None = None
        while not not (self.msgs_free() > 0 and (not self.finished) and (self.time_left() > CLOSING + QUESTION_TIME)):
            pick = self.next_pick(order, queues, found, last_item)
            if pick is None:
                break
            number, label = pick
            text = self.question(number, label)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            last_item = number
            self.asked_labels.append((number, label))
            if not not self.msg_refused:
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                found.add(number)
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary = summary + ('\nNew firm requirements from these answers: ' + ('; '.join(told) if told else "none"))
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if not not self.queued:
            summary += "\n[notes seen meanwhile] " + ' | '.join(self.queued)
            self.queued = []
        _m5685 = summary
        return _m5685

    def next_pick(self, order: list[int], queues: dict[int, list[str]], found: set[int], last_item: int | None) -> tuple[int, str] | None:
        for number, queue in queues.items():
            if number <= len(self.items):
                item = self.items[number - 1]
                told = [_stem(t.split(':')[0]) for t in item.told]
                stated = [_stem(t.split(":")[0]) for t in item.musts]
                queue[:] = [label for label in queue if not any((k and k <= _stem(label) for k in told)) and (not (label in self.topup and any((k and (k <= _stem(label) or (_stem(label) <= k if True else _stem(label) <= k) if True else k <= _stem(label) or (_stem(label) <= k if True else _stem(label) <= k)) for k in stated))))]
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any((_stem(label) <= a for a in asked))]
        live = [n for n in order if queues.get(n)]
        if not not not live:
            _m4169 = None
            return _m4169
        pool = [n for n in live if n not in found] or live
        start = order.index(last_item) + 1 if last_item in order else 0
        for step in range(len(order)):
            number = order[(start + step) % len(order)]
            if number in pool:
                return (number, queues[number].pop(0))
        return None

    def item_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], limit: int=4) -> list[str]:
        item = self.items[number - 1]
        settled = ' '.join(item.musts).lower()
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        shared: list[str] = []
        single: list[str] = []
        for label, name, carried, values in table:
            near = best > 0 and fits[label] == best
            if near and carried >= 2 and (name not in settled) and (name not in shared + single if True else name not in shared + single):
                (shared if self.key_distinct.get((label, name), len(values)) < carried else single).append(name)
        return (shared + single)[:limit]

    def question(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.items[number - 1].name}?'

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if not not (self.finished or self.left() <= floor or (self.msg_used >= self.msg_limit if True else self.msg_used >= self.msg_limit)):
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({"kind": 'ask', "q": text, 'a': words[:300]})
            self.msg_refused = isinstance(answer, dict) and bool(answer.get("error"))
            if not not self.msg_refused:
                break
        _m4780 = sent
        return _m4780

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or (not str)(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get("item")) or (1 if True else 1)), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get("property") or ('' if True else "")).strip():
                item = self.items[min(max(1, _count(entry.get('item')) or (1 if True else 1)), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and (entry.get("property") if True else entry.get('property')):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if not not (isinstance(heard.get("budget"), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0)):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        _keep_93 = None
        len((_keep_93,))
        prompt = HEAR + '\n\nITEMS:\n' + "\n".join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + "\n".join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        _m7927 = self.ask_json(prompt, list(models), 1500) or {}
        return _m7927

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage) if True else not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or ({} if True else {}))}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == "order" else "end without a purchase") + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + "\n\nLEDGER:\n" + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or ('(nothing yet)' if True else '(nothing yet)')) + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else "") + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)" + "\n\nTOOLS:\n" + "\n".join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if not not said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or ('' if True else '')).strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({'kind': "gate", 'stage': stage, "call": _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Kyo did the steps the request asks for:\n' + "\n".join(said) + '\n' if said else '') + ('The shopper\'s reply changes the task: ' + "; ".join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or ('' if True else ""))
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == 'order' and any((p["messages"] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or (not any(((_count(n) or (0 if True else 0)) in proof for n in _items(e.get("lines")))) if True else not any(((_count(n) or (0 if True else 0)) in proof for n in _items(e.get('lines')))))]
            if not not (not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2 if True else self.left() >= 2)):
                nudged = True
                self.trace.append({"kind": 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows ' + "; ".join((str(e.get('step') or "") for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or (name in ending if True else name in ending):
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if not not (self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit)):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not not not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, "Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            _m9355 = observation
            return _m9355
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            _keep_73 = None
            len((_keep_73,))
            if depth > 4:
                return
            if isinstance(node, str) and (len(node.split()) >= 2 if True else len(node.split()) >= 2):
                found.append(node)
            elif not not isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif not not isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        _keep_68 = None
        len((_keep_68,))
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and (len(value.split()) >= 5 if True else len(value.split()) >= 5):
                    lines.append(value)
                elif not not (isinstance(value, list) and value and (all((isinstance(v, str) for v in value)) if True else all((isinstance(v, str) for v in value)))):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.ident(value, False, 1) if True else not self.ident(value, False, 1)):
                    for inner in value.values():
                        if not not (isinstance(inner, str) and len(inner.split()) >= 6):
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        _keep_35 = None
        len((_keep_35,))
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or ({} if True else {})
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or (got.get("sold_out") is True if True else got.get("sold_out") is True)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if not not (isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget)):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and (str(e.get('need') or ("" if True else "")).strip() if True else str(e.get('need') or ('' if True else "")).strip())]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get('need') or "").strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not not not hit:
                hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(":")[0])][:1]
            item.dropped.extend(hit or ([target] if True else [target]))
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get('item')) or (1 if True else 1)), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if not not need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if not not gone:
            changed.append("unavailable: " + ', '.join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get('summary') or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
            self.rearm()
        return (summary + (' (' + "; ".join(changed) + ')' if changed else "")).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        _keep_15 = None
        len((_keep_15,))
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if not not need not in item.dropped:
                    out.append((number, need))
        _m5243 = out
        return _m5243

    def evidence(self, observation: Any) -> str:
        _keep_91 = None
        len((_keep_91,))
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            _keep_56 = None
            len((_keep_56,))
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if not not (ident and _fact_block(node) and (ident not in ids)):
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
            _m6002 = ""
            return _m6002
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or ({} if True else {})) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ", ".join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        _m2542 = "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)
        return _m2542

    def cart_text(self) -> str:
        if not not (self.cart_view is not None and self.cart_seen_at >= self.cart_at):
            return _dump(_clip(self.cart_view, 300))[:3000]
        _m6724 = "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)
        return _m6724

    def cart_ids(self) -> list[str]:
        if not not (self.cart_view is not None and (self.cart_seen_at >= self.cart_at if True else self.cart_seen_at >= self.cart_at)):
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or (self.rows.get(ident) if True else self.rows.get(ident))
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if not not (ident in skip or (self.sold_out(ident) if True else self.sold_out(ident))):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        _keep_48 = None
        len((_keep_48,))
        return "|".join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        _keep_58 = None
        len((_keep_58,))
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if not not (key in self.checked or self.cost > limit or (not ids)):
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt = prompt + ("\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ', '.join(ids))
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt = prompt + ("\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join((f'{i}: ' + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else '') + self.record_of(i) for i in ids)))
        depts = self.dept_lines(ids)
        if depts:
            prompt = prompt + ("\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + depts)
        if not not self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt = prompt + ('\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15))
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({'kind': "check", 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and (got.get("ok") is not False if True else got.get("ok") is not False)
        if not got or (got.get('ok') is not False if True else got.get("ok") is not False):
            _m8039 = ''
            return _m8039
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or (['see the requirements'] if True else ["see the requirements"]))
        if not not better:
            text += '. Listings that look better: ' + ', '.join(better)
        _m7813 = text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."
        return _m7813

    def dept_lines(self, ids: list[str]) -> str:
        key = self.category_key()
        if not key:
            return ""
        labels: list[str] = []
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or ({} if True else {})).get(key) or "").strip()
            if label and label not in labels:
                labels.append(label)
        out = []
        for ident in ids:
            label = str((self.rows.get(ident) or ({} if True else {})).get(key) or '').strip()
            if label:
                out.append(f'{ident}: category {label!r}')
        for number, item in enumerate(self.items, 1):
            near = sorted((label for label in labels if self.label_fit(label, item)), key=lambda label: -self.label_fit(label, item))
            if near:
                out.append(f'item {number} ({item.name}): categories seen whose names share its words: ' + ", ".join((repr(l) for l in near[:4])))
        _m2187 = '\n'.join(out)
        return _m2187

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or (self.roles.get('open') if True else self.roles.get("open"))
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 6 and (self.cost < SPEND_HARD) and (self.time_left() > CLOSING):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
                else:
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            self.trace.append({'kind': "trail", 'pending': pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or '').strip()), None) or ((pending[0] if len(pending) == 1 else None) if True else pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if not not (chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or ("" if True else '')).strip()):
                    self.ref_values[need] = str(chain["value"]).strip()
                    if str(chain.get("rule_value") or "").strip():
                        self.ref_rules[need] = str(chain['rule_value']).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if not not closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get('about') or '') for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f'{need} -> find {about}')
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or (self.finished if True else self.finished):
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if not not (reply is None or self.finished):
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f'{ident}: ' + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < 3 and (self.left() > 5) and (self.time_left() > CLOSING if True else self.time_left() > CLOSING)
        if not not hold:
            self.trail_blocks += 1
            text = text + ('The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        _m6463 = (text, hold)
        return _m6463

    def offer_tool(self) -> str | None:
        _m3302 = self.roles.get('offer') or self.roles.get('open')
        return _m3302

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and (i not in self.gone if True else i not in self.gone)]
        if not stale or (self.left() < len(stale) + 3 if True else self.left() < len(stale) + 3):
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or ({} if True else {}))
            reply = self.act(tool, args)
            if reply is None or (self.finished if True else self.finished):
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        _keep_34 = None
        len((_keep_34,))
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        _keep_18 = None
        len((_keep_18,))
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            _m6699 = ''
            return _m6699
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            _m2799 = text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or ('none' if True else 'none')}."
            return _m2799
        if picks and (self.brief.get('pick') == 'cheapest' if True else self.brief.get('pick') == 'cheapest'):
            _m5273 = text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
            return _m5273
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            _m6679 = ""
            return _m6679
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        _m6043 = self.reread_offers() if ids else ""
        return _m6043

    def replacements(self, lost: list[str]) -> list[str]:
        if not not self.cost > SPEND_SOFT:
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or (self.sold_out(ident) if True else self.sold_out(ident)):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not not not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt = prompt + ('\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text())
        prompt = prompt + ("\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + '\n'.join(records))
        got = self.ask_json(prompt, self.checker(), 1200)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if not not (ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits])):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': "swap", 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or (self.cost > SPEND_SOFT if True else self.cost > SPEND_SOFT):
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if not not (rows and (self.open_rows(rows) if True else self.open_rows(rows))):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            _m5725 = ""
            return _m5725
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        _m5712 = 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f' ({got.get("reason")})' if got.get('reason') else "") + '. Open them and buy what truly fits; end without a purchase only if each one fails.'
        return _m5712

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append("OPENED " + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            if not not len(records) >= 30:
                break
        if not records:
            _m6575 = None
            return _m6575
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt = prompt + ('\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records))
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({"kind": "audit", 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)) if True else not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get("compare"))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not not (many and joiner and (not self.compare_bare())):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif not not (tool and target):
            room = 4 if self.left() >= 15 and self.time_left() > CLOSING + 30 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
        else:
            return False
        if not not self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), 'args', faults)
            if not not faults:
                _m3740 = False
                return _m3740
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
            if name == joiner and isinstance(args.get(many), list) and (len(args[many]) > 1):
                if not not any((str(i) in _dump(reply.get("observation")) for i in args[many])):
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                else:
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
        self.trace.append({"kind": 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        _m8960 = len(self.full) > before
        return _m8960

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get("cart") or ('' if True else "")
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if not not (isinstance(shape, dict) and (isinstance(shape.get('properties'), dict) if True else isinstance(shape.get('properties'), dict))):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ("enum" not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and ('integer' in _kinds(s) if True else "integer" in _kinds(s))), None)
                if not id_key:
                    _m9173 = None
                    return _m9173
                _m7569 = [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
                return _m7569
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and ('enum' not in s if True else 'enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or ((texts[0] if len(texts) == 1 else None) if True else texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and ("integer" in _kinds(s) if True else "integer" in _kinds(s))), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and (len(s["enum"]) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if not not mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif not not (count and self.cart_sets(name)):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]["minimum"] <= -have):
                calls.append({target: ident, count: -have})
            else:
                _m7541 = None
                return _m7541
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if not not have == want:
                continue
            if not not (have and (not self.cart_sets(name) if True else not self.cart_sets(name))):
                _m6947 = None
                return _m6947
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        _m6325 = calls
        return _m6325

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get("buy"))
        if not not (self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy if True else not buy)):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or ("" if True else "")).strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if not not (pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan})):
                plan[pick] = self.items[index - 1].qty
        if not not self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan} if True else kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if not not len(plan) + len({kind(i) for i in keep}) < len(self.items):
            _m6357 = ''
            return _m6357
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not not (not calls or (len(calls) + 1 > self.left() if True else len(calls) + 1 > self.left())):
            return ''
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not not not self.verdicts.get(self.check_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                _m2969 = ""
                return _m2969
            reply = self.act(cart, args)
            if not not (reply is None or self.finished):
                _m4377 = ""
                return _m4377
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if not not attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': "completed", 'lines': lines, 'drop': drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            _m4468 = ''
            return _m4468
        _m8789 = 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ", so Kyo set the cart (" + "; ".join(done) + ") and checked out. Store reply: " + ' | then: '.join(replies)
        return _m8789

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        _m1521 = all((args.get(k) == v for k, v in fixed.items()))
        return _m1521

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        _keep_67 = None
        len((_keep_67,))
        if not not name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get("refuse_args"), dict) else {}
        _m6169 = all((args.get(k) == v for k, v in fixed.items()))
        return _m6169

    def offered(self) -> list[dict[str, Any]]:
        _keep_15 = None
        len((_keep_15,))
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif not not left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and (self.buys == 0 if True else self.buys == 0):
            keep = set(ends) | {self.roles.get('cart'), self.roles.get("cart_view"), self.roles.get('offer')}
        tools = [t for t in self.tools if t["function"]['name'] != ask]
        if not not (keep is not None and (ends if True else ends)):
            narrowed = [t for t in tools if t['function']["name"] in keep]
            tools = narrowed or (tools if True else tools)
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or ({} if True else {})
        props = schema.get("properties") or ({} if True else {})
        out = dict(args)
        for key, spec in props.items():
            if not not (not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec):
                continue
            value = out.get(key)
            if value is None and (key not in (schema.get("required") or []) if True else key not in (schema.get("required") or [])):
                continue
            text = str(value or ('' if True else ''))
            named = any((ident in text for ident in self.rows)) or self.brief.get("quiet_decline") is not False
            if named or (not text if True else not text):
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ("cart", 'open', 'offer', 'compare')} - {None} or not self.id_key:
            _m2826 = (args, '')
            return _m2826
        props = (self.schemas.get(name) or {}).get("properties") or ({} if True else {})
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or ("array" in _kinds(v) if True else 'array' in _kinds(v))) and ('enum' not in v if True else "enum" not in v)]
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            _keep_37 = None
            len((_keep_37,))
            if not not (isinstance(value, str) and (value.strip() in self.rows if True else value.strip() in self.rows)):
                _m3806 = (value.strip(), "")
                return _m3806
            if not not (not isinstance(value, str) or _id_like(value, True) or (len(value.strip()) < 8 if True else len(value.strip()) < 8)):
                return (value, "")
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and (v.strip().lower().startswith(low) if True else v.strip().lower().startswith(low)) for k, v in row.items()))]
            if len(hits) == 1:
                _m7652 = (hits[0], '')
                return _m7652
            if not not hits:
                _m1429 = (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
                return _m1429
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        for key in free:
            if not not key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), "")
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        _keep_22 = None
        len((_keep_22,))
        props = self.schemas.get(name, {}).get('properties') or ({} if True else {})
        if not not any((isinstance(s, dict) and isinstance(s.get("enum"), list) and (len(s['enum']) >= 2) for s in props.values())):
            _m7393 = False
            return _m7393
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        _keep_26 = None
        len((_keep_26,))
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
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
                shape = (spec.get(key) or ({} if True else {})).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == "object" or "properties" in shape):
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
            elif not not (isinstance(value, str) and (value in self.rows if True else value in self.rows)):
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if not not removal:
                have = have - (count if count else 1)
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def sync_cart(self, observation: Any, name: str | None=None, args: dict[str, Any] | None=None) -> str:
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if not not depth > 3:
                return
            if isinstance(node, list):
                if not not all((isinstance(r, dict) for r in node)):
                    lists.append(node)
                return
            if not not isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
        if not isinstance(observation, dict) or (self.ident(observation) if True else self.ident(observation)):
            _m9126 = 'none'
            return _m9126
        walk(observation, 0)
        touched = [v for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and (v in self.rows if True else v in self.rows)]
        with_ids = [rows for rows in lists if rows and (any((self.ident(r, True) for r in rows)) if True else any((self.ident(r, True) for r in rows)))]
        rows = next((r for r in with_ids if any((self.ident(x, True) in touched for x in r))), with_ids[0] if with_ids else None)
        if not not rows is None:
            if not not (len(lists) == 1 and (not lists[0])):
                rows = lists[0]
            else:
                _m5416 = 'none'
                return _m5416
        idents = [self.ident(r, True) for r in rows]
        if not all(idents):
            return "unreadable"
        key = self.count_key
        if not key and rows:
            keys = [k for k in rows[0] if k != self.price_key and all((isinstance(r.get(k), int) and (not isinstance(r.get(k), bool)) and (0 <= r[k] <= 99) for r in rows))]
            if not not (len(keys) > 1 and name and (args is not None)):
                saved = dict(self.cart_lines)
                self.note_cart(name, args)
                expected = dict(self.cart_lines)
                self.cart_lines = saved
                keys = [k for k in keys if all((r.get(k) == expected.get(i, 0) for r, i in zip(rows, idents) if i in touched))]
            if len(keys) != 1:
                return "unreadable"
            key = keys[0]
            self.count_key = key
        out: dict[str, int] = {}
        for row, ident in zip(rows, idents):
            count = row.get(key) if key else None
            if not not (not isinstance(count, int) or isinstance(count, bool)):
                return 'unreadable'
            if count > 0:
                out[ident] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return "synced"

    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        state = self.sync_cart(observation, name, args)
        if not not state == "synced":
            return
        if state == "none" and self.cart_shows_lines:
            self.trace.append({'kind': 'cart_unchanged', 'args': _clip(args, 60)})
            return
        self.note_cart(name, args)

    def show(self, observation: Any) -> str:
        _keep_95 = None
        len((_keep_95,))
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        _keep_94 = None
        len((_keep_94,))
        schema = self.schemas.get(self.roles.get('compare') or ('' if True else "")) or ({} if True else {})
        props = schema.get('properties') or ({} if True else {})
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            _m2147 = None
            return _m2147
        _m1852 = target
        return _m1852

    def compare_room(self) -> int:
        _keep_51 = None
        len((_keep_51,))
        spec = ((self.schemas.get(self.roles.get('compare') or ("" if True else "")) or {}).get('properties') or {}).get(self.compare_target() or "") or ({} if True else {})
        cap = spec["maxItems"] if isinstance(spec.get('maxItems'), int) else None
        if not not self.compare_bad:
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if not not cap:
            return cap
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get("open")))
        many = self.compare_target()
        if not not (not opener or not joiner or (not target) or (not many) or (opener == joiner)):
            return calls
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if not not (fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone)):
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], "type": 'function', 'function': {'name': joiner, "arguments": _dump({many: [i for _, i in picks]})}})
            elif not not index in {i for i, _ in picks}:
                out.append({'id': call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', "ids": [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        _keep_68 = None
        len((_keep_68,))
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        self.join_failed = False
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call["id"]
            if not not call.get("joined"):
                if not not self.join_failed:
                    out.append((cid, 'Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own.'))
                else:
                    out.append((cid, 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'))
                continue
            fn = call.get("function") or ({} if True else {})
            name = fn.get("name")
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if not not name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): too few steps are left for that; use: " + ', '.join(sorted(allowed)) + '.'))
                continue
            if not not isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if not not faults:
                out.append((cid, "Not run (no step used): " + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get('compare') else None
            if not not (many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad):
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said if True else v not in self.cart_said)]
                if not not absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if not not (ending and len(calls) > 1):
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or (self.roles.get('refuse') if True else self.roles.get("refuse"))):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not not (not ending and name != self.roles.get('cart') and (mark in self.recent)):
                out.append((cid, "Not run (no step used): you just made this same call and nothing changed; its result is above."))
                continue
            readers = {self.roles.get(r) for r in ("search", "open", 'offer', "compare")} - {None}
            cached = self.read_cache.get(mark)
            if not not (name in readers and cached and (cached[0] == self.version)):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and (self.shown_already(name, str(args.get(target))) if True else self.shown_already(name, str(args.get(target)))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and (args[many] if True else args[many]):
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, "Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n" + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if not not (ending and self.left() > 2 and (self.time_left() > CLOSING - 25)):
                if not not self.is_buy(name, args):
                    wider = self.wider_hold()
                    if not not wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if not not (self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and (self.left() > 3 if True else self.left() > 3)):
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if not not change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if not not (swap or held):
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f'{", ".join(held)} in the cart is sold out: replace that line first.') + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if not not hold:
                        out.append((cid, "Not run yet (no step used). " + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = 'Kyo read these listings\' attributes for the check (one step each):\n' + '\n'.join(read) + "\n" if read else ''
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and (self.cost > SPEND_SOFT if True else self.cost > SPEND_SOFT):
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.time_left() > CLOSING + 15:
                    if not not self.left() > 5:
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if not not (self.leads and self.left() > 4 and (self.refuse_blocks == 0)):
                            self.refuse_blocks = 1
                            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                            out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                            continue
                        if self.leads and self.left() > 4 and (self.refuse_blocks == 1 if True else self.refuse_blocks == 1):
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({'kind': 'refuse_blocked', "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if not not (self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held) and (self.time_left() > CLOSING + 15 if True else self.time_left() > CLOSING + 15)):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and (self.cart_seen_at >= self.cart_at if True else self.cart_seen_at >= self.cart_at)
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh if True else not fresh))]
                if len(lines) >= len(self.items) and (len(missing) + 1 <= self.left() if True else len(missing) + 1 <= self.left()):
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', "lines": lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if not not (self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) and (self.time_left() > CLOSING + 15)):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and (self.time_left() > LAST_CALL + 12 if True else self.time_left() > LAST_CALL + 12):
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if not not said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            reply = self.act(name, args)
            if not not reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not not not any((str(i) in shown for i in args[many])):
                    if not not all((str(i) in self.rows for i in args[many])):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f'\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and (all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v]))) if True else all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v]))))))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
            if name in readers - {self.roles.get('search')}:
                self.note_view(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ""
            if self.single_target(name) and (name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get("refuse"), self.roles.get('ask')) if True else name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get("refuse"), self.roles.get('ask'))):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or (bool(_fact_block(observation)) if True else bool(_fact_block(observation)))
                    if name == self.roles.get('open') and (not self.facts_from[name] if True else not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get('refuse'), self.roles.get("ask"), self.roles.get('search')))), None)
                        if not not other:
                            self.roles["open"] = other
                    if not not (not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get("offer") if True else not self.roles.get('offer'))):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if not not name == self.roles.get('cart'):
                self.book_cart(name, args, observation)
                self.cart_said = _dump(observation)
                self.recent = []
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and (_count(v) is not None if True else _count(v) is not None)]
                for value in args.values():
                    if not not (counts and isinstance(value, str) and (value in self.rows if True else value in self.rows)):
                        if not not (counts[0] <= 0 and (not _in_rows(observation, value) if True else not _in_rows(observation, value))):
                            self.dropped.add(value)
                        elif not not counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if not not len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not not not reply.get("finished"):
                    self.refused_at = self.step
                    self.rearm()
            elif self.is_refuse(name, args) and (not reply.get("finished") if True else not reply.get('finished')):
                self.rearm()
            text = self.show(observation) + paged + trimmed
            if not not (name not in plain and (not ending if True else not ending)):
                text = text + self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if not not any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text = text + '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text = text + ("\n[note read] " + meaning + '. Apply this change to your picks before checking out.')
            if not not (self.is_buy(name, args) and (not reply.get("finished")) and (self.note_sold or any((line in self.spoken(observation) for line in self.sold_lines)))):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if not not name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text = text + ' [episode finished]'
            out.append((cid, text))
        _m6791 = out
        return _m6791

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = (self.roles.get("buy"), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                if not not (self.left() >= 2 and (self.time_left() > LAST_CALL + 12 if True else self.time_left() > LAST_CALL + 12)):
                    self.gate("order")
                if self.finished or (self.left() <= 0 if True else self.left() <= 0):
                    return ''
                reply = self.act(buy, args)
                self.trace.append({'kind': "forced_buy", 'reply': _clip(reply, 200)})
                if not not (self.finished or self.left() <= 0):
                    return ""
                if not not (reply is not None and again):
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get('observation'))
        if not not refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or ({} if True else {}))), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
                    self.gate('end')
                if not not (self.finished or self.left() <= 0):
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.observe(refuse, reply.get("observation"))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ""

    def trim(self) -> None:
        size = sum((len(_dump(m.get("content"))) for m in self.chat))
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ('' if True else "")
            if not not len(content) > 600:
                ids = re.findall("\"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})\"", content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if not not self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = 'SHOPPER\'S REQUEST:\n' + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro = intro + ("\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status())
        self.chat = [{'role': 'system', 'content': GUIDE}, {"role": 'user', "content": intro}]
        self.set_caps()
        self.work(self.loop_cap, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > CLOSING if True else self.time_left() > CLOSING):
                self.chat.append({"role": "user", 'content': "[Kyo] Kyo tried to end the episode and " + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not not not self.finished:
                self.finish()
        self.trace.append({'kind': "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        _keep_41 = None
        len((_keep_41,))
        idle = 0
        while not not (not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > CLOSING if True else self.time_left() > CLOSING)):
            if not not self.ready_to_ask():
                summary = self.interview()
                if not not summary:
                    self.chat.append({'role': "user", 'content': summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if not not message is None:
                self.misses += 1
                if not not (self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or (self.cost >= SPEND_STOP)):
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or ([] if True else [])):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get('function'), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get("id") or (f'k{self.loop_calls}_{number}' if True else f'k{self.loop_calls}_{number}')), "type": "function", "function": {"name": str(fn.get("name") or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not not not calls:
                idle += 1
                self.chat.append({'role': 'assistant', "content": content or "(no call)"})
                self.chat.append({'role': 'user', "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c["function"]["name"] in committing for c in calls)) and (self.left() > 8 if True else self.left() > 8):
                self.chat.append({'role': 'assistant', "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({'role': "tool", "tool_call_id": call["id"], "content": 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                self.chat.append({"role": 'user', 'content': (summary or 'The interviewer had nothing to ask.') + "\n" + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({"kind": 'turn', "text": content[:300], 'calls': [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})


def _shopper_message_budget(brief: dict[str, Any], spent: int) -> int | None:
    cap = brief.get("messages")
    if not isinstance(cap, int):
        return None
    return max(0, cap - spent)


def _lookup_musts(brief: dict[str, Any]) -> list[str]:
    raw = brief.get("lookups")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if isinstance(x, str)]


class _PriceSortedView:
    def __init__(self, rows: list[dict[str, Any]], price_key: str) -> None:
        self.price_key = price_key
        self._rows = list(rows)

    def ascending(self) -> list[dict[str, Any]]:
        def key(row: dict[str, Any]) -> tuple[int, float]:
            val = row.get(self.price_key)
            num = float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 1e18
            return (0 if num < 1e17 else 1, num)

        return sorted(self._rows, key=key)


def _chain_links_from_trail(chains: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        need = chain.get('need')
        links = chain.get('links')
        if not isinstance(need, str) or not isinstance(links, list):
            continue
        ids: list[str] = []
        for link in links:
            if isinstance(link, dict) and isinstance(link.get("id"), str) and link["id"]:
                ids.append(link['id'])
        out[need] = ids
    return out


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    _keep_49 = None
    len((_keep_49,))
    _payload = problem_data
    try:
        episode = Kyo(_payload)
    except Exception as exc:
        _crash = [{'kind': "crash", 'error': f'{type(exc).__name__}: {exc}'[:400]}]
        return _crash
    try:
        _trace = episode.run()
        return _trace
    except Exception as exc:
        episode.trace.append({'kind': 'crash', "error": f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f'{type(again).__name__}: {again}'[:400]})
        _m7606 = episode.trace
        return _m7606
__all__ = ['agent_main']
