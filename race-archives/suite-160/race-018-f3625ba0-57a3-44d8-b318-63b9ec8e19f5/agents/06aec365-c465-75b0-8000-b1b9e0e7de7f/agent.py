"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
_LUCIOUS_END = object()

class _LuciousElse(Exception):
    pass

def _lucious_miss():
    raise _LuciousElse

def _lucious_ok():
    return None

def _lucious_take(flag):
    (_lucious_miss, _lucious_ok)[int(bool(flag))]()
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_STOP = 0.085
SPEND_LIGHT = 0.03
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
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words is not on the listings: drop words (keep the product noun) or browse the item\'s category instead; no rows never shows that the store lacks the item. When a keyword search matches every word and returns too many rows to see, put all the item\'s required values in one query to narrow it. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Some details (what other products an item suits, codes it carries) may appear only in a listing\'s description, which a keyword search may not read: browse the item\'s category, sliced by brand or a price ceiling when the store allows, and read the rows\' descriptions.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants of the same word are fine, and a value that contains all of the required value\'s words counts (a value "<A> <B>" meets a requirement "<B>"). A value with a different word does not count, even a word that means nearly the same: no synonyms and no near kinds (an abbreviation or code is not the word it stands for, and sharing one word is not enough). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list attribute that includes it) or in the title or description counts. A number of another kind does not (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value, that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value (the same value as, a value unlike, or a value near the one on another listing that the request describes). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. Open every page of that chain before checking out. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule exactly as the page states it.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it (a request that gives "<value>" as the "<property>" gives "<property>: <value>"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request\'s own words ("<property>: <relation> the <property> of the <kind> <shop> sells with <feature>", where <relation> is the request\'s own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in the request\'s own words, or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it as a must of that item in the request\'s own words ("<property>: <relation> the <other item> in this order") and leave it out of lookups.\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them, ahead of every other key.\nName a hinted property by a listed key of that item when one fits the hint, else by its plain name.\n2. Then the listed keys in the order given (most carried first); avoid keys seen on only one or two of the item\'s listings.\nSkip a property the request already settles for that item, a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property\'s name, no example values, no brackets, no second property.\nChoose COUNT, best first (fewer only when the lists hold fewer usable keys). Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least <number> <unit>", "<value>", "not <value>"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least <number> <noun>", not "at least <number>").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value containing all of the required value\'s words counts, a different word does not count even when it means nearly the same, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nCategory: when the CATEGORIES section shows a store category whose name carries the item\'s own words while a cart line\'s category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings\' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = "the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
LAST_PICK = 'A shopping agent is about to end without buying, and that earns nothing unless no listing of a requested kind can be bought at all. Choose what to order instead, with these reading rules: ' + READING + '\nFor each requested item take the listing of that kind, in stock and inside the budget, that meets the most requirements. A record that does not state a property may still meet it; a record that plainly contradicts a requirement counts against that listing. Among equally good listings follow the request\'s pick rule. The chosen listings together must stay inside the budget.\nReturn ONE JSON object and nothing else: {"lines": [{"item": item number, "id": listing id}], "reason": "one sentence"}, with "lines": [] only when some requested item has no listing of its kind in stock inside the budget at all.'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property\'s value itself, else false, "value": the property\'s own value as stated on that last page, or "", "rule_value": when that page states a rule that gives a value for this requirement, the value the rule gives, else ""}]}\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    try:
        _lucious_take(depth > 7)
    except _LuciousElse:
        pass
    else:
        return '...'
    try:
        _lucious_take(isinstance(value, str))
    except _LuciousElse:
        pass
    else:
        return value if len(value) <= limit else value[:limit] + '...'
    try:
        _lucious_take(isinstance(value, dict))
    except _LuciousElse:
        pass
    else:
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    try:
        _lucious_take(isinstance(value, list))
    except _LuciousElse:
        pass
    else:
        return [_clip(v, limit, depth + 1) for v in value]
    return value

def _json_obj(text: Any) -> dict[str, Any] | None:
    try:
        _lucious_take(not isinstance(text, str))
    except _LuciousElse:
        pass
    else:
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    try:
        _lucious_take(fence)
    except _LuciousElse:
        pass
    else:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    try:
        _lucious_take(0 <= left < right)
    except _LuciousElse:
        pass
    else:
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

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    try:
        _lucious_take(not prop or not key)
    except _LuciousElse:
        pass
    else:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    try:
        _lucious_take(isinstance(kind, str))
    except _LuciousElse:
        pass
    else:
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    try:
        _lucious_take(kind == 'string')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, str)
    try:
        _lucious_take(kind == 'integer')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, int) and (not isinstance(value, bool))
    try:
        _lucious_take(kind == 'number')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    try:
        _lucious_take(kind == 'boolean')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, bool)
    try:
        _lucious_take(kind == 'array')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, list)
    try:
        _lucious_take(kind == 'object')
    except _LuciousElse:
        pass
    else:
        return isinstance(value, dict)
    try:
        _lucious_take(kind == 'null')
    except _LuciousElse:
        pass
    else:
        return value is None
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; list what cannot be fixed."""
    try:
        _lucious_take(not isinstance(schema, dict))
    except _LuciousElse:
        pass
    else:
        return value
    kinds = _kinds(schema)
    try:
        _lucious_take(kinds and (not any((_fits_kind(value, k) for k in kinds))))
    except _LuciousElse:
        pass
    else:
        _lucious_it_1 = iter(kinds)
        while True:
            try:
                kind = next(_lucious_it_1)
            except StopIteration:
                break
            try:
                _lucious_take(kind == 'integer' and isinstance(value, float) and value.is_integer())
            except _LuciousElse:
                pass
            else:
                value = int(value)
                break
            try:
                _lucious_take(kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value))
            except _LuciousElse:
                pass
            else:
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            try:
                _lucious_take(kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)))
            except _LuciousElse:
                pass
            else:
                value = str(value)
                break
            try:
                _lucious_take(kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')))
            except _LuciousElse:
                pass
            else:
                value = value.strip().lower() == 'true'
                break
            try:
                _lucious_take(kind == 'array' and (not isinstance(value, list)) and (value is not None))
            except _LuciousElse:
                pass
            else:
                value = [value]
                break
        try:
            _lucious_take(not any((_fits_kind(value, k) for k in kinds)))
        except _LuciousElse:
            pass
        else:
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    try:
        _lucious_take(isinstance(options, list) and value not in options)
    except _LuciousElse:
        pass
    else:
        try:
            _lucious_take(isinstance(value, str))
        except _LuciousElse:
            pass
        else:
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            try:
                _lucious_take(match)
            except _LuciousElse:
                pass
            else:
                value = match[0]
        try:
            _lucious_take(value not in options)
        except _LuciousElse:
            pass
        else:
            faults.append(f'{where}: allowed values are {options}')
    try:
        _lucious_take(isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']))
    except _LuciousElse:
        pass
    else:
        value = value[:schema['maxLength']]
    try:
        _lucious_take(isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']))
    except _LuciousElse:
        pass
    else:
        faults.append(f'{where} is too short')
    try:
        _lucious_take(isinstance(value, (int, float)) and (not isinstance(value, bool)))
    except _LuciousElse:
        pass
    else:
        try:
            _lucious_take(isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum'])
        except _LuciousElse:
            pass
        else:
            faults.append(f"{where} is below {schema['minimum']}")
        try:
            _lucious_take(isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum'])
        except _LuciousElse:
            pass
        else:
            faults.append(f"{where} is above {schema['maximum']}")
    try:
        _lucious_take(isinstance(value, list))
    except _LuciousElse:
        pass
    else:
        try:
            _lucious_take(isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems'])
        except _LuciousElse:
            pass
        else:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        try:
            _lucious_take(isinstance(schema.get('minItems'), int) and len(value) < schema['minItems'])
        except _LuciousElse:
            pass
        else:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        try:
            _lucious_take(isinstance(schema.get('items'), dict))
        except _LuciousElse:
            pass
        else:
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    try:
        _lucious_take(isinstance(value, dict))
    except _LuciousElse:
        pass
    else:
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        try:
            _lucious_take(schema.get('additionalProperties') is False)
        except _LuciousElse:
            pass
        else:
            _lucious_it_2 = iter([n for n in value if n not in props])
            while True:
                try:
                    name = next(_lucious_it_2)
                except StopIteration:
                    break
                faults.append(f'{where}.{name} is not an allowed argument')
        _lucious_it_3 = iter(schema.get('required') or [])
        while True:
            try:
                name = next(_lucious_it_3)
            except StopIteration:
                break
            try:
                _lucious_take(name not in value)
            except _LuciousElse:
                pass
            else:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _items(value: Any) -> list[Any]:
    try:
        _lucious_take(isinstance(value, list))
    except _LuciousElse:
        pass
    else:
        return value
    try:
        _lucious_take(isinstance(value, (str, dict)) and value)
    except _LuciousElse:
        pass
    else:
        return [value]
    return []

def _count(value: Any) -> int | None:
    try:
        _lucious_take(isinstance(value, bool))
    except _LuciousElse:
        pass
    else:
        return None
    try:
        _lucious_take(isinstance(value, int))
    except _LuciousElse:
        pass
    else:
        return value
    try:
        _lucious_take(isinstance(value, float) and value.is_integer())
    except _LuciousElse:
        pass
    else:
        return int(value)
    try:
        _lucious_take(isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value))
    except _LuciousElse:
        pass
    else:
        return int(float(value))
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    try:
        _lucious_take(not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip())
    except _LuciousElse:
        pass
    else:
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    _lucious_it_4 = iter(rows[0])
    while True:
        try:
            key = next(_lucious_it_4)
        except StopIteration:
            break
        values = [row.get(key) for row in rows]
        try:
            _lucious_take(not all((_id_like(v, 'id' in str(key).lower()) for v in values)))
        except _LuciousElse:
            pass
        else:
            continue
        values = [v.strip() for v in values]
        try:
            _lucious_take(len(rows) > 1 and len(set(values)) < len(values))
        except _LuciousElse:
            pass
        else:
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        try:
            _lucious_take(best is None or score > best[0])
        except _LuciousElse:
            pass
        else:
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    """Whether a list in the result holds a record carrying this id (a cart listing still showing the line)."""
    try:
        _lucious_take(depth > 5)
    except _LuciousElse:
        pass
    else:
        return False
    try:
        _lucious_take(isinstance(node, list))
    except _LuciousElse:
        pass
    else:
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    try:
        _lucious_take(isinstance(node, dict))
    except _LuciousElse:
        pass
    else:
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    try:
        _lucious_take(not isinstance(node, dict))
    except _LuciousElse:
        pass
    else:
        return None
    best = None
    _lucious_it_5 = iter(node.values())
    while True:
        try:
            value = next(_lucious_it_5)
        except StopIteration:
            break
        try:
            _lucious_take(isinstance(value, dict) and len(value) >= 2)
        except _LuciousElse:
            pass
        else:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            try:
                _lucious_take(len(scalars) >= 2 and (best is None or len(scalars) > len(best)))
            except _LuciousElse:
                pass
            else:
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

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _count(self.view.get('max_steps'))
        try:
            _lucious_take(steps is None and isinstance(self.view.get('max_steps'), float))
        except _LuciousElse:
            pass
        else:
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
        self.picked = False
        self.searched: list[dict[str, Any]] = []
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
        return self.max_steps - self.step

    def time_left(self) -> float:
        """Seconds of Kyo's own budget still left (the network stops a task at 300 s, finished or not)."""
        return DEADLINE - (time.monotonic() - self.t0)

    def checker(self) -> list[str]:
        """Models for a late check (cart check, decline review, sold-out swap, requirement unmixing): the heavy model
        first, the light one first once this task's spend passes SPEND_LIGHT (one run key pays for every task)."""
        return [HEAVY, LIGHT] if self.cost < SPEND_LIGHT else [LIGHT, HEAVY]

    def set_caps(self) -> None:
        """Model turns and calls grow with the task's step cap (long orders have more steps); the clock and the spend
        stop still bound them."""
        self.loop_cap = max(LOOP_CALLS, min(110, self.max_steps))
        self.call_cap = max(ALL_CALLS, self.loop_cap + 25)

    def rearm(self) -> None:
        """A change (new budget, requirement added or dropped, a refused finish) re-opens the one-shot checks: the
        decline review, the search-more push, the late hold and the cheapest widening run again on the new task."""
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools that show a listing's page: the offer view when the store has one apart from the attributes view
        (an attributes list alone is not the page), else the single-listing view; and the side-by-side view."""
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        try:
            _lucious_take(known)
        except _LuciousElse:
            pass
        else:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        _lucious_it_7 = iter(models)
        while True:
            try:
                model = next(_lucious_it_7)
            except StopIteration:
                break
            try:
                _lucious_take(self.calls >= self.call_cap or self.cost >= SPEND_STOP)
            except _LuciousElse:
                pass
            else:
                return None
            try:
                _lucious_take(self.time_left() < LAST_CALL)
            except _LuciousElse:
                pass
            else:
                self.out_of_time = True
                return None
            self.calls += 1
            tries = 0
            for _lucious_tick_6 in iter(lambda: True if True else _LUCIOUS_END, _LUCIOUS_END):
                message, status = self.call_model(model, messages, max_tokens, tools)
                try:
                    _lucious_take(message is not None)
                except _LuciousElse:
                    pass
                else:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                try:
                    _lucious_take(not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL)
                except _LuciousElse:
                    pass
                else:
                    break
                try:
                    _lucious_take(self.time_left() - WAITS[tries] < LAST_CALL)
                except _LuciousElse:
                    pass
                else:
                    self.out_of_time = True
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            try:
                _lucious_take(status == 402)
            except _LuciousElse:
                pass
            else:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        try:
            _lucious_take(model.startswith('anthropic/') and len(messages) > 1)
        except _LuciousElse:
            pass
        else:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            _lucious_it_8 = iter(marks)
            while True:
                try:
                    index = next(_lucious_it_8)
                except StopIteration:
                    break
                content = sent[index].get('content')
                try:
                    _lucious_take(isinstance(content, str) and content)
                except _LuciousElse:
                    pass
                else:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        try:
            _lucious_take(tools)
        except _LuciousElse:
            pass
        else:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        try:
            _lucious_take(isinstance(data, dict) and data.get('choices'))
        except _LuciousElse:
            pass
        else:
            usage = data.get('usage') or {}
            try:
                _lucious_take(isinstance(usage.get('cost'), (int, float)))
            except _LuciousElse:
                pass
            else:
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            try:
                _lucious_take(message.get('content') or message.get('tool_calls'))
            except _LuciousElse:
                pass
            else:
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        try:
            _lucious_take(isinstance(data, dict) and isinstance(data.get('error'), dict))
        except _LuciousElse:
            pass
        else:
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        _lucious_it_9 = iter(models)
        while True:
            try:
                model = next(_lucious_it_9)
            except StopIteration:
                break
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            try:
                _lucious_take(got is not None)
            except _LuciousElse:
                pass
            else:
                return got
            try:
                _lucious_take(self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or (self.cost >= SPEND_STOP))
            except _LuciousElse:
                pass
            else:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        _lucious_it_11 = iter(range(4))
        while True:
            try:
                _ = next(_lucious_it_11)
            except StopIteration:
                break
            result = _env.post_verbose('/environment/call', payload)
            try:
                _lucious_take(isinstance(result.data, dict))
            except _LuciousElse:
                pass
            else:
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                try:
                    _lucious_take(data.get('finished'))
                except _LuciousElse:
                    pass
                else:
                    self.finished = True
                seen = data.get('observation')
                try:
                    _lucious_take(name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))))
                except _LuciousElse:
                    pass
                else:
                    _lucious_it_10 = iter(args.values())
                    while True:
                        try:
                            value = next(_lucious_it_10)
                        except StopIteration:
                            break
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            try:
                _lucious_take(status == 409)
            except _LuciousElse:
                pass
            else:
                low = body.lower()
                try:
                    _lucious_take('finish' in low or 'quarantin' in low or 'finaliz' in low)
                except _LuciousElse:
                    pass
                else:
                    self.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                try:
                    _lucious_take(found)
                except _LuciousElse:
                    pass
                else:
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                try:
                    _lucious_take('reuse' in low or 'different action' in low)
                except _LuciousElse:
                    pass
                else:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            try:
                _lucious_take(status in (400, 401, 402, 403, 404, 422))
            except _LuciousElse:
                pass
            else:
                try:
                    _lucious_take(status == 402)
                except _LuciousElse:
                    pass
                else:
                    self.broke = True
                return None
        return None

    def read_request(self) -> None:
        lines = []
        _lucious_it_12 = iter(self.tools)
        while True:
            try:
                tool = next(_lucious_it_12)
            except StopIteration:
                break
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        try:
            _lucious_take(self.view.get('opening') is not None)
        except _LuciousElse:
            pass
        else:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        try:
            _lucious_take(not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))))
        except _LuciousElse:
            pass
        else:
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        _lucious_it_13 = iter(('buy_args', 'refuse_args'))
        while True:
            try:
                key = next(_lucious_it_13)
            except StopIteration:
                break
            try:
                _lucious_take(not isinstance(brief.get(key), dict))
            except _LuciousElse:
                pass
            else:
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        try:
            _lucious_take(buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']))
        except _LuciousElse:
            pass
        else:
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            try:
                _lucious_take(isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']))
            except _LuciousElse:
                pass
            else:
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        _lucious_it_14 = iter(_items(brief.get('items')))
        while True:
            try:
                entry = next(_lucious_it_14)
            except StopIteration:
                break
            try:
                _lucious_take(not isinstance(entry, dict))
            except _LuciousElse:
                pass
            else:
                continue
            quantity = _count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        try:
            _lucious_take(not self.items)
        except _LuciousElse:
            pass
        else:
            self.items = [Item('the requested product', '', 1, [])]
        _lucious_it_15 = iter(_items(brief.get('lookups')))
        while True:
            try:
                lookup = next(_lucious_it_15)
            except StopIteration:
                break
            text = str(lookup).strip()
            try:
                _lucious_take(not any((text in item.musts for item in self.items)))
            except _LuciousElse:
                pass
            else:
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            try:
                _lucious_take(text and (number, text) not in self.refs)
            except _LuciousElse:
                pass
            else:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        try:
            _lucious_take(isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget))
        except _LuciousElse:
            pass
        else:
            budget = float(budget)
        try:
            _lucious_take(isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0))
        except _LuciousElse:
            pass
        else:
            self.budget = budget
        limit = _count(brief.get('messages'))
        try:
            _lucious_take(limit is not None and limit >= 0)
        except _LuciousElse:
            try:
                _lucious_take(self.roles.get('ask'))
            except _LuciousElse:
                pass
            else:
                self.msg_limit = 3
        else:
            self.msg_limit = limit
        steps = _count(brief.get('steps'))
        try:
            _lucious_take(steps is not None and 10 <= steps < self.max_steps)
        except _LuciousElse:
            pass
        else:
            self.max_steps = steps
        _lucious_it_16 = iter(_items(brief.get('process')))
        while True:
            try:
                entry = next(_lucious_it_16)
            except StopIteration:
                break
            try:
                _lucious_take(isinstance(entry, dict) and str(entry.get('step') or '').strip())
            except _LuciousElse:
                pass
            else:
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        try:
            _lucious_take(self.roles.get('ask'))
        except _LuciousElse:
            pass
        else:
            if len(self.items) > 1 and self.msg_limit >= 3 and not any(p['when'] == 'order' for p in self.process):
                self.process.append({'step': 'show the shopper the final cart with its total and ask them to confirm it', 'when': 'order', 'messages': 1})
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        _lucious_it_17 = iter(enumerate(self.items, 1))
        while True:
            try:
                number, item = next(_lucious_it_17)
            except StopIteration:
                break
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            try:
                _lucious_take(not pointers or not rest)
            except _LuciousElse:
                pass
            else:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, self.checker(), 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            try:
                _lucious_take(not moved)
            except _LuciousElse:
                pass
            else:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        _lucious_it_18 = iter(enumerate(self.items, 1))
        while True:
            try:
                number, item = next(_lucious_it_18)
            except StopIteration:
                break
            needs = [m + (f' (value on the page it points at: {self.ref_values[m]}' + (f'; a rule stated on that page gives: {self.ref_rules[m]}' if m in self.ref_rules else '') + ')' if m in self.ref_values else '') for m in item.musts + item.told if m not in item.dropped]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            try:
                _lucious_take(item.shelf)
            except _LuciousElse:
                pass
            else:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            try:
                _lucious_take(item.free)
            except _LuciousElse:
                pass
            else:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        brief = self.brief
        try:
            _lucious_take(self.budget is not None)
        except _LuciousElse:
            pass
        else:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        try:
            _lucious_take(pick == 'cheapest')
        except _LuciousElse:
            try:
                _lucious_take(pick in ('most', 'least'))
            except _LuciousElse:
                parts.append('pick: any listing that fits everything')
            else:
                parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: the cheapest listing that fits (lowest price)')
        try:
            _lucious_take(brief.get('in_stock_only', True) is not False)
        except _LuciousElse:
            pass
        else:
            parts.append('in stock only')
        _lucious_it_19 = iter(_items(brief.get('wishes')))
        while True:
            try:
                wish = next(_lucious_it_19)
            except StopIteration:
                break
            parts.append(f'wish (tie-breaker only): {wish}')
        _lucious_it_20 = iter(_items(brief.get('rules')))
        while True:
            try:
                rule = next(_lucious_it_20)
            except StopIteration:
                break
            parts.append(f'rule from the request: {rule}')
        try:
            _lucious_take(brief.get('quiet_decline') is True)
        except _LuciousElse:
            pass
        else:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        _lucious_it_21 = iter(self.process)
        while True:
            try:
                entry = next(_lucious_it_21)
            except StopIteration:
                break
            when = {'choose': 'before choosing', 'order': 'at the order (Kyo does it when you check out)', 'end': 'at an ending without a purchase (Kyo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        try:
            _lucious_take(facts and (self.facts_from.get(facts) is True or self.roles.get('offer')))
        except _LuciousElse:
            pass
        else:
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Kyo] step {self.step} of {self.max_steps} used, {left} left.'
        try:
            _lucious_take(left <= self.tail() + 1)
        except _LuciousElse:
            try:
                _lucious_take(left <= 6 or (left <= len(self.items) + 5 and self.buys == 0))
            except _LuciousElse:
                try:
                    _lucious_take(self.buys == 0 and left <= 12)
                except _LuciousElse:
                    try:
                        _lucious_take(self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2))
                    except _LuciousElse:
                        pass
                    else:
                        text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
                else:
                    text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
            else:
                text += ' Few steps left: set the cart to the chosen lines and check out now.'
        else:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        secs = self.time_left()
        try:
            _lucious_take(secs < CLOSING + 50)
        except _LuciousElse:
            pass
        else:
            text += f' Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        try:
            _lucious_take(lookups and self.buys == 0)
        except _LuciousElse:
            pass
        else:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        try:
            _lucious_take(self.leads)
        except _LuciousElse:
            pass
        else:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        try:
            _lucious_take(self.notes)
        except _LuciousElse:
            pass
        else:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        try:
            _lucious_take(not isinstance(node, dict) or len(node) < 3)
        except _LuciousElse:
            pass
        else:
            return None
        try:
            _lucious_take(self.id_key and _id_like(node.get(self.id_key), True))
        except _LuciousElse:
            pass
        else:
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        try:
            _lucious_take(not (in_list or depth == 0 or nested))
        except _LuciousElse:
            pass
        else:
            return None
        found = _learn_id([node])
        try:
            _lucious_take(found and ('id' in str(found).lower() or node[found].strip() in self.rows))
        except _LuciousElse:
            pass
        else:
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        try:
            _lucious_take(not rows or len(rows) < len(node) / 2)
        except _LuciousElse:
            pass
        else:
            return False
        try:
            _lucious_take(self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)))
        except _LuciousElse:
            pass
        else:
            return True
        found = _learn_id(rows)
        try:
            _lucious_take(found)
        except _LuciousElse:
            pass
        else:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            try:
                _lucious_take(depth > 6)
            except _LuciousElse:
                pass
            else:
                return
            try:
                _lucious_take(isinstance(node, dict))
            except _LuciousElse:
                try:
                    _lucious_take(isinstance(node, list))
                except _LuciousElse:
                    pass
                else:
                    rows = self.learn_rows(node)
                    _lucious_it_23 = iter(node)
                    while True:
                        try:
                            value = next(_lucious_it_23)
                        except StopIteration:
                            break
                        walk(value, depth + 1, rows)
            else:
                ident = self.ident(node, in_list, depth)
                try:
                    _lucious_take(ident)
                except _LuciousElse:
                    pass
                else:
                    try:
                        _lucious_take(ident not in self.rows)
                    except _LuciousElse:
                        pass
                    else:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    try:
                        _lucious_take(any((old.get(k) is True and v is False for k, v in node.items())))
                    except _LuciousElse:
                        try:
                            _lucious_take(any((old.get(k) is False and v is True for k, v in node.items())))
                        except _LuciousElse:
                            pass
                        else:
                            self.unavailable.discard(ident)
                    else:
                        self.unavailable.add(ident)
                    self.rows[ident].update(node)
                    try:
                        _lucious_take(_fact_block(node))
                    except _LuciousElse:
                        try:
                            _lucious_take(ident in self.full)
                        except _LuciousElse:
                            pass
                        else:
                            self.full[ident].update(node)
                    else:
                        self.full[ident] = dict(self.rows[ident])
                    return
                _lucious_it_22 = iter(node.values())
                while True:
                    try:
                        value = next(_lucious_it_22)
                    except StopIteration:
                        break
                    walk(value, depth + 1)
        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            try:
                _lucious_take(depth > 6 or not isinstance(schema, dict))
            except _LuciousElse:
                pass
            else:
                return
            _lucious_it_24 = iter(schema.get('enum') or [])
            while True:
                try:
                    option = next(_lucious_it_24)
                except StopIteration:
                    break
                try:
                    _lucious_take(isinstance(option, str))
                except _LuciousElse:
                    pass
                else:
                    found.add(option.strip().lower())
            _lucious_it_25 = iter(schema.values())
            while True:
                try:
                    value = next(_lucious_it_25)
                except StopIteration:
                    break
                try:
                    _lucious_take(isinstance(value, dict))
                except _LuciousElse:
                    pass
                else:
                    walk(value, depth + 1)
        _lucious_it_26 = iter(self.schemas.values())
        while True:
            try:
                schema = next(_lucious_it_26)
            except StopIteration:
                break
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        _lucious_it_28 = iter(self.rows.values())
        while True:
            try:
                row = next(_lucious_it_28)
            except StopIteration:
                break
            _lucious_it_27 = iter(row.items())
            while True:
                try:
                    key, value = next(_lucious_it_27)
                except StopIteration:
                    break
                try:
                    _lucious_take(isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)))
                except _LuciousElse:
                    pass
                else:
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        _lucious_it_29 = iter(counts.items())
        while True:
            try:
                key, values = next(_lucious_it_29)
            except StopIteration:
                break
            try:
                _lucious_take(present[key] < 0.8 * total or len(values) > 40)
            except _LuciousElse:
                pass
            else:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            try:
                _lucious_take(best is None or score < best[0])
            except _LuciousElse:
                pass
            else:
                best = (score, key)
        return best[1] if best else None

    def label_fit(self, label: str, item: Item) -> int:
        """How well a store category label fits a requested item: 3 the label is the item's shelf or name, 2 every
        word of the label is among the item's words (a short label such as an abbreviation), 1 they share a word of
        four letters or more, 0 not at all."""
        mine = _stem(item.shelf or '') | _stem(item.name)
        theirs = _stem(label)
        try:
            _lucious_take(not theirs)
        except _LuciousElse:
            pass
        else:
            return 0
        try:
            _lucious_take(_words(label) in (_words(item.shelf or ''), _words(item.name)))
        except _LuciousElse:
            pass
        else:
            return 3
        try:
            _lucious_take(theirs <= mine)
        except _LuciousElse:
            pass
        else:
            return 2
        try:
            _lucious_take({w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4})
        except _LuciousElse:
            pass
        else:
            return 1
        return 0

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        _lucious_it_31 = iter(self.seen_order)
        while True:
            try:
                ident = next(_lucious_it_31)
            except StopIteration:
                break
            record = self.full.get(ident)
            try:
                _lucious_take(not record)
            except _LuciousElse:
                pass
            else:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            _lucious_it_30 = iter(block.items())
            while True:
                try:
                    key, value = next(_lucious_it_30)
                except StopIteration:
                    break
                name = _words(key)
                try:
                    _lucious_take(not name or name in done)
                except _LuciousElse:
                    pass
                else:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                try:
                    _lucious_take(text not in slot['values'])
                except _LuciousElse:
                    pass
                else:
                    slot['values'].append(text)
        matched = {label for label in groups if any((self.label_fit(label, item) for item in self.items))}
        try:
            _lucious_take(matched)
        except _LuciousElse:
            pass
        else:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        _lucious_it_34 = iter(sorted(groups.items(), key=lambda kv: -kv[1]['n']))
        while True:
            try:
                label, group = next(_lucious_it_34)
            except StopIteration:
                break
            group['keys'] = self.merge_keys(group['keys'])
            kept: list[tuple[str, dict[str, Any]]] = []
            _lucious_it_32 = iter(sorted(group['keys'].items(), key=lambda kv: -kv[1]['n']))
            while True:
                try:
                    name, slot = next(_lucious_it_32)
                except StopIteration:
                    break
                values = list(slot['per'].values())
                try:
                    _lucious_take(not values)
                except _LuciousElse:
                    pass
                else:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                try:
                    _lucious_take(mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values))
                except _LuciousElse:
                    pass
                else:
                    continue
                try:
                    _lucious_take(len(values) >= 4 and len(set(values)) == len(values) and (mean > 12))
                except _LuciousElse:
                    pass
                else:
                    continue
                twin = any((name.startswith(other + ' ') and slot['per'] == spec['per'] for other, spec in kept))
                try:
                    _lucious_take(twin)
                except _LuciousElse:
                    pass
                else:
                    continue
                kept.append((name, slot))
            _lucious_it_33 = iter(kept[:14])
            while True:
                try:
                    name, slot = next(_lucious_it_33)
                except StopIteration:
                    break
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
                self.key_distinct[label, name] = len(set(slot['per'].values()))
        return table

    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Stores spell one property several ways across listings (a shorter or longer key, a run-together word).
        Two keys are one property when one key's words are all in the other's and their values agree on a listing
        both carry or share a value, or when they share a word and agree on every one of at least three listings
        both carry. The merged key counts every listing carrying either and is named by its longest spelling (the
        one a question can name without also naming a neighbour)."""

        def flat(text: str) -> str:
            return re.sub('[^a-z0-9]+', '', text)
        names = sorted(keys, key=lambda n: (-len(n.split()), -keys[n]['n'], n))
        merged: dict[str, dict[str, Any]] = {}
        _lucious_it_38 = iter(names)
        while True:
            try:
                name = next(_lucious_it_38)
            except StopIteration:
                break
            slot = keys[name]
            mine = _stem(name)
            target = None
            _lucious_it_35 = iter(merged.items())
            while True:
                try:
                    big, spec = next(_lucious_it_35)
                except StopIteration:
                    break
                theirs = _stem(big)
                both = [i for i in slot['per'] if i in spec['per']]
                agree = sum((1 for i in both if flat(slot['per'][i]) == flat(spec['per'][i])))
                share = bool({flat(v) for v in slot['values']} & {flat(v) for v in spec['values']})
                try:
                    _lucious_take((mine <= theirs or theirs <= mine) and (agree or share) and (agree == len(both)))
                except _LuciousElse:
                    pass
                else:
                    target = big
                    break
                try:
                    _lucious_take(mine & theirs and len(both) >= 3 and (agree == len(both)))
                except _LuciousElse:
                    pass
                else:
                    target = big
                    break
            try:
                _lucious_take(target is None)
            except _LuciousElse:
                pass
            else:
                merged[name] = {'n': slot['n'], 'values': list(slot['values']), 'per': dict(slot['per'])}
                continue
            spec = merged[target]
            _lucious_it_36 = iter(slot['per'].items())
            while True:
                try:
                    ident, text = next(_lucious_it_36)
                except StopIteration:
                    break
                spec['per'].setdefault(ident, text)
            _lucious_it_37 = iter(slot['values'])
            while True:
                try:
                    value = next(_lucious_it_37)
                except StopIteration:
                    break
                try:
                    _lucious_take(value not in spec['values'])
                except _LuciousElse:
                    pass
                else:
                    spec['values'].append(value)
            spec['n'] = len(spec['per'])
        return merged

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        _lucious_it_39 = iter(self.seen_order)
        while True:
            try:
                ident = next(_lucious_it_39)
            except StopIteration:
                break
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any((self.label_fit(label, item) for item in self.items))]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        try:
            _lucious_take(not tool or not target or any((k != target for k in schema.get('required') or [])))
        except _LuciousElse:
            pass
        else:
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        try:
            _lucious_take(isinstance(value, dict))
        except _LuciousElse:
            pass
        else:
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        try:
            _lucious_take(isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value))
        except _LuciousElse:
            pass
        else:
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        try:
            _lucious_take(self.sold_out(ident))
        except _LuciousElse:
            pass
        else:
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
        try:
            _lucious_take(isinstance(observation, dict))
        except _LuciousElse:
            pass
        else:
            ident = self.ident(observation)
            try:
                _lucious_take(ident)
            except _LuciousElse:
                pass
            else:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            _lucious_it_40 = iter(observation.values())
            while True:
                try:
                    value = next(_lucious_it_40)
                except StopIteration:
                    break
                try:
                    _lucious_take(isinstance(value, list))
                except _LuciousElse:
                    try:
                        _lucious_take(isinstance(value, dict))
                    except _LuciousElse:
                        pass
                    else:
                        found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                else:
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        try:
            _lucious_take(isinstance(observation, list))
        except _LuciousElse:
            pass
        else:
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def note_view(self, name: str, observation: Any) -> None:
        """Which fields each viewing tool shows, and which a listing has shown since the last change: a view whose
        fields were all shown already is not run again."""
        _lucious_it_41 = iter(self.records_in(observation))
        while True:
            try:
                ident, record = next(_lucious_it_41)
            except StopIteration:
                break
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        try:
            _lucious_take(name in self.shapes)
        except _LuciousElse:
            pass
        else:
            return self.shapes[name]
        try:
            _lucious_take(not isinstance(observation, dict))
        except _LuciousElse:
            pass
        else:
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        try:
            _lucious_take(not lists)
        except _LuciousElse:
            pass
        else:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        try:
            _lucious_take(not rows)
        except _LuciousElse:
            pass
        else:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        try:
            _lucious_take(shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)))
        except _LuciousElse:
            pass
        else:
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        try:
            _lucious_take(not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)))
        except _LuciousElse:
            pass
        else:
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        try:
            _lucious_take(not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)))
        except _LuciousElse:
            pass
        else:
            return None
        try:
            _lucious_take(value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))))
        except _LuciousElse:
            pass
        else:
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        try:
            _lucious_take(not plan)
        except _LuciousElse:
            pass
        else:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        try:
            _lucious_take(value is None or later or count <= shown)
        except _LuciousElse:
            pass
        else:
            return ''
        try:
            _lucious_take(pick == 'cheapest' and self.price_sorted(observation))
        except _LuciousElse:
            pass
        else:
            return ''
        pages = -(-(count - shown) // shown)
        try:
            _lucious_take(pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12)
        except _LuciousElse:
            pass
        else:
            return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        _lucious_it_42 = iter(range(pages))
        while True:
            try:
                _ = next(_lucious_it_42)
            except StopIteration:
                break
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            try:
                _lucious_take(faults)
            except _LuciousElse:
                pass
            else:
                break
            reply = self.act(name, call)
            try:
                _lucious_take(reply is None or self.finished)
            except _LuciousElse:
                pass
            else:
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
            try:
                _lucious_take(value is None)
            except _LuciousElse:
                pass
            else:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        return '\n[Kyo read the rest of this search itself, one step a page]\n' + '\n'.join(texts) if texts else ''

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        try:
            _lucious_take(self.brief.get('pick') != 'cheapest' or not self.price_key)
        except _LuciousElse:
            pass
        else:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        try:
            _lucious_take(len(rows) < 2)
        except _LuciousElse:
            pass
        else:
            return ''
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        try:
            _lucious_take(self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items))
        except _LuciousElse:
            pass
        else:
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        try:
            _lucious_take(not tool or not target)
        except _LuciousElse:
            pass
        else:
            return []
        texts = []
        _lucious_it_43 = iter([i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))])
        while True:
            try:
                ident = next(_lucious_it_43)
            except StopIteration:
                break
            try:
                _lucious_take(self.left() <= keep or self.finished or self.facts_from.get(tool) is False)
            except _LuciousElse:
                pass
            else:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            try:
                _lucious_take(faults)
            except _LuciousElse:
                pass
            else:
                break
            reply = self.act(tool, args)
            try:
                _lucious_take(reply is None)
            except _LuciousElse:
                pass
            else:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            try:
                _lucious_take(isinstance(seen, dict) and self.ident(seen))
            except _LuciousElse:
                pass
            else:
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        try:
            _lucious_take(texts)
        except _LuciousElse:
            pass
        else:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        try:
            _lucious_take(self.brief.get('pick') != 'cheapest')
        except _LuciousElse:
            pass
        else:
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        try:
            _lucious_take(mark in self.cheaper_marks or not cat_key)
        except _LuciousElse:
            pass
        else:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        _lucious_it_44 = iter(cart)
        while True:
            try:
                line = next(_lucious_it_44)
            except StopIteration:
                break
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            try:
                _lucious_take(price is None or not kind)
            except _LuciousElse:
                pass
            else:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        try:
            _lucious_take(not lines)
        except _LuciousElse:
            pass
        else:
            return ''
        meaning = self.take_note(lines)
        try:
            _lucious_take(meaning)
        except _LuciousElse:
            pass
        else:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        try:
            _lucious_take(not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare())
        except _LuciousElse:
            pass
        else:
            self.widen_singly()
            return
        try:
            _lucious_take(self.left() <= 12)
        except _LuciousElse:
            pass
        else:
            return
        cap = self.compare_room()
        try:
            _lucious_take(cap < 2)
        except _LuciousElse:
            pass
        else:
            self.widen_singly()
            return
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        _lucious_it_45 = iter(ranked)
        while True:
            try:
                ids = next(_lucious_it_45)
            except StopIteration:
                break
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        try:
            _lucious_take(len(picks) < 2)
        except _LuciousElse:
            pass
        else:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, 'args', faults)
        try:
            _lucious_take(faults)
        except _LuciousElse:
            pass
        else:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        try:
            _lucious_take(reply is not None)
        except _LuciousElse:
            pass
        else:
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            try:
                _lucious_take(rows)
            except _LuciousElse:
                pass
            else:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            try:
                _lucious_take(not any((str(i) in _dump(seen) for i in picks)))
            except _LuciousElse:
                self.compare_ok = max(self.compare_ok, len(picks))
            else:
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            try:
                _lucious_take(rows and self.facts_from.get(tool) is False)
            except _LuciousElse:
                pass
            else:
                self.widen_singly()
                return
            try:
                _lucious_take(len(self.full) == before and len(picks) > 2 and (self.left() > 12))
            except _LuciousElse:
                pass
            else:
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                try:
                    _lucious_take(not faults)
                except _LuciousElse:
                    pass
                else:
                    reply = self.act(tool, args)
                    try:
                        _lucious_take(reply is not None)
                    except _LuciousElse:
                        pass
                    else:
                        self.observe(tool, reply.get('observation'))
                        try:
                            _lucious_take(any((str(i) in _dump(reply.get('observation')) for i in picks[:2])))
                        except _LuciousElse:
                            pass
                        else:
                            self.compare_ok = max(self.compare_ok, 2)

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        try:
            _lucious_take(not tool or not target)
        except _LuciousElse:
            pass
        else:
            return
        lanes: list[list[str]] = []
        _lucious_it_46 = iter(self.item_groups())
        while True:
            try:
                ids = next(_lucious_it_46)
            except StopIteration:
                break
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            try:
                _lucious_take(self.brief.get('pick') == 'cheapest')
            except _LuciousElse:
                pass
            else:
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            lanes.append(pool[:max(0, min(4 if len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:min(8, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        try:
            _lucious_take(self.asked or not self.roles.get('ask') or self.msgs_free() <= 0)
        except _LuciousElse:
            pass
        else:
            return False
        try:
            _lucious_take(self.left() <= 8)
        except _LuciousElse:
            pass
        else:
            return False
        opened = len(self.full)
        try:
            _lucious_take(opened >= 3 * len(self.items) + 1)
        except _LuciousElse:
            pass
        else:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        try:
            _lucious_take(not target)
        except _LuciousElse:
            pass
        else:
            return None
        args: dict[str, Any] = {target: text}
        _lucious_it_47 = iter(schema.get('required') or [])
        while True:
            try:
                key = next(_lucious_it_47)
            except StopIteration:
                break
            try:
                _lucious_take(key not in args)
            except _LuciousElse:
                pass
            else:
                spec = props.get(key) or {}
                try:
                    _lucious_take(isinstance(spec.get('enum'), list) and spec['enum'])
                except _LuciousElse:
                    return None
                else:
                    args[key] = spec['enum'][0]
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        """Ask the shopper one question at a time and read each answer before choosing the next. One plan ranks each
        item's properties; the questions take turns between the items, an item whose answer revealed a preference waits
        until the others had their turn, and an answer that wants a single question gets that property asked alone once
        more. Every message goes to a question (the request's own steps keep theirs)."""
        self.asked = True
        count = self.msgs_free()
        try:
            _lucious_take(count <= 0 or self.time_left() <= CLOSING + QUESTION_TIME or (not self.roles.get('ask')))
        except _LuciousElse:
            pass
        else:
            return ''
        self.widen()
        table = self.key_table()
        settled = []
        _lucious_it_48 = iter(enumerate(self.items, 1))
        while True:
            try:
                number, item = next(_lucious_it_48)
            except StopIteration:
                break
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        _lucious_it_49 = iter(enumerate(table, 1))
        while True:
            try:
                index, (label, name, carried, values) = next(_lucious_it_49)
            except StopIteration:
                break
            try:
                _lucious_take(label != last)
            except _LuciousElse:
                pass
            else:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values))
        want = count + len(self.items) + 2
        prompt = PLAN.replace('COUNT', str(want)) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        try:
            _lucious_take(first)
        except _LuciousElse:
            pass
        else:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        queues: dict[int, list[str]] = {}
        order: list[int] = []
        _lucious_it_50 = iter(_items(plan.get('ask')))
        while True:
            try:
                pick = next(_lucious_it_50)
            except StopIteration:
                break
            try:
                _lucious_take(not isinstance(pick, dict))
            except _LuciousElse:
                pass
            else:
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            try:
                _lucious_take(not label or len(label.split()) > 5)
            except _LuciousElse:
                pass
            else:
                continue
            queue = queues.setdefault(number, [])
            try:
                _lucious_take(all((_words(label) != _words(other) for other in queue)))
            except _LuciousElse:
                pass
            else:
                queue.append(label)
            try:
                _lucious_take(number not in order)
            except _LuciousElse:
                pass
            else:
                order.append(number)
        _lucious_it_52 = iter(range(1, len(self.items) + 1))
        while True:
            try:
                number = next(_lucious_it_52)
            except StopIteration:
                break
            queue = queues.setdefault(number, [])
            _lucious_it_51 = iter(self.item_keys(number, table, 8))
            while True:
                try:
                    name = next(_lucious_it_51)
                except StopIteration:
                    break
                try:
                    _lucious_take(all((_words(name) != _words(other) for other in queue)))
                except _LuciousElse:
                    pass
                else:
                    queue.append(name)
                    self.topup.add(name)
            try:
                _lucious_take(queue and number not in order)
            except _LuciousElse:
                pass
            else:
                order.append(number)
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'queues': _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        last_item: int | None = None
        for _lucious_tick_53 in iter(lambda: True if self.msgs_free() > 0 and (not self.finished) and (self.time_left() > CLOSING + QUESTION_TIME) else _LUCIOUS_END, _LUCIOUS_END):
            pick = self.next_pick(order, queues, found, last_item)
            try:
                _lucious_take(pick is None)
            except _LuciousElse:
                pass
            else:
                break
            number, label = pick
            text = self.question(number, label)
            batch = self.send([(number, label, text)])
            try:
                _lucious_take(not batch)
            except _LuciousElse:
                pass
            else:
                break
            last_item = number
            self.asked_labels.append((number, label))
            try:
                _lucious_take(self.msg_refused)
            except _LuciousElse:
                pass
            else:
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
            before = len(told)
            self.take_heard(heard, told)
            try:
                _lucious_take(len(told) > before)
            except _LuciousElse:
                pass
            else:
                found.add(number)
        try:
            _lucious_take(not sent)
        except _LuciousElse:
            pass
        else:
            return ''
        try:
            _lucious_take(told)
        except _LuciousElse:
            pass
        else:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        searched = self.value_search(sorted(found)) if found else ''
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        if searched:
            summary += '\n' + searched
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        try:
            _lucious_take(pointed)
        except _LuciousElse:
            pass
        else:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        try:
            _lucious_take(self.queued)
        except _LuciousElse:
            pass
        else:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def next_pick(self, order: list[int], queues: dict[int, list[str]], found: set[int], last_item: int | None) -> tuple[int, str] | None:
        """The next question: the item after the one asked last that still has a property queued, items whose answers
        revealed nothing yet first. A property an answer has already settled for that item is not asked again."""
        _lucious_it_54 = iter(queues.items())
        while True:
            try:
                number, queue = next(_lucious_it_54)
            except StopIteration:
                break
            try:
                _lucious_take(number <= len(self.items))
            except _LuciousElse:
                pass
            else:
                item = self.items[number - 1]
                told = [_stem(t.split(':')[0]) for t in item.told]
                stated = [_stem(t.split(':')[0]) for t in item.musts]
                queue[:] = [label for label in queue if not any((k and k <= _stem(label) for k in told)) and (not (label in self.topup and any((k and (k <= _stem(label) or _stem(label) <= k) for k in stated))))]
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any((_stem(label) <= a for a in asked))]
        live = [n for n in order if queues.get(n)]
        try:
            _lucious_take(not live)
        except _LuciousElse:
            pass
        else:
            return None
        pool = [n for n in live if n not in found] or live
        start = order.index(last_item) + 1 if last_item in order else 0
        _lucious_it_55 = iter(range(len(order)))
        while True:
            try:
                step = next(_lucious_it_55)
            except StopIteration:
                break
            number = order[(start + step) % len(order)]
            try:
                _lucious_take(number in pool)
            except _LuciousElse:
                pass
            else:
                return (number, queues[number].pop(0))
        return None

    def item_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], limit: int=4) -> list[str]:
        """The most carried keys of the category that matches an item (its label or the words of its name)."""
        item = self.items[number - 1]
        settled = ' '.join(item.musts).lower()
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        shared: list[str] = []
        single: list[str] = []
        for label, name, carried, values in table:
            near = best > 0 and fits[label] == best
            try:
                _lucious_take(near and carried >= 2 and (name not in settled) and (name not in shared + single))
            except _LuciousElse:
                pass
            else:
                (shared if self.key_distinct.get((label, name), len(values)) < carried else single).append(name)
        return (shared + single)[:limit]

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22 or self.time_left() <= CLOSING + QUESTION_TIME:
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
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.label_fit(str((self.rows.get(i) or {}).get(cat_key) or ''), item) > 0) and (cap is None or (self.price_of(i) or 0) <= cap)]
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
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

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

    def question(self, number: int, label: str) -> str:
        name = self.items[number - 1].name
        short = re.split('[,;(]', name)[0].strip() or name
        return f'What {label} would you like for the {short}?'

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            try:
                _lucious_take(self.finished or self.left() <= floor or self.msg_used >= self.msg_limit)
            except _LuciousElse:
                pass
            else:
                break
            args = self.message_args(text)
            try:
                _lucious_take(not args)
            except _LuciousElse:
                pass
            else:
                break
            reply = self.act(tool, args)
            try:
                _lucious_take(reply is None)
            except _LuciousElse:
                pass
            else:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            self.msg_refused = isinstance(answer, dict) and bool(answer.get('error'))
            try:
                _lucious_take(self.msg_refused)
            except _LuciousElse:
                pass
            else:
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            try:
                _lucious_take(not isinstance(entry, dict) or not str(entry.get('need') or '').strip())
            except _LuciousElse:
                pass
            else:
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            try:
                _lucious_take(need not in self.items[number - 1].told)
            except _LuciousElse:
                pass
            else:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            try:
                _lucious_take(entry.get('points') is True and (number, need) not in self.refs)
            except _LuciousElse:
                pass
            else:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            try:
                _lucious_take(isinstance(entry, dict) and str(entry.get('property') or '').strip())
            except _LuciousElse:
                pass
            else:
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            try:
                _lucious_take(isinstance(entry, dict) and entry.get('property'))
            except _LuciousElse:
                pass
            else:
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        try:
            _lucious_take(isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0))
        except _LuciousElse:
            pass
        else:
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p['when'] == stage]
        try:
            _lucious_take(not steps or self.finished or (not self.gate_due(stage)))
        except _LuciousElse:
            pass
        else:
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            try:
                _lucious_take(got is None)
            except _LuciousElse:
                pass
            else:
                break
            told: list[str] = []
            hold = ''
            try:
                _lucious_take(said)
            except _LuciousElse:
                pass
            else:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            try:
                _lucious_take(told or hold)
            except _LuciousElse:
                pass
            else:
                self.version += 1
                return (False, ('Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            try:
                _lucious_take(not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2))
            except _LuciousElse:
                pass
            else:
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            try:
                _lucious_take(not call or name not in self.schemas or name in ending)
            except _LuciousElse:
                pass
            else:
                try:
                    _lucious_take(not call)
                except _LuciousElse:
                    pass
                else:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            try:
                _lucious_take(self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit))
            except _LuciousElse:
                pass
            else:
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            try:
                _lucious_take(faults)
            except _LuciousElse:
                pass
            else:
                break
            try:
                _lucious_take(name == self.roles.get('ask'))
            except _LuciousElse:
                pass
            else:
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, '', text)], 1)
                try:
                    _lucious_take(not batch)
                except _LuciousElse:
                    pass
                else:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.act(name, args)
            try:
                _lucious_take(reply is None)
            except _LuciousElse:
                pass
            else:
                break
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        try:
            _lucious_take(isinstance(observation, str))
        except _LuciousElse:
            pass
        else:
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            try:
                _lucious_take(depth > 4)
            except _LuciousElse:
                pass
            else:
                return
            try:
                _lucious_take(isinstance(node, str) and len(node.split()) >= 2)
            except _LuciousElse:
                try:
                    _lucious_take(isinstance(node, dict))
                except _LuciousElse:
                    try:
                        _lucious_take(isinstance(node, list))
                    except _LuciousElse:
                        pass
                    else:
                        for value in node:
                            walk(value, depth + 1)
                else:
                    for value in node.values():
                        walk(value, depth + 1)
            else:
                found.append(node)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        try:
            _lucious_take(isinstance(observation, dict) and (not self.ident(observation)))
        except _LuciousElse:
            pass
        else:
            for key, value in observation.items():
                try:
                    _lucious_take(isinstance(value, str) and len(value.split()) >= 5)
                except _LuciousElse:
                    try:
                        _lucious_take(isinstance(value, list) and value and all((isinstance(v, str) for v in value)))
                    except _LuciousElse:
                        try:
                            _lucious_take(isinstance(value, dict) and (not self.ident(value, False, 1)))
                        except _LuciousElse:
                            pass
                        else:
                            for inner in value.values():
                                try:
                                    _lucious_take(isinstance(inner, str) and len(inner.split()) >= 6)
                                except _LuciousElse:
                                    pass
                                else:
                                    lines.append(inner)
                    else:
                        lines.extend((v for v in value if len(v.split()) >= 5))
                else:
                    lines.append(value)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        try:
            _lucious_take(self.note_sold and len(lines) == 1)
        except _LuciousElse:
            pass
        else:
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
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
            self.rearm()
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
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        depts = self.dept_lines(ids)
        if depts:
            prompt += "\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + depts
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = self.ask_json(prompt, self.checker(), 1500)
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

    def dept_lines(self, ids: list[str]) -> str:
        """Each cart line's store category, and per item the categories seen whose names share the item's words: a
        neighbouring category (a near kind of product on another shelf) shows up next to the right one."""
        key = self.category_key()
        if not key:
            return ''
        labels: list[str] = []
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(key) or '').strip()
            if label and label not in labels:
                labels.append(label)
        out = []
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(key) or '').strip()
            if label:
                out.append(f'{ident}: category {label!r}')
        for number, item in enumerate(self.items, 1):
            near = sorted((label for label in labels if self.label_fit(label, item)), key=lambda label: -self.label_fit(label, item))
            if near:
                out.append(f'item {number} ({item.name}): categories seen whose names share its words: ' + ', '.join((repr(l) for l in near[:4])))
        return '\n'.join(out)

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
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
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
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
                    if str(chain.get('rule_value') or '').strip():
                        self.ref_rules[need] = str(chain['rule_value']).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get('about') or '') for l, i in zip(links, ids) if i not in self.rows), '') or 'the listing it points at'
                    lost.append(f'{need} -> find {about}')
            pending = [p for p in pending if p not in self.ref_values]
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
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < 3 and (self.left() > 5) and (self.time_left() > CLOSING)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + '. Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.'
        return (text, hold)

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
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
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
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
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > SPEND_SOFT:
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
        got = self.ask_json(prompt, self.checker(), 1200)
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
        if self.audited or self.cost > SPEND_SOFT:
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
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def last_pick(self) -> str:
        cart, buy = self.roles.get('cart'), self.roles.get('buy')
        if self.picked or self.finished or not cart or not buy or self.left() < 2:
            return ''
        self.picked = True
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident)
            records.append(('OPENED ' + _dump(_clip(record, 200))[:900]) if record else ('ROW ' + _dump(_clip(self.rows.get(ident), 120))[:400]))
            if len(records) >= 24:
                break
        if not records:
            return ''
        prompt = LAST_PICK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200)
        self.trace.append({'kind': 'last_pick', 'got': got})
        plan: dict[str, int] = {}
        taken: set[int] = set()
        if got is None:
            for number, ids in self.lead_items.items():
                index = _count(number)
                pick = next((i for i in ids if not self.sold_out(i)), None)
                if pick and index and 1 <= index <= len(self.items) and pick not in plan:
                    plan[pick] = self.items[index - 1].qty
                    taken.add(index)
        for entry in _items((got or {}).get('lines')):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get('id') or '')
            index = _count(entry.get('item')) or 0
            if ident in self.rows and not self.sold_out(ident) and 1 <= index <= len(self.items) and index not in taken and ident not in plan:
                plan[ident] = self.items[index - 1].qty
                taken.add(index)
        if len(taken) < len(self.items):
            return ''
        prices = [(self.price_of(i), n) for i, n in plan.items()]
        if isinstance(self.budget, (int, float)) and all(p is not None for p, _ in prices):
            each = self.brief.get('budget_scope') == 'each'
            if (max(p for p, _ in prices) > self.budget) if each else (sum(p * n for p, n in prices) > self.budget):
                return ''
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + 1 > self.left():
            return ''
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.book_cart(cart, args, reply.get('observation'))
            self.observe(cart, reply.get('observation'))
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        said = ''
        if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
            go, said = self.gate('order')
            if not go:
                return said
        if self.finished or self.left() <= 0:
            return ''
        reply = self.act(buy, args)
        self.trace.append({'kind': 'last_pick_buy', 'lines': plan, 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        return (said + ' ' if said else '') + 'Not run (Kyo ordered the closest listings instead). Store reply: ' + self.show(reply.get('observation'))

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            room = 4 if self.left() >= 15 and self.time_left() > CLOSING + 30 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
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
            if name == joiner and isinstance(args.get(many), list) and (len(args[many]) > 1):
                if any((str(i) in _dump(reply.get('observation')) for i in args[many])):
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                else:
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
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
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
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
            self.book_cart(cart, args, reply.get('observation'))
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
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ', so Kyo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
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
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

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
        """Read the cart from a cart tool's own reply when it lists the cart's lines (listing records, each with one
        small count): "synced" when that list was read and is now Kyo's record, "unreadable" when such a list is there
        but its counts cannot be told apart from other small numbers, "none" when the reply carries no such list."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(node, list):
                if all((isinstance(r, dict) for r in node)):
                    lists.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
        if not isinstance(observation, dict) or self.ident(observation):
            return 'none'
        walk(observation, 0)
        touched = [v for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v in self.rows]
        with_ids = [rows for rows in lists if rows and any((self.ident(r, True) for r in rows))]
        rows = next((r for r in with_ids if any((self.ident(x, True) in touched for x in r))), with_ids[0] if with_ids else None)
        if rows is None:
            if len(lists) == 1 and (not lists[0]):
                rows = lists[0]
            else:
                return 'none'
        idents = [self.ident(r, True) for r in rows]
        if not all(idents):
            return 'unreadable'
        key = self.count_key
        if not key and rows:
            keys = [k for k in rows[0] if k != self.price_key and all((isinstance(r.get(k), int) and (not isinstance(r.get(k), bool)) and (0 <= r[k] <= 99) for r in rows))]
            if len(keys) > 1 and name and (args is not None):
                saved = dict(self.cart_lines)
                self.note_cart(name, args)
                expected = dict(self.cart_lines)
                self.cart_lines = saved
                keys = [k for k in keys if all((r.get(k) == expected.get(i, 0) for r, i in zip(rows, idents) if i in touched))]
            if len(keys) != 1:
                return 'unreadable'
            key = keys[0]
            self.count_key = key
        out: dict[str, int] = {}
        for row, ident in zip(rows, idents):
            count = row.get(key) if key else None
            if not isinstance(count, int) or isinstance(count, bool):
                return 'unreadable'
            if count > 0:
                out[ident] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return 'synced'

    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Record a cart call: from the reply's own line list when it can be read; a store that lists the cart after
        every change it made changed nothing when a reply has no list of listings at all (a refusal); otherwise
        (no list, or one that cannot be read) Kyo's own arithmetic."""
        self.cart_touched = True
        self.cart_at = self.step
        state = self.sync_cart(observation, name, args)
        if state == 'synced':
            return
        if state == 'none' and self.cart_shows_lines:
            self.trace.append({'kind': 'cart_unchanged', 'args': _clip(args, 60)})
            return
        self.note_cart(name, args)

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
        """How many listings one side-by-side call may carry: once a call has failed, what the store has shown to work
        (a stated limit can be wrong), else the schema's limit, else what has worked."""
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        cap = spec['maxItems'] if isinstance(spec.get('maxItems'), int) else None
        if self.compare_bad:
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if cap:
            return cap
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
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
        self.join_failed = False
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                if self.join_failed:
                    out.append((cid, 'Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own.'))
                else:
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
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Kyo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2 and (self.time_left() > CLOSING - 25):
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.time_left() > CLOSING + 15:
                    if self.left() > 5:
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
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held) and (self.time_left() > CLOSING + 15):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) and (self.time_left() > CLOSING + 15):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart() or self.last_pick()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and (self.time_left() > LAST_CALL + 12):
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    if all((str(i) in self.rows for i in args[many])):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f'\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
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
                self.book_cart(name, args, observation)
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
                    self.rearm()
            elif self.is_refuse(name, args) and (not reply.get('finished')):
                self.rearm()
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            if self.is_buy(name, args) and (not reply.get('finished')) and (self.note_sold or any((line in self.spoken(observation) for line in self.sold_lines))):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def finish(self, again: bool=False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
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
            done = self.last_pick()
            if self.finished or self.left() <= 0:
                return ''
            if done and again:
                return 'the checkout was refused: ' + done
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
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
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.set_caps()
        self.work(self.loop_cap, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > CLOSING):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > CLOSING):
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
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or (self.cost >= SPEND_STOP):
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
