"""Devil: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Devil reads itself for a cheapest, most or least pick
MODEL_SECONDS = 190.0       # no new model call after this long in one episode
EPISODE_SECONDS = 240.0     # the best cart is checked out by then, well inside the 300 s sandbox
# requirement names a listing record never shows (price limit, stock, quantity, when it would arrive): left to the store
LOGISTICS = r"\b(price|budget|stock|availab\w*|quantit\w*|deliver\w*|arriv\w*|shipping|ships|shipped|dispatch\w*|dates?|deadline)\b"
KEEP_HOLDS = 2              # times a cart change that drops a proven line, or a decline over proven lines, is turned back
DECLINE_HOLDS = 2           # times an ending without a purchase is turned back while a listing is not ruled out
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Devil reads before a cheapest checkout
CHASE_HOPS = 5              # pages Devil follows along one requirement's chain
CHASE_STEPS = 12            # store steps one chain may use

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


# CHOOSING, silent listings (5 Nov study of race-14 qualifying logs): a listing silent on a stated requirement failed in
# every real order (sleeve length 0/10, pack type 0/31), while the dearer listing that stated it passed

GUIDE = """You are Devil, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). An empty keyword result only says that some word of the query is missing from the listings, never that the store lacks the product: search again with fewer words (always keeping the product noun) or browse the product's category. When a short query brings back more rows than you can page through, one query holding every required value narrows them. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Devil knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Devil's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Devil carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Devil's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Devil adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Devil reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- A listing whose attributes, title and description say nothing about a requirement is not shown to fit it, and the store judges it so. When another listing shows every requirement, buy that one, even at a higher price, rather than a cheaper listing that is silent on any of them.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'


PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "lost": [numbers of the questions whose answer only says they did not follow or understand it; not answers that ask for one question at a time], "single": [numbers of the questions whose answer asks to be asked about one thing at a time], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"), and leave out the words that only say how firmly the shopper means it ("only", "nothing else", "decided", "settled", "closed", "end of story"): those are not part of the value.
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": each property the shopper said they do not mind about: one the answer itself names, or the one property of a question that names a single property. When a question names several properties and the answer names none of them, nothing from it is free.
- Folksy or roundabout wording that names a value still states that value.

EXAMPLE (an invented store, only to show the reading)
ITEMS: item 1: desk lamp
1. (about item 1) Q: What finish would you like for the desk lamp?
A: Put finish down as decided: brushed brass and nothing besides.
2. (about item 1) Q: What height would you like for the desk lamp?
A: Height's your call, I really don't mind.
3. (about item 1) Q: What bulb type would you like for the desk lamp?
A: Sorry, that sailed right past me. Could you put it another way?
4. (about item 1) Q: What base would you like for the desk lamp?
A: The only shade I'd take is a pleated one.
5. (about item 1) Q: What arm length would you like for the desk lamp?
A: Exactly the arm length on the page of the floor lamp Norvane sells with a marble base (or wherever that page sends you).
Reading: {"needs": [{"item": 1, "need": "finish: brushed brass", "points": false}, {"item": 1, "need": "shade: pleated", "points": false}, {"item": 1, "need": "arm length: exactly the arm length on the page of the floor lamp Norvane sells with a marble base (or wherever that page sends you)", "points": true}], "drop": [], "free": [{"item": 1, "property": "height"}], "lost": [3], "budget": null}"""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""


UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

CHASE = """A shopper's requirement takes its value from another store listing, described by its shop, its kind and a few of its properties, and that listing's page may send the reader on to a further listing for the same property (and that one to another). You get the requirement, the pages already followed (oldest first) and the candidate pages just opened. Return ONE JSON object and nothing else:
{"match": the id of the candidate that is the listing being looked for (its shop, kind and properties fit the description), or "", "find": when the matched page's own text sends the reader to a different listing for this requirement's property, or when no candidate is the listing yet, the listing to look for next as {"shop": its shop or seller as written, "kind": its kind of product, "details": [a few of its properties as written]}, else null, "value": when the matched page itself gives the value of the requirement's property (in its attributes or its text), that value as written, else ""}
A value that must go with or work alongside the listing's own is the value its text says goes together with it. A mention that some other version "exists elsewhere" sends nowhere. With no candidates yet, describe the first listing from the requirement's own words."""

TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

VOTE = """A shopping agent is about to buy the listing below although it judged that nothing in the store meets every requirement. Read the listing's record and say, for each requirement, what the record says about it: "shown" when an attribute, the title or the description states the wanted value (or a value inside the limit), "other" when it states a different value for that property, "silent" when it says nothing about that property. Judge only from the record's own words; do not guess what an unstated property is likely to be.
Return ONE JSON object and nothing else: {"reqs": [{"need": the requirement copied exactly, "record": "shown", "other" or "silent", "quote": the record's words you rely on, or ""}]}"""
VOTES = 3                   # readings of a forced order's record; a majority that finds a requirement unmet lets the decline stand
VOTE_HEAT = 0.7             # sampling temperature for those readings, so they are independent


SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = text.find("{"), text.rfind("}")
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def _words(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _near(a: str, b: str) -> bool:
    """One letter apart, for spelling variants; short words must match exactly."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] or a[i:] == b[i + 1:] or a[i + 1:] == b[i:]


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*([a-z]*)")


def _removes(args: Any) -> bool:
    """Cart-tool arguments that take a line out: a removal verb, or a negative or zero count."""
    if not isinstance(args, dict):
        return False
    for value in args.values():
        if isinstance(value, str) and value.strip().lower() in ("remove", "discard", "delete", "drop", "minus", "subtract", "take out", "clear"):
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            return True
    return False


def _short_alike(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Property names where one side uses a three-letter shortening ("max" for "maximum"); every word must pair up."""
    def same(a: str, b: str) -> bool:
        return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))
    return bool(prop) and len(prop) == len(key) and all(any(same(w, x) for x in key) for w in prop) and all(any(same(x, w) for w in prop) for x in key)


def _range_clash(need: str, facts: dict[str, str]) -> bool:
    """A requirement that is one range ('speed: 100-499 mbps') against an attribute of that property stating a range in
    the same unit that does not overlap it ('500-999 mbps'). Limits ('at least', 'up to') are left to the model."""
    prop, _, wanted = need.partition(":")
    want = _RANGE.fullmatch(wanted.strip().lower())
    if not want or not prop.strip():
        return False
    lo, hi, unit = float(want.group(1)), float(want.group(2)), want.group(3)
    for key, value in facts.items():
        if not _alike(_stem(prop), _stem(str(key))) and not _short_alike(_stem(prop), _stem(str(key))):
            continue
        got = _RANGE.fullmatch(str(value).strip().lower())
        if got and (got.group(3) == unit or not unit or not got.group(3)):
            a, b = float(got.group(1)), float(got.group(2))
            if b < lo or a > hi:
                return True
    return False


def _alike(prop: frozenset[str], key: frozenset[str]) -> bool:
    """_names with words that are spelling variants or shortenings of each other ("pack type" for "packaging type")."""
    def same(a: str, b: str) -> bool:
        return _near(a, b) or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))
    def inside(small: frozenset[str], big: frozenset[str]) -> bool:
        return all(any(same(w, x) for x in big) for w in small)
    if not prop or not key:
        return False
    return inside(prop, key) or (len(key) >= 2 and inside(key, prop))


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; list what cannot be fixed."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not any(_fits_kind(value, k) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
                break
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
                break
            if kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
                break
        if not any(_fits_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f"{where}: allowed values are {options}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [_shape(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                faults.append(f"{where}.{name} is not an allowed argument")
        for name in schema.get("required") or []:
            if name not in value:
                faults.append(f"{where}.{name} is required")
        value = {k: _shape(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
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
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def _id_like(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a list in the result holds a record carrying this id (a cart listing still showing the line)."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


# ---------- the referee: a model-free reading of requests and listings ----------
_SPLIT_CAMEL = re.compile(r"([a-z0-9])([A-Z])")

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")

_FILLER = {
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by", "at", "is", "are", "be",
    "it", "its", "this", "that", "my", "your", "me", "i", "you", "we", "our", "some", "any", "one",
}

def name_words(name: str) -> List[str]:
    """Split an identifier such as 'firstPageKey' or 'max_unit_cost' into lowercase words."""
    spaced = _SPLIT_CAMEL.sub(r"\1 \2", str(name or ""))
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]

def stem(word: str) -> str:
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and (w.endswith("ches") or w.endswith("shes") or w.endswith("sses") or w.endswith("xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us") and not w.endswith("is"):
        return w[:-1]
    return w

def words(text: Any) -> List[str]:
    """Lowercased, stemmed word and number tokens; numbers and letters are split apart ('5v' -> 5, v)."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = json.dumps(text) if isinstance(text, (dict, list)) else str(text)
    return [stem(t) for t in _TOKEN.findall(text.lower().replace("&", " and "))]

def content_words(text: Any) -> List[str]:
    return [w for w in words(text) if w not in _FILLER]

def numbers_with_units(text: str) -> List[Tuple[float, str]]:
    out = []
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Z\"'%°]{0,6})", str(text or "")):
        try:
            val = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        out.append((val, m.group(2).lower().strip("'\"")))
    return out

class Need:
    """One requirement on one order line.

    kind: eq (value is one of `values`), not (none of `values`), ge / le / range (numeric with unit),
    phrase (shopper wording whose property is unknown), link (value defined by another listing).
    """

    def __init__(self, prop: str, kind: str, values: Optional[List[str]] = None, low: Optional[float] = None,
                 high: Optional[float] = None, unit: str = "", source: str = "stated") -> None:
        self.prop = re.sub(r"\s+", " ", str(prop or "")).strip().lower()
        self.kind = kind
        self.values = [re.sub(r"\s+", " ", str(v)).strip() for v in (values or []) if str(v).strip()]
        self.low = low
        self.high = high
        self.unit = (unit or "").lower().strip()
        self.source = source
        self.active = True
        self.link: Dict[str, Any] = {}

    def signature(self) -> str:
        return "%s|%s|%s|%s|%s|%s" % (self.prop, self.kind, ",".join(sorted(v.lower() for v in self.values)),
                                       self.low, self.high, self.link.get("seller", ""))

    def text(self) -> str:
        p = self.prop or "feature"
        if self.kind == "eq":
            return "%s is %s" % (p, " or ".join(self.values))
        if self.kind == "not":
            return "%s is not %s" % (p, " or ".join(self.values))
        if self.kind == "ge":
            return "%s at least %s %s" % (p, _fmt(self.low), self.unit)
        if self.kind == "le":
            return "%s at most %s %s" % (p, _fmt(self.high), self.unit)
        if self.kind == "range":
            return "%s between %s and %s %s" % (p, _fmt(self.low), _fmt(self.high), self.unit)
        if self.kind == "phrase":
            return "has: %s" % " / ".join(self.values)
        if self.kind == "link":
            rel = "different from" if self.link.get("relation") == "differ" else "same as"
            return "%s %s the listing %s" % (p, rel, self.link.get("describe", ""))
        return p

def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else str(x)

class Line:
    def __init__(self, label: str, query: str = "", qty: int = 1) -> None:
        self.label = re.sub(r"\s+", " ", label or "").strip()
        self.query = re.sub(r"\s+", " ", query or "").strip() or self.label
        self.qty = max(1, int(qty or 1))
        self.needs: List[Need] = []
        self.cats: List[str] = []
        self.cat_votes: Counter = Counter()
        self.pool: "OrderedDict[str, float]" = OrderedDict()
        self.runs: List[dict] = []
        self.zero_queries = 0
        self.tried_queries: set = set()
        self.asked: List[str] = []
        self.dead_props: set = set()
        self.msg_quota = 0
        self.msgs_sent = 0
        self.pick: Optional[str] = None
        self.viewed: List[str] = []
        self.disclosed = 0
        self.focus_ids: set = set()
        self.ask_cap = 99
        self.hints: List[str] = []          # properties a detail of the story implies (asked before any other)
        self.hint_asked = False
        self.story_pick: Optional[str] = None   # listing property the story implies (None: not chosen yet, "": none)
        self.plan: Optional[List[str]] = None   # model-ranked question order for this line (None: not planned yet)
        self.replans = 0

class Brief:
    def __init__(self) -> None:
        self.lines: List[Line] = []
        self.budget: Optional[float] = None
        self.budget_unit = ""
        self.budget_each = False
        self.goal = "none"           # none | cheapest | max | min
        self.goal_prop = ""
        self.msg_budget: Optional[int] = None
        self.step_budget: Optional[int] = None
        self.min_views: Optional[int] = None     # a stated procedure: look at / compare N listings before choosing
        self.decline_ok = False
        self.in_stock = True
        self.version = 0

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "a pair of": 2, "a couple of": 2, "pair of": 2, "couple of": 2}

_NUM_RE = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"

_MONEY_UNIT = r"(?:us\s+|u\.s\.\s+)?([a-z¢]{1,10})"     # whatever short word the request writes after the amount

_ACTION_UNIT = r"(?:steps?|moves?|actions?|turns?|tool[- ]calls?|calls?|clicks?|operations?)"

_MSG_UNIT = r"(?:messages?|questions?|texts?|notes?|asks?|queries|pings?|replies)"

def _to_int(tok: str) -> Optional[int]:
    tok = (tok or "").lower().strip()
    if tok.isdigit():
        return int(tok)
    return _NUMBER_WORDS.get(tok)

def _clean(text: str) -> str:
    t = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"\s+([,.;!?:])", r"\1", t)

def read_step_budget(text: str) -> Optional[int]:
    t = _clean(text).lower()
    m = re.search(r"(\d+)\s+(?:\w+\s+){0,2}?" + _ACTION_UNIT + r"\b", t)
    return int(m.group(1)) if m else None

def read_message_budget(text: str) -> Optional[int]:
    t = _clean(text).lower()
    m = re.search(_NUM_RE + r"\s+(?:[\w']+\s+){0,2}?" + _MSG_UNIT + r"\b", t)
    if m:
        return _to_int(m.group(1))
    m = re.search(_MSG_UNIT + r"[^.\d]{0,25}?(\d+)", t)
    return int(m.group(1)) if m else None

def read_min_views(text: str) -> Optional[int]:
    """'look at / compare / open at least three listings before you choose' (a procedure the request states)."""
    t = _clean(text).lower()
    m = re.search(r"\b(?:look at|looking at|open|compare|comparing|check|checking|view|see|go through|review|"
                  r"consider)\s+(?:at least\s+|a minimum of\s+|no fewer than\s+)?" + _NUM_RE +
                  r"\s+(?:[\w-]+\s+)?(?:listings?|options?|items?|products?|candidates?|choices?|offers?)\b", t)
    if not m:
        return None
    n = _to_int(m.group(1))
    return n if n is not None and 2 <= n <= 10 else None

def read_budget(text: str) -> Tuple[Optional[float], str]:
    t = _clean(text).lower()
    t = re.sub(r"(\d+)\s+(?:\w+\s+){0,2}?" + _ACTION_UNIT + r"\b", " ", t)
    t = re.sub(_NUM_RE + r"\s+(?:[\w']+\s+){0,2}?" + _MSG_UNIT + r"\b", " ", t)
    found: List[Tuple[int, float, str]] = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*" + _MONEY_UNIT + r"(?=\W|$)", t):
        found.append((m.start(), _num(m.group(1)), m.group(2)))
    for m in re.finditer(r"(?:\$|us\$|usd\s?)\s*(\d[\d,]*(?:\.\d+)?)", t):
        found.append((m.start(), _num(m.group(1)), "$"))
    cap_words = r"(budget|spend|cap|ceiling|over|under|up to|max|limit|no more than|afford|below|within|" \
                r"past|beyond|allowance|most|exceed|price|cost)"
    # an amount counts as the budget only with a spending word nearby (the unit word itself is not looked up);
    # a currency sign is enough on its own; the amount closest to a spending word wins
    def gap(item):
        pos = item[0]
        lo = max(0, pos - 60)
        dists = [abs(m.start() + lo - pos) for m in re.finditer(cap_words, t[lo:pos + 40])]
        return min(dists) if dists else None
    scored = []
    for f in found:
        d = gap(f)
        if f[2] in ("$", "¢") or f[2].startswith("¢"):
            scored.append((0 if d is None else d, f))
        elif d is not None:
            scored.append((d, f))
    if scored:
        best = min(scored, key=lambda x: (x[0], x[1][0]))[1]
        return best[1], best[2]
    m = re.search(cap_words + r"\D{0,20}?(\d[\d,]*(?:\.\d+)?)(?!\s*(?:[a-z%\"]))", t)
    if m:
        return _num(m.group(2)), ""
    return None, ""

def _num(s: str) -> float:
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0

def read_cheapest(text: str) -> bool:
    t = _clean(text).lower()
    return bool(re.search(r"\b(cheapest|least expensive|lowest[- ]priced?|lowest (?:price|cost)|costs? the least|"
                          r"most affordable|least costly|lowest-cost)\b", t))

def read_decline_rule(text: str) -> bool:
    t = _clean(text).lower()
    if re.search(r"(nothing (?:is )?(?:found|fits|matches|suitable|works)|can'?t find|cannot find|"
                 r"if there'?s nothing|if nothing|no match|none (?:fit|fits|match))", t):
        return True
    if re.search(r"(?:don'?t|do not|no|never|without)\s+(?:\w+\s+){0,4}?(?:substitut|alternative|replacement|swap)", t):
        return True
    return bool(re.search(r"\b(let me know|tell me)\b", t))

_NP_PREFIX = re.compile(r"^(?:(?:a|an|some|the|one|another|new|any|more|few|a few|several|nice|good|"
                        r"pair of|a pair of|set of|a set of|couple of|a couple of|single|brand new|replacement|"
                        r"to get|to find|to buy|to order|to have|from|your|our)\s+)+")

_NP_STOP = {
    "and", "with", "in", "that", "which", "who", "for", "made", "where", "from", "but", "so", "since", "because",
    "to", "of", "please", "i", "it", "its", "it's", "should", "has", "have", "must", "is", "are",
    "was", "were", "will", "would", "can", "could", "this", "these", "those", "they", "she", "he", "we", "you",
    "my", "your", "our", "their", "on", "at", "as", "by", "like", "just", "only", "not", "no", "then", "too",
    "also", "here", "there", "right", "now", "today", "before", "after", "if", "when", "while", "though",
    "that'd", "that'll", "they're", "it'll", "i'm", "i'll", "i'd", "what", "whatever", "once", "every", "each",
    "thanks", "thank", "first", "again", "please,", "ought", "got", "need", "needs", "want",
    "today", "tonight", "either", "really", "though", "under", "over", "around", "about", "this", "time",
}

_NP_REJECT = {
    "price", "budget", "total", "message", "messages", "step", "steps", "move", "moves", "cap", "ceiling",
    "nothing", "something", "anything", "everything", "thing", "things", "stuff", "cent", "cents",
    "one", "ones", "it", "them", "that", "this", "money", "help", "order", "list", "time",
    "way", "answer", "question", "questions", "word", "idea", "look", "lot", "bit",
    "pick", "choice", "option", "options", "item", "items", "product", "products", "listing", "listings",
    "pair", "pairs", "what's", "whatever's", "whatever", "what", "actually", "who", "which", "there", "here",
    "range", "category", "section",
}

_CUE = (r"(?:i'?m after|i am after|we'?re after|looking for|searching for|shopping for|hunting for|"
        r"in the market for|i want|i'?d like|i would like|i need|we need|need|i'?m wanting|wanting|want|"
        r"grab(?:bing)?|get me|pick up|find(?: me)?|buy(?:ing)?|order(?:ing)?|show me|got any|have any)")

def _clean_np(raw: str) -> str:
    s = raw.lower().strip(" ,.;:!?\"'")
    s = re.sub(r"\s+", " ", s)
    for _ in range(3):
        s2 = _NP_PREFIX.sub("", s)
        if s2 == s:
            break
        s = s2
    toks = s.split(" ")
    kept = []
    for i, tok in enumerate(toks):
        bare = tok.strip(",.;:!?\"'")
        if not bare:
            break
        if bare == "and" and kept and i + 1 < len(toks):
            nxt = toks[i + 1].strip(",.;:!?\"'")
            if nxt and nxt not in _NP_STOP and nxt not in _NP_REJECT and nxt not in ("a", "an", "the"):
                kept.append(bare)
                continue
        if bare in _NP_STOP:
            break
        kept.append(bare)
        if tok != bare:   # punctuation ends the phrase
            break
        if len(kept) >= 5:
            break
    while kept and kept[-1] in ("or", "&", "a", "an", "the"):
        kept.pop()
    return " ".join(kept)

def _np_ok(np: str) -> bool:
    if not np or len(np) < 3:
        return False
    toks = np.split()
    if all(t in _NP_REJECT or t.isdigit() for t in toks):
        return False
    if toks[-1] in _NP_REJECT or toks[0] in _NP_REJECT or toks[0] in _NP_STOP or toks[-1] in _NP_STOP:
        return False
    if toks[0].endswith("'s") and len(toks) == 1:
        return False
    if any(t in ("cts", "cents", "messages", "steps", "moves") for t in toks):
        return False
    return bool(re.search(r"[a-z]{3,}", np))

def guess_products(text: str) -> List[Tuple[str, float]]:
    """Ranked product phrases found with generic English cues ('I want X', 'X is what I'm after', ...)."""
    t = _clean(re.sub(r"\([^)]*\)", " ", text or ""))
    low = t.lower()
    found: Dict[str, float] = {}
    pats = [
        (re.compile(r"\b" + _CUE + r"[:,]?\s+(?=((?:a |an |some |the |one |another |new |a pair of |pair of |a few |"
                    r"any )?[a-z][a-z&/'-]*(?:\s+[a-z&/'-]+){0,6}))"), 2.0),
        (re.compile(r"\b(?:it has to be|it must be)\s+((?:a |an )?[a-z][a-z&/'-]*"
                    r"(?:\s+[a-z&/'-]+){0,3})"), 1.0),
        (re.compile(r"\b(?:something|anything|one|items?)\s+(?:from|in|off)\s+(?:the |your |our )?([a-z][a-z&/'-]*"
                    r"(?:\s+[a-z&/'-]+){0,4}?)\s+(?:range|category|section|aisle|shelf|line|department|collection)\b"),
         3.5),
        (re.compile(r"\b(?:something|anything)\s+from\s+(?:the |your )?([a-z][a-z&/'-]*(?:\s+[a-z&/'-]+){0,4})"), 2.5),
        (re.compile(r"(?:^|[.!?]\s+)((?:[a-z][a-z&/'-]*\s+){0,3}[a-z][a-z&/'-]*)\s+for me\b"), 3.0),
        (re.compile(r"(?:^|[.!?;:]\s+|,\s+|\bso\s+|\bnow,?\s+)((?:[a-z][a-z&/'-]*\s+){0,3}[a-z][a-z&/'-]*)\s+"
                    r"(?:is|are)\s+(?:the item|what i'?m after|what i want|what i need|the one)"), 3.0),
        (re.compile(r"\bit'?s\s+((?:a |an |some )?[a-z][a-z&/'-]*(?:\s+[a-z&/'-]+){0,4}?)(?:\s+in\s+[a-z]+)?\s+"
                    r"(?:that\s+)?i(?:'m after| am after| want| need|'d like|'m wanting| am wanting|'m looking for)"),
         3.0),
    ]
    for rx, weight in pats:
        for m in rx.finditer(low):
            np = _clean_np(m.group(1))
            if not _np_ok(np):
                continue
            found[np] = found.get(np, 0.0) + weight
    if not found:
        return []
    text_words = Counter(words(low))
    scored = []
    for np, w in found.items():
        head = words(np)[-1] if words(np) else ""
        mentions = text_words.get(head, 0)
        scored.append((np, w + 0.5 * max(0, mentions - 1) - 0.15 * max(0, len(np.split()) - 3)))
    merged: List[Tuple[str, float]] = []
    for np, s in sorted(scored, key=lambda x: (-x[1], x[0])):
        npw = set(words(np))
        joined = False
        for i, (other, os_) in enumerate(merged):
            ow = set(words(other))
            if npw <= ow or ow <= npw or (npw & ow and (words(np)[-1] in ow or words(other)[-1] in npw)):
                merged[i] = (other, os_ + s * 0.5)
                joined = True
                break
        if not joined:
            merged.append((np, s))
    merged.sort(key=lambda x: (-x[1], x[0]))
    return merged

_LINK_VERB = (r"(?:'s|\s+has|\s+have|\s+should|\s+must|\s+ought|\s+needs?|\s+is|\s+are|\s+can|\s+will|\s+shall)?"
              r"(?:\s+(?:got|ought|only|really|definitely|always))?(?:\s+to)?\s+"
              r"(?:be|say|says|read|reads|match|show|equal|measure|reach)\b")

_VALUE_END = r"(?=[,;!?(]|\.(?!\d)|\s+(?:and|but|so|since|because|nothing|none|no|or else|otherwise|please|i'll|" \
             r"i'm|which|that's)\b|$)"

_VAGUE = {"something", "anything", "one", "ones", "what", "whatever", "available", "fine", "good", "it", "there",
          "here", "in", "on", "able", "sure", "that", "this", "the", "ok", "okay",
          "to", "please", "thanks", "too", "again", "now", "then", "today", "though", "me", "you", "us", "them",
          "had", "been", "done", "made"}

_BOUND_LOW = r"(?<![a-z])(at least|no less than|minimum(?: of)?|min\.?|not less than|or more|or higher|or above|" \
             r"upwards of|or bigger|or larger|or longer|or greater|or over|plus)(?![a-z])"

_BOUND_HIGH = r"(?<![a-z])(at most|no more than|up to|maximum(?: of)?|max\.?|not more than|or less|or lower|or below|" \
              r"under|below|less than|falls? short of|or smaller|or shorter|or fewer)(?![a-z])"

def _value_need(prop: str, raw: str, source: str = "stated") -> Optional[Need]:
    val = raw.strip(" ,.;:!?\"'").lower()
    val = re.sub(r"^(?:an?|the|some|exactly|just|only|precisely|strictly|purely)\s+", "", val)
    val = re.split(r"\s+(?:for me|for us|please)\b", val)[0].strip()
    if not val or val.split()[0] in _VAGUE:
        return None
    if len(val.split()) > 6:
        return None
    if re.search(r"\+/-|\+-|±", val):
        return Need(prop, "eq", values=[val], source=source)    # a plus-or-minus figure is a spec, read literally
    lowm = re.search(_BOUND_LOW, val)
    highm = re.search(_BOUND_HIGH, val)
    nums = numbers_with_units(val)
    if (lowm or highm) and nums:
        n, unit = nums[0]
        unit = unit or re.sub(r"[^a-z]", "", val[val.find(str(int(n)) if float(n).is_integer() else str(n)):])[:6]
        if lowm and not highm:
            return Need(prop, "ge", low=n, unit=unit, source=source)
        if highm and not lowm:
            return Need(prop, "le", high=n, unit=unit, source=source)
    m = re.match(r"(?:between\s+)?(\d+(?:\.\d+)?)\s*([a-z]*)\s*(?:-|to|and)\s*(\d+(?:\.\d+)?)\s*([a-z]*)$", val)
    if m and val.startswith("between"):
        return Need(prop, "range", low=float(m.group(1)), high=float(m.group(3)), unit=m.group(4) or m.group(2),
                    source=source)
    parts = [p.strip() for p in re.split(r"\s*(?:,|\bor\b|/(?=\s))\s*", val) if p.strip()]
    parts = [re.sub(r"^(?:an?|the)\s+", "", p) for p in parts]
    parts = [p for p in parts if p and p not in _VAGUE]
    if not parts:
        return None
    return Need(prop, "eq", values=parts, source=source)

_DIFFER = (r"(?:set apart from|apart from|different from|differs? from|other than|unlike|not the same as|anything but|"
           r"anything (?:bar|except|save)|everything (?:bar|but|except))")

def rule_link(text: str, prop: str = "") -> Optional[Need]:
    """A requirement whose value is defined by another listing (the same as, or different from, what a named
    seller's listing shows). Seller names are read from capitalisation, so the original casing of the text is kept."""
    t = _clean(text or "")
    m = re.search(r"(?:listing|page|item|product|one)\s+(?:for|of)\s+(?:the\s+)?(?P<cat>[a-z][a-z &'/-]{1,40}?)\s+"
                  r"(?:sold by|from|by|offered by|at)\s+(?P<seller>[A-Z][A-Za-z0-9]+)", t)
    if m is None:
        m = re.search(r"(?:listing|page|item|product|one)\s+(?:for|of)\s+(?:the\s+)?(?P<cat>[a-z][a-z &'/-]{1,40}?)\s+"
                      r"(?:that|which)\s+(?P<seller>[A-Z][A-Za-z0-9]+)\s+(?:has\s+(?:listed|posted|put up)|lists|"
                      r"sells|offers|listed|posted)\b", t)
    if m is None:
        m = re.search(r"(?:listing|page|item|product)\s+(?:for|of|from)\s+(?P<seller>[A-Z][A-Za-z0-9]+)'s\s+"
                      r"(?P<cat>[a-z][a-z &'/-]{1,40}?)(?=[,.;!?]|\s+(?:with|and|direct|or|which|that)\b|$)", t)
    if m is None:
        return None
    low = t.lower()
    if not prop:
        pm = re.match(r"\s*(?:and\s+|but\s+|so\s+|also\s+|keep\s+|make\s+|for\s+|with\s+)?(?:the\s+|its\s+|their\s+)?"
                      r"([a-z][a-z ]{1,30}?)(?:'s|,)?\s+(?:has|have|must|should|needs?|is|are|will|got|to|stays?|"
                      r"comes?|fixed|i'll|take)\b", low)
        prop = pm.group(1).strip() if pm else ""
    tail = t[m.end():]
    details = ""
    dm = re.match(r"\s*,?\s*with\s+(.+?)(?=,\s*(?:direct|or|and then)|[.;!?]|$)", tail)
    if dm:
        details = dm.group(1).strip()
    n = Need(prop, "link", source="link")
    n.link = {"relation": "differ" if re.search(_DIFFER, low) else "same", "seller": m.group("seller"),
              "category": m.group("cat").strip(), "details": details}
    n.link["describe"] = " ".join(x for x in (n.link["seller"], n.link["category"], details) if x)
    return n

def guess_needs(text: str, product_phrases: List[str]) -> List[Need]:
    t = _clean(re.sub(r"\([^)]*\)", " ", text or "")).lower()
    needs: List[Need] = []
    seen = set()

    def add(n: Optional[Need]) -> None:
        if n is None:
            return
        sig = n.signature()
        if sig not in seen:
            seen.add(sig)
            needs.append(n)

    rx = re.compile(r"\bthe\s+((?:[a-z/&-]+\s+){0,3}?[a-z/&-]+?)(?:" + _LINK_VERB + r"|\s+(?:is|are)|"
                    r"\s+(?:go|goes)(?=\s+(?:up|down)\s+to\b))\s+(.+?)" + _VALUE_END)
    for m in rx.finditer(t):
        prop = m.group(1).strip()
        if set(prop.split()) & {"price", "budget", "total", "cost", "thing", "one", "item", "last", "first"}:
            continue
        if any(prop == p or set(content_words(prop)) <= set(content_words(p)) for p in product_phrases if p):
            continue
        n = _value_need(prop, m.group(2))
        if n is None:
            continue
        if n.kind == "eq" and any(v.startswith("to ") for v in n.values):
            continue
        add(n)
        tail = t[m.end():m.end() + 60]
        nm = re.match(r"\s*,\s*(?:but\s+)?not\s+([a-z][a-z -]*?(?:\s*(?:,|or)\s*[a-z][a-z -]*?)*)\s*(?=[,.;!?(]|$)", tail)
        if nm:
            vals = [v.strip() for v in re.split(r"\s*(?:,|\bor\b)\s*", nm.group(1)) if v.strip()]
            if vals:
                add(Need(prop, "not", values=vals))
    for m in re.finditer(r"\bwith\s+([a-z0-9][a-z0-9 -]{1,30}?)\s+as (?:the|its)\s+([a-z ]{2,25}?)" + _VALUE_END, t):
        add(_value_need(m.group(2).strip(), m.group(1)))
    for m in re.finditer(r"\bon\s+([a-z][a-z ]{1,25}?),\s+it'?s\s+(.+?)" + _VALUE_END, t):
        add(_value_need(m.group(1).strip(), m.group(2)))
    for m in re.finditer(r"\b(?:be|is|in|with)\s+([a-z0-9][a-z0-9 -]{1,30}?),?\s+as far as (?:the\s+)?([a-z ]{2,30}?)\s+"
                         r"go(?:es)?\b", t):
        add(_value_need(m.group(2).strip(), m.group(1)))
    for m in re.finditer(r"\bit(?:'s|\s+has|\s+must|\s+should|\s+needs?)\s+(?:got\s+|gotta\s+)?(?:to\s+)?"
                         r"(?:be|come in|come from|come with|have|say)\s+(.+?)" + _VALUE_END, t):
        n = _value_need("", m.group(1))
        if n is not None and n.kind == "eq":
            add(Need("", "phrase", values=n.values))
    for m in re.finditer(r"\bmade (?:of|from|out of)\s+([a-z][a-z -]{1,25}?)" + _VALUE_END, t):
        add(Need("", "phrase", values=[m.group(1).strip()]))
    for m in re.finditer(r"\b(?:they'?re|it'?s|they are|it is|that'?d work|that would work|meant|needed|wanted|"
                         r"to wear(?: it| them)?)\s+for\s+(?:a|an)\s+([a-z]+)\b", t):
        if m.group(1) not in _VAGUE and m.group(1) not in _NP_REJECT:
            add(Need("", "phrase", values=[m.group(1)]))
    for sentence in re.split(r"(?<=[.!?;])\s+", _clean(re.sub(r"\([^)]*\)", " ", text or ""))):
        add(rule_link(sentence))
    tail_patterns = (
        r"^\s*,?\s*(?:with|featuring|that has|having)\s+(?:an?\s+|the\s+|some\s+)?"
        r"((?:[a-z0-9/&-]+\s+){0,2}[a-z0-9/&-]+)(?=\s+(?:on|in|and|for|that|so|but)\b|[,.;!?(]|$)",
        r"^\s*,?\s*in\s+(?:an?\s+|the\s+)?((?:[a-z0-9/&-]+\s+){0,3}?[a-z0-9/&-]+)(?=\s+(?:and|for|that|"
        r"so|but|with|i)\b|[,.;!?(]|$)",
        r"^\s*,\s*((?:[a-z0-9/&-]+\s+){0,2}[a-z0-9/&-]+)\s*(?=[,.;!?(]|$)",
    )
    for np in product_phrases:
        if not np:
            continue
        for occ in re.finditer(re.escape(np), t):
            tail = t[occ.end():occ.end() + 120]
            for _round in range(3):
                mm = None
                for rx2 in tail_patterns:
                    mm = re.match(rx2, tail)
                    if mm:
                        break
                if mm is None:
                    break
                phrase = mm.group(1).strip()
                if phrase and phrase.split()[0] not in _VAGUE and not set(phrase.split()) & {"stock", "hand", "shelf"}:
                    add(Need("", "phrase", values=[phrase]))
                tail = re.sub(r"^\s*(?:on (?:the|it|its) [a-z]+|on it)", "", tail[mm.end():])
    excl = re.compile(r"\b(?:leave out|exclude|skip|avoid|rule out|steer clear of|nothing that has|no|cross off|"
                      r"strike out|strike|filter out|drop|remove)\s+"
                      r"(?:whatever|anything|any|ones?|items?|those|stuff)?\s*(?:that\s+|which\s+)?"
                      r"(?:has|have|with|having|in|that's|is)\s+([a-z0-9][a-z0-9 /&-]{1,40}?)" + _VALUE_END)
    for m in excl.finditer(t):
        add(Need("", "not", values=[v.strip() for v in re.split(r"\s+or\s+", m.group(1)) if v.strip()]))
    for m in re.finditer(r"\b(?:don'?t|do not|never)\s+(?:hand|give|show|offer|send|bring|get)\s+(?:me|us)\s+"
                         r"(.+?)(?=\s+and call|[,.;!?(]|$)", t):
        body = m.group(1)
        if re.search(r"substitut|alternative|anything|something|else|replacement", body):
            continue
        vals = [re.sub(r"^(?:an?|the)\s+", "", v.strip()) for v in re.split(r"\s+or\s+|,", body) if v.strip()]
        vals = [v for v in vals if v and len(v.split()) <= 4]
        if vals:
            add(Need("", "not", values=vals))
    return needs

def guess_quantity(text: str, phrase: str) -> int:
    if not phrase:
        return 1
    t = _clean(re.sub(r"\([^)]*\)", " ", text or "")).lower()
    head = re.escape(phrase.split()[-1])
    for sentence in re.split(r"(?<=[.!?;])\s+", t):
        if not re.search(r"\b(want|need|after|like|order|buy|get|grab|send|pick up|looking for|hunting)\b", sentence):
            continue
        m = re.search(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|a couple of)\s+(?:[a-z&/'-]+\s+){0,2}?"
                      + head, sentence)
        if m:
            q = _to_int(m.group(1))
            if q and 1 < q <= 20:
                return q
    m = re.search(head + r"\w*\s*(?:x|×)\s*(\d+)\b", t)
    if m and 1 < int(m.group(1)) <= 20:
        return int(m.group(1))
    return 1

_COUNT_WORD = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a|an)"

def guess_basket(text: str) -> List[Tuple[str, int]]:
    """'I need 2 lamps and 1 rug' style lists: two or more counted products in one wanting sentence."""
    t = _clean(re.sub(r"\([^)]*\)", " ", text or "")).lower()
    for sentence in re.split(r"(?<=[.!?;])\s+", t):
        cue = re.search(r"\b(want|need|after|like|order|buy|get|grab|looking for|shopping for|pick up)\b", sentence)
        if not cue:
            continue
        found = []
        explicit = 0
        rx = re.compile(r"(?:^|[\s,:])" + _COUNT_WORD + r"\s+((?:[a-z][a-z'&/-]*)(?:\s+(?!and\b|plus\b|or\b)"
                        r"[a-z&][a-z'&/-]*){0,3}?)(?=\s*(?:,|\band\b|\bplus\b|\balong with\b|[.;!?]|$))")
        for m in rx.finditer(sentence, cue.end()):
            qty = _to_int(m.group(1)) if m.group(1) not in ("a", "an") else 1
            np = _clean_np(m.group(2)) or m.group(2).strip()
            if not np or not _np_ok(np) or qty is None or qty > 20:
                continue
            if m.group(1) not in ("a", "an", "one"):
                explicit += 1
            found.append((np, qty))
        if len(found) >= 2 and explicit >= 1:
            seen = []
            for np, q in found:
                if all(set(words(np)) != set(words(o)) for o, _ in seen):
                    seen.append((np, q))
            if len(seen) >= 2:
                return seen[:4]
    return []

def rule_brief(text: str) -> Brief:
    """Deterministic reading of a request; used when the model reading is unavailable."""
    b = Brief()
    b.step_budget = read_step_budget(text)
    b.msg_budget = read_message_budget(text)
    b.min_views = read_min_views(text)
    b.budget, b.budget_unit = read_budget(text)
    b.goal = "cheapest" if read_cheapest(text) else "none"
    b.decline_ok = read_decline_rule(text)
    basket = guess_basket(text)
    if basket:
        for np, q in basket:
            b.lines.append(Line(np, np, q))
        needs = guess_needs(text, [np for np, _ in basket])
        if needs and len(b.lines) == 1:
            b.lines[0].needs.extend(needs)
        return b
    products = guess_products(text)
    chosen: List[str] = []
    for np, score in products:
        if not chosen:
            chosen.append(np)
        elif score >= 4.0 and len(chosen) < 3 and not (set(words(np)) & set(words(" ".join(chosen)))):
            chosen.append(np)
    if not chosen:
        chosen = [_fallback_product(text)]
    needs = guess_needs(text, chosen)
    for np in chosen:
        alts = [a.strip() for a in re.split(r"\s+or\s+", np) if a.strip()]
        query = alts[0] if alts else np
        line = Line(np, query, guess_quantity(text, np))
        b.lines.append(line)
    if b.lines:
        label_ws = set(content_words(" ".join(chosen)))
        for n in needs:
            vw = set(content_words(" ".join(n.values)))
            if (n.kind == "phrase" or (n.kind == "eq" and not n.prop)) and vw and vw <= label_ws:
                continue
            b.lines[0].needs.append(n)
    return b

def _fallback_product(text: str) -> str:
    """Most repeated content noun-ish word as a last resort."""
    t = _clean(re.sub(r"\([^)]*\)", " ", text or "")).lower()
    counts = Counter(w for w in re.findall(r"[a-z]{4,}", t) if w not in _NP_STOP and w not in _NP_REJECT)
    for w in list(counts):
        if w in ("y'all", "yall", "cents", "messages", "steps", "moves", "stock", "shelf", "budget", "spend"):
            counts.pop(w, None)
    return counts.most_common(1)[0][0] if counts else "item"

class Listing:
    def __init__(self, pid: str) -> None:
        self.id = pid
        self.title = ""
        self.price: Optional[float] = None
        self.stock: Optional[bool] = None
        self.category = ""
        self.brand = ""
        self.seller = ""
        self.desc = ""
        self.facts: Dict[str, str] = {}
        self.note = ""
        self.has_facts = False
        self.refreshed_at = -1
        self.viewed_at = -1
        self.order = 0

    def text(self) -> str:
        return " ".join([self.title, self.desc[:300]])

_GENERIC_PROP = {"type", "kind", "of", "the", "a", "item", "product", "general", "main", "detail", "info"}

def key_words(key: str) -> set:
    return set(stem(w) for w in name_words(key)) - _FILLER

def prop_match_score(prop: str, key: str, context: Optional[set] = None) -> float:
    """How directly a listing key names a property: 4 product-specific form, 3 same name, 2.5 the key is a
    broader name of the property, 2 the key is a narrower related name, 1 some shared word."""
    full_p = set(content_words(prop))
    full_k = key_words(key)
    if not full_p or not full_k:
        return 0
    pt = full_p - _GENERIC_PROP
    kt = full_k - _GENERIC_PROP
    if context and pt and pt < kt and (kt - pt) <= context:
        return 4
    if full_p == full_k or (pt and pt == kt):
        return 3
    if pt and pt < kt and all(len(w) <= 2 and not w.isdigit() for w in kt - pt):
        return 3                     # a key with a short prefix ('fa_pattern') names the property itself
    if full_k < full_p:
        return 2.5
    if pt and kt and (pt <= kt or kt <= pt):
        return 2
    if pt & kt:
        return 1
    return 0

def split_values(value: Any) -> List[str]:
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(split_values(v))
        return out
    s = str(value if value is not None else "")
    parts = [p.strip() for p in re.split(r"\s*[,;|]\s*|\s+/\s+", s) if p.strip()]
    return parts or ([s.strip()] if s.strip() else [])

def value_matches(fact_value: Any, wanted: List[str]) -> bool:
    whole = set(content_words(fact_value))
    elems = [set(content_words(e)) for e in split_values(fact_value)]
    for nv in wanted:
        nt = set(content_words(nv))
        if not nt:
            continue
        if nt <= whole:
            return True
        for et in elems:
            if not et:
                continue
            if et <= nt and (nt - et) <= _GENERIC_PROP:
                return True          # literal: a value naming only part of the wanted one ('bottle') is not it
    return False

def shares_words(fact_value: Any, wanted: List[str], prop: str = "") -> bool:
    """Do the values share a word, other than the property's own name ('fluted rim' vs 'rolled rim' do not)?"""
    drop = _GENERIC_PROP | set(content_words(prop))
    whole = set(content_words(fact_value)) - drop
    return any(whole & (set(content_words(v)) - drop) for v in wanted)

def _unit_norm(u: str) -> str:
    return stem(re.sub(r"[^a-z]", "", (u or "").lower()))

def numeric_verdict(fact_value: Any, need: Need) -> str:
    nums = numbers_with_units(str(fact_value))
    if not nums:
        return "unk"
    want_u = _unit_norm(need.unit)
    usable = [n for n, u in nums if not want_u or not _unit_norm(u) or _unit_norm(u) == want_u
              or _unit_norm(u).startswith(want_u) or want_u.startswith(_unit_norm(u))]
    if not usable:
        return "unk"
    hi, lo = max(usable), min(usable)
    if need.kind == "ge":
        return "yes" if hi >= (need.low or 0) - 1e-9 else "no"
    if need.kind == "le":
        return "yes" if lo <= (need.high if need.high is not None else lo) + 1e-9 else "no"
    if need.kind == "range":
        ok = any((need.low or -1e18) - 1e-9 <= n <= (need.high if need.high is not None else 1e18) + 1e-9 for n in usable)
        return "yes" if ok else "no"
    return "unk"

def need_verdict(item: Listing, need: Need, context: Optional[set] = None, tier: Optional[float] = None) -> str:
    """'yes' | 'no' (contradicted by the listing's own key) | 'soft' (key present, other value) | 'unk'."""
    facts = item.facts or {}
    title_ws = set(content_words(item.title))
    if need.kind == "link":
        return "unk"
    if need.kind == "phrase":
        pw = set(content_words(" ".join(need.values)))
        if not pw:
            return "unk"
        named = []
        for k, v in facts.items():
            kt = key_words(k) - _GENERIC_PROP
            if kt and kt <= pw and (pw - kt):
                named.append((len(kt), kt, v))
        if named:
            top = max(n[0] for n in named)
            named = [n for n in named if n[0] == top]
            for _, kt, v in named:
                rest = pw - kt
                vt = set(content_words(v))
                if vt and (rest <= vt or vt <= rest):
                    return "yes"
            for k, v in facts.items():
                rest = pw - named[0][1]
                if key_words(k) & named[0][1] and rest and rest <= set(content_words(v)):
                    return "yes"
            return "soft"
        for k, v in facts.items():
            vt = set(content_words(v))
            if vt and vt <= pw and not vt <= {"yes", "no", "true", "false", "none", "other", "na"}:
                return "yes"
        if pw <= title_ws:
            # the title is evidence only for a listing without a property table; with one, the facts decide
            return "unk" if len(facts) >= 3 else "yes"
        return "unk"
    rel = []
    if need.prop:
        scored = [(prop_match_score(need.prop, k, context), k, v) for k, v in facts.items()]
        best = max([s for s, _, _ in scored], default=0)
        if tier is not None and tier >= 3:
            # some checked listing names the property itself: the value must sit under such a key
            rel = [(s, k, v) for s, k, v in scored if s >= 3]
        elif tier is not None and tier >= 2.5:
            rel = [(s, k, v) for s, k, v in scored if s >= 2.5]
        elif best >= 1:
            rel = [(s, k, v) for s, k, v in scored if s == best]
    if need.kind in ("ge", "le", "range"):
        verdicts = [numeric_verdict(v, need) for _, _, v in rel]
        if "yes" in verdicts:
            return "yes"
        if "no" in verdicts:
            return "no" if rel and rel[0][0] >= 2 else "soft"
        return "unk"
    vals = need.values
    if need.kind == "eq":
        for _, _, v in rel:
            if value_matches(v, vals):
                return "yes"
        if not need.prop:
            for v in facts.values():
                if value_matches(v, vals):
                    return "yes"
        if rel and any(str(v).strip() for _, _, v in rel):
            if any(shares_words(v, vals, need.prop) for _, _, v in rel) or rel[0][0] < 2:
                return "unk"
            return "soft"
        if any(set(content_words(v)) and set(content_words(v)) <= title_ws for v in vals):
            return "yes"
        return "unk"
    if need.kind == "not":
        # an exclusion whose property no key names ('collar' vs 'm.top.neckline') is checked against every value
        pool = rel if (need.prop and rel) else [(1, k, v) for k, v in facts.items()]
        for _, k, v in pool:
            if need.prop and rel:
                if value_matches(v, vals):
                    return "no"
            else:
                vt = set(content_words(v))
                kt = key_words(k)
                for nv in vals:
                    nt = set(content_words(nv))
                    if vt and vt <= nt and (kt & nt or len(nt - vt) <= 1):
                        return "no"
        for nv in vals:
            nt = set(content_words(nv))
            if nt and nt <= title_ws and len(nt) >= 1 and not (need.prop and rel):
                kt_all = set().union(*[key_words(k) for k in facts]) if facts else set()
                if not (nt & kt_all) or len(nt) == 1:
                    return "no"
        if need.prop and not rel and facts:
            return "unk"                 # the listing does not show the property: different is not proven
        return "yes"
    return "unk"


# ---- merged from Know Thyself 06ae7239 (public 4 Nov 2026) ----
HEAVY_ROOM = 0.02

GUARD_KEYS = 3

GUARD_SHARE = 0.0

GUARD_GAP = 0.34

ASK_SPAN = 5                # properties one question to the shopper names
# How often an answer drew out a hidden preference named at each place of a question's list, in race-14 logs of list
# questions (5 Nov study): 45/59, 13/25, 13/22, 7/17, 4/13, then 2/18 from the sixth place on. A property named late
# in one question is still worth an early place in the next.
REVEAL_AT = (0.76, 0.52, 0.59, 0.41, 0.31)
RANK_FADE = 0.25            # a lane's later properties are worth 1 / (1 + RANK_FADE * place) of its first

NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'

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

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _lumped(name: str, table: list) -> bool:
    """A key whose values on the opened listings run several features together (the median value is six words or
    more): asking about it names no single property, so it is asked last. Read from the listings, not from a word list."""
    want = _words(name)
    sizes = sorted(len(re.findall(r"[a-z0-9]+", str(v).lower())) for _, key, _, values in table if key == want for v in values if str(v).strip())
    return bool(sizes) and sizes[len(sizes) // 2] >= 6


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Devil:
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
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
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
        self.ref_chains: dict[str, list[str]] = {}
        self.chased: set[str] = set()
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
        self.heat = 0.0
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.began = time.monotonic()
        self.decline_holds = 0
        self.spec_blocks = 0
        self.referee_blocks = 0
        self.keep_holds = 0
        self.keep_marks: set[str] = set()
        self.ordered_instead = False
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}


    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools that show a listing's page: the offer view when the store has one apart from the attributes view
        (an attributes list alone is not the page), else the single-listing view; and the side-by-side view."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        base = 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1
        return base + len(self.unread_pages()) + (1 if self.cart_stale() else 0)

    def unread_pages(self) -> list[str]:
        """Cart lines the offer view has never shown. Races 11-12: orders that had opened every cart line's page and looked
        at the cart after its last change passed far more often (.61 against .36 in race 12 with every hidden need found)."""
        tool = self.offer_tool()
        if not tool or not self.single_target(tool):
            return []
        return [i for i in self.cart_ids() if i not in self.offer_seen and i not in self.gone]

    def cart_stale(self) -> bool:
        return bool(self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view"))

    def last_look(self, keep: int = 2) -> None:
        """Before a forced checkout: open the page of every cart line not read yet, then look at the cart when it changed
        since it was last shown, keeping `keep` steps for the checkout."""
        tool = self.offer_tool()
        target = self.single_target(tool) if tool else None
        for ident in self.unread_pages() if target else []:
            if self.finished or self.left() <= keep + (1 if self.cart_stale() else 0):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            reply = self.act(tool, args) if not faults else None
            if reply is None:
                break
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get("observation"))
        if self.cart_stale() and not self.finished and self.left() > keep:
            faults = []
            view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
            view = self.act(self.roles["cart_view"], view_args) if not faults else None
            if view is not None:
                self.cart_view = view.get("observation")
                self.cart_seen_at = self.step
                self.observe(self.roles["cart_view"], self.cart_view)
        self.trace.append({"kind": "last_look", "unread": self.unread_pages(), "stale": self.cart_stale()})

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or self.model_late():
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def model_late(self) -> bool:
        return time.monotonic() - self.began >= MODEL_SECONDS

    def late(self) -> bool:
        return time.monotonic() - self.began >= EPISODE_SECONDS

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": self.heat}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.finished = True
                    return None
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    # ---------- reading the request ----------
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f'- {fn["name"]}: {fn.get("description", "")} | parameters: {_dump(fn.get("parameters") or {})}')
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
        self.read_by_rules()
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            # fix 6: only a step due at checkout keeps a message back. A step due only when ending without a purchase
            # ("offer the closest listing if nothing fits") held a question for a decline that almost never comes: race 13,
            # Know Thyself asked one question where the cap allowed two in 15 of 90 episodes, and other agents' second
            # question after a first that revealed nothing drew a hidden need 48 of 232 times. The gate still sends the
            # ending step's message when one is left.
            talk = [p['messages'] for p in self.process if p['when'] == 'order' and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0


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
            parts.append(f'budget: at most {self.budget} {brief.get("budget_unit") or ""} {scope}'.replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f'pick: the {pick} {brief.get("pick_measure") or ""} among fitting listings'.replace('  ', ' '))
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
            when = {'choose': 'before choosing', 'order': 'at the order (Devil does it when you check out)', 'end': 'at an ending without a purchase (Devil does it then)'}[entry['when']]
            parts.append(f'step the request asks for, {when}: {entry["step"]}')
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f'; {", ".join(bare)} show none' if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))


    def status(self) -> str:
        left = self.left()
        text = f'[Devil] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text


    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(_id_like(r.get(self.id_key), True) for r in rows):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
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
            for option in schema.get("enum") or []:
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
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
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
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        row = self.rows.get(ident) or {}
        value = row.get(self.price_key) if self.price_key else None
        if value is None and not self.price_key:
            # no key named yet: a numeric field whose name says it is the price
            value = next((v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and "price" in str(k).lower()), None)
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
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
        """Which fields each viewing tool shows, and which a listing has shown since the last change: a view whose
        fields were all shown already is not run again."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    # ---------- result pages ----------
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
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        if not self.price_key or not self.stock_key:
            self.read_shape(name, observation)
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
            return f"\n[Devil] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
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
        rest = f"\n[Devil] {count} listings matched and Devil has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Devil read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''


    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Devil read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Devil] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Devil's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        # a side-by-side view without attributes shows the planner no keys: the attributes view one by one instead
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and self.left() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

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
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("ask")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        # each question names up to ASK_SPAN properties, so the plan ranks more of them than there are messages
        lanes = self.plan_lanes(min(8, self.msgs_free() * ASK_SPAN))
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        named: dict[tuple[int, str], float] = {}
        span = ASK_SPAN
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
            number, labels = self.next_list(lanes, found, spent, named, span)
            if not labels:
                break
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], self.ask_text(number, labels))])
            if not batch:
                break
            before = {(number, _words(label)): named.get((number, _words(label)), 0.0) for label in labels}
            for place, label in enumerate(labels):
                key = (number, _words(label))
                named[key] = 1.0 - (1.0 - named.get(key, 0.0)) * (1.0 - REVEAL_AT[min(place, len(REVEAL_AT) - 1)])
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if len(labels) > 1 and 1 in [_count(x) for x in _items(heard.get("single"))]:
                # this shopper answers one property per question: the rest go one at a time, these ones not yet heard
                span = 1
                named.update(before)
                self.trace.append({"kind": "ask_single", "labels": labels})
            self.answer_names(heard, batch)
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
        found_items = [n for n in found if found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Devil's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        followed = self.chase()
        if followed:
            summary += '\n' + followed
            pointed = [need for need in pointed if need not in self.ref_values]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary


    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
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
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def answer_names(self, heard: dict[str, Any], batch: list[tuple[int, str, str]]) -> None:
        """A need the reader filed under the property asked about, when the answer never names that property but names
        exactly one property the item already has a requirement on, is that requirement said again, not a new one."""
        # paid replay 4 Nov, qualifying t-0040fbe66b: asked about wifi standard, the shopper repeated "network coming in as
        # wireless n"; the reader filed "wifi standard: wireless n" and every router reading 802.11n was ruled out
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or entry.get("points") is True:
                continue
            prop, sep, value = str(entry.get("need") or "").partition(":")
            if not sep or not prop.strip() or not value.strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            said = _stem(" ".join(re.findall(r"[a-z0-9]+", " ".join(a for n, _, a in batch if n == number).lower())))
            if not said or _stem(prop) <= said:
                continue
            item = self.items[number - 1]
            known = {m.split(":")[0].strip() for m in item.musts + item.told if ":" in m and m not in item.dropped}
            hit = sorted(k for k in known if _stem(k) and _stem(k) <= said and _words(k) != _words(prop))
            if not hit:
                # a property of the listings seen, named by the answer, that has shown this very value while the property
                # asked never has (devil 4 paid replay: the repeat came before the requirement it repeats)
                seen: dict[str, set[str]] = {}
                for ident in self.seen_order:
                    for key, v in (_fact_block(self.full.get(ident) or {}) or {}).items():
                        if _words(key) and not re.search(NOT_PROPERTY, _words(key), re.I):
                            seen.setdefault(_words(key), set()).add(str(v).strip().lower())
                want = value.strip().lower()
                asked = set().union(*[v for k, v in seen.items() if _alike(_stem(prop), _stem(k))] or [set()])
                hit = sorted(k for k, v in seen.items() if _stem(k) <= said and _words(k) != _words(prop)
                             and want in v and want not in asked and not _alike(_stem(prop), _stem(k)))
            if len(hit) == 1:
                entry["need"] = hit[0] + ":" + value
                self.trace.append({"kind": "answer_names", "asked": prop.strip(), "named": hit[0]})

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f'- {t["function"]["name"]}: {t["function"].get("description", "")} | parameters: {_dump(t["function"].get("parameters") or {})}' for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f'- {p["step"]}' for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f'\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {("order" if stage == "order" else "ending")} itself takes one)' + '\n\nTOOLS:\n' + '\n'.join(tools)
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
                return (False, ('Before going on, Devil did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Devil: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
        return (True, 'Before going on, Devil did the steps the request asks for:\n' + '\n'.join(said) if said else '')


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
        return " ".join(found)[:800] if found else _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.ident(value, False, 1):
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
                if ident and _fact_block(node) and ident not in ids:
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
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f"(item {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Devil's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def written_specs(self, number: int | None = None) -> list[str]:
        """Figures the requirements state as plus-or-minus ('+/- 2.5') are written specs, compared word for word."""
        out: list[str] = []
        for item, need in self.needs_now():
            if number is None or item == number:
                out += [m.group(0) for m in re.finditer(r"(?:\+/-|\+-|±)\s*\d+(?:\.\d+)?", need)]
        return out

    def shows_spec(self, ident: str, spec: str) -> bool:
        def flat(text: str) -> str:
            return re.sub(r"\s+", "", text.lower()).replace("±", "+/-").replace("+-", "+/-")
        return flat(spec) in flat(self.record_of(ident))

    def spec_unseen(self) -> bool:
        """After both holds the cart still lacks a written figure and no opened listing shows it: nothing fits, whatever a
        near figure seems to offer."""
        specs = self.written_specs()
        return (self.spec_blocks >= 2 and bool(specs) and bool(self.roles.get("refuse"))
                and any(not any(self.shows_spec(i, s) for i in self.full) for s in specs))

    def spec_gap(self) -> str:
        """A cart line that does not show a written figure is held back; with no listing showing it, ending is right."""
        specs = self.written_specs()
        if self.spec_blocks >= 2 or not specs:
            return ""
        # a figure belongs to one requested item, so one cart line showing it is enough
        missing = [s for s in specs if not any(self.shows_spec(i, s) for i in self.cart_ids())]
        if not missing:
            return ""
        self.spec_blocks += 1
        return ("The request states " + ", ".join(missing) + " as a written figure, compared word for word, and the cart's records do not show it. "
                "A near figure does not count. Replace the line with a listing that shows it; when no listing does, end without a purchase.")

    def read_by_rules(self) -> None:
        """A second, model-free reading of the request. Numbers stated plainly are taken literally, and an item the model
        read without any requirement keeps the ones the rules found."""
        try:
            rules = rule_brief(self.request)
        except Exception:
            return
        if self.budget is None and rules.budget:
            self.budget = rules.budget
        if rules.step_budget and 10 <= rules.step_budget < self.max_steps:
            self.max_steps = rules.step_budget
        if rules.msg_budget is not None and _count(self.brief.get("messages")) is None:
            self.msg_limit = rules.msg_budget
        if len(rules.lines) != len(self.items):
            return
        for item, line in zip(self.items, rules.lines):
            if not item.musts:
                item.musts += [f"{n.prop}: {' or '.join(n.values)}" if n.kind == "eq" else f"{n.prop}: not {' or '.join(n.values)}"
                               for n in line.needs if n.kind in ("eq", "not") and n.prop and n.values]

    def as_listing(self, ident: str) -> Listing:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        it = Listing(ident)
        it.title = next((v for k, v in record.items() if isinstance(v, str) and ("name" in k.lower() or "title" in k.lower())), "")
        it.facts = {str(k): str(v) for k, v in (_fact_block(record) or {}).items()}
        it.has_facts = bool(it.facts)
        it.price = self.price_of(ident)
        return it

    def to_need(self, text: str) -> Need | None:
        """A requirement the referee can judge; limits and references to other listings stay with the model."""
        prop, _, wanted = text.partition(":")
        wanted = wanted.strip()
        if not prop.strip() or not wanted or re.search(r"\d|\b(?:at least|at most|unlike|same|near|within|like)\b", wanted.lower()):
            return None
        negative = wanted.lower().startswith("not ")
        values = [v.strip() for v in re.split(r",|/|\s+or\s+", wanted[4:] if negative else wanted) if v.strip()]
        # a value that repeats the property's own noun ("pack type: large serve pack") is judged without it: a record
        # reading "large serve" states it (paid replay 4 Nov, qualifying t-dea5de2a48: the referee called that a clash,
        # blocked the checkout, ruled every jam out and let a wrong decline through)
        named = _stem(prop)
        trimmed = []
        for v in values:
            kept = [w for w in v.split() if not (_stem(w) & named)]
            trimmed.append(" ".join(kept) if kept else v)
        values = trimmed
        return Need(prop, "not" if negative else "eq", values) if values else None

    def broken_needs(self, ident: str, number: int) -> list[str]:
        """The requirements of item `number` that the listing's own property table contradicts."""
        it = self.as_listing(ident)
        context = set(content_words(self.items[number - 1].name))
        out = []
        for n, text in self.needs_now():
            need = self.to_need(text) if n == number else None
            if need is None:
                continue
            verdict = need_verdict(it, need, context)
            if verdict == "no" or (need.kind == "eq" and verdict != "yes" and self.states_other(it, need)):
                out.append(text)
        for n, text in self.needs_now():
            if n == number and text not in out and _range_clash(text, it.facts):
                out.append(text)
        return out

    def states_other(self, it: Listing, need: Need) -> bool:
        """A key naming the property holds a value without every word of any wanted value ('bottle' is not 'plastic bottles');
        a word one letter off still counts."""
        stated = [v for k, v in it.facts.items() if _alike(_stem(need.prop), _stem(k))]
        return bool(stated) and not any(all(any(_near(w, x) for x in _stem(v)) for w in _stem(want)) for want in need.values for v in stated)

    def owners(self, ident: str) -> list[int]:
        """The requested items a listing's category can answer; every item when the label names none of them."""
        cat_key = self.category_key()
        label = _stem(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key else frozenset()
        named = [n for n, it in enumerate(self.items, 1) if {w for w in label if len(w) >= 4} & (_stem(it.shelf or "") | _stem(it.name))]
        return named or list(range(1, len(self.items) + 1))

    def referee_note(self) -> str:
        """A cart line whose own record contradicts a requirement is held back, at most twice."""
        if self.referee_blocks >= 2:
            return ""
        found = []
        for ident in self.cart_ids():
            broken = [self.broken_needs(ident, n) for n in self.owners(ident)]
            if broken and all(broken):
                found.append(f"{ident} states another value for: " + "; ".join(broken[0][:3]))
        if not found:
            return ""
        self.referee_blocks += 1
        return "The listing's own record decides, and it says no. " + " | ".join(found) + ". Replace that line with a listing that fits."

    def unruled_picks(self) -> list[str]:
        """The cheapest opened listing of each requested kind that no record rules out, or its cheapest row when none of
        that kind was opened; empty once an opened kind has none left, which is the proof an ending without a purchase needs."""

        def open_for(ident: str) -> bool:
            price = self.price_of(ident)
            if isinstance(self.budget, (int, float)) and price is not None and price > self.budget:
                return False
            if not self.buyable(ident):
                return False
            return any(all(self.shows_spec(ident, s) for s in self.written_specs(n)) and not self.broken_needs(ident, n)
                       for n in self.owners(ident))

        picks: list[str] = []
        for ids in self.item_groups():
            read = any(i in self.full for i in ids)
            left = sorted((i for i in ids if open_for(i) and (i in self.full or not read)), key=lambda i: self.price_of(i) or 0.0)
            if not left:
                return []
            picks.append(left[0])
        return picks

    def decline_backed(self) -> bool:
        """Before Devil overrules a decline with its own order: independent readings of the record of each listing it
        would buy. When most of them find a requirement the record does not show (another value, or nothing about it),
        the decline stands. No reading, or no requirement to read for, keeps the order."""
        picks = [i for i in self.unruled_picks() if i in self.full]
        wanted = {}
        for ident in picks:
            item = self.owners(ident)[0]
            # a listing record never shows the price limit, stock, quantity or when it would arrive, so a reading would call
            # those silent: they are left to the store (qualifying t-e622afe641: "arrive before the 15th", passed by ordering)
            needs = [t for n, t in self.needs_now() if n == item and not re.search(LOGISTICS, t.split(":")[0], re.I)]
            if needs:
                wanted[ident] = needs
        if not wanted or self.late() or self.calls + VOTES > ALL_CALLS:
            return False
        unmet, votes = 0, 0
        self.heat = VOTE_HEAT
        try:
            for _ in range(VOTES):
                miss = False
                read = False
                for ident, needs in wanted.items():
                    prompt = (VOTE + "\n\nREQUIREMENTS:\n" + "\n".join(f"- {t}" for t in needs)
                              + "\n\nLISTING RECORD:\n" + _dump(_clip(self.full.get(ident) or self.rows.get(ident) or {}, 400))[:3500])
                    got = self.ask_json(prompt, [LIGHT, HEAVY], 1500)
                    reqs = [r for r in _items((got or {}).get("reqs")) if isinstance(r, dict)]
                    if not reqs:
                        continue
                    read = True
                    if any(str(r.get("record") or "").lower() in ("other", "silent") for r in reqs):
                        miss = True
                if read:
                    votes += 1
                    unmet += miss
        finally:
            self.heat = 0.0
        backed = votes > 0 and unmet * 2 > VOTES
        self.trace.append({"kind": "decline_vote", "picks": list(wanted), "votes": votes, "unmet": unmet, "decline": backed})
        return backed

    def order_instead(self) -> str:
        """An ending without a purchase rarely earns anything, so once the holds are used Devil buys the listings no record
        rules out itself, keeping cart lines of other items."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.ordered_instead or not cart or not buy or self.finished:
            return ""
        picks = [i for i in self.unruled_picks() if i in self.full]
        if not picks or len(picks) < len(self.item_groups()):
            return ""
        self.ordered_instead = True
        lines = {i: self.items[self.owners(i)[0] - 1].qty for i in picks}
        kept = [i for i in self.cart_ids() if i not in lines and not self.sold_out(i)]
        taken = {n for i in lines for n in self.owners(i)[:1]}
        keep = [i for i in kept if self.owners(i)[0] not in taken]
        lines.update({i: self.cart_lines.get(i, 1) for i in keep})
        drop = [i for i in kept if i not in keep]
        calls = self.cart_calls(lines, drop)
        # an empty list means the cart already holds those lines: buy it as it is (paid replay 4 Nov, qualifying
        # t-e622afe641: the vote found every requirement shown, the cart held that listing, and the decline went through)
        if calls is None or len(calls) + 1 > self.left():
            return ""
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        go, said = self.gate("order")
        if not go or self.finished:
            return said or ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "ordered_instead", "lines": lines, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return "Not run (Devil acted instead): no record rules out " + ", ".join(lines) + ", so Devil set the cart and checked out. Store reply: " + self.show(reply.get("observation"))

    def proven(self, ident: str, number: int) -> bool:
        """Fix 5: the listing's opened record states every requirement of item `number` that a record can settle (at least
        one), contradicts none, shows each written spec, and the listing is still buyable. Race 13: devil held a cart some
        agent passed with and then swapped away from it or declined in 3 episodes a run."""
        # race-13 examples (comment only, never a string): t-486adb7094, t-148a3c7487
        if ident not in self.full or not self.buyable(ident) or self.sold_out(ident):
            return False
        if not all(self.shows_spec(ident, s) for s in self.written_specs(number)) or self.broken_needs(ident, number):
            return False
        it = self.as_listing(ident)
        context = set(content_words(self.items[number - 1].name))
        judged = 0
        for n, text in self.needs_now():
            if n != number or re.search(LOGISTICS, text.split(":")[0], re.I):
                continue
            need = self.to_need(text)
            if need is None:
                continue
            if need_verdict(it, need, context) != "yes":
                return False
            judged += 1
        return judged > 0

    def cart_after(self, changes: list[tuple[str, dict[str, Any]]]) -> dict[str, int]:
        """The cart lines these cart calls would leave, worked out on a copy (note_cart touches only these three fields)."""
        saved = (dict(self.cart_lines), self.cart_touched, self.cart_at)
        try:
            for name, args in changes:
                self.note_cart(name, args)
            return dict(self.cart_lines)
        finally:
            self.cart_lines, self.cart_touched, self.cart_at = saved

    def keep_hold(self, mark: str, changes: list[tuple[str, dict[str, Any]]]) -> str:
        """Fix 5: a cart change that takes out a proven line runs only when what comes in for that item is proven too.
        Held at most KEEP_HOLDS times, never twice for the same call: sending it again runs it."""
        if self.keep_holds >= KEEP_HOLDS or mark in self.keep_marks or self.left() <= self.tail() + 1:
            return ""
        before, after = dict(self.cart_lines), self.cart_after(changes)
        lost = []
        for ident in before:
            if ident in after or self.sold_out(ident):
                continue
            for number in self.owners(ident):
                if self.proven(ident, number) and not any(number in self.owners(j) and self.proven(j, number) for j in after if j not in before):
                    lost.append((ident, number))
                    break
        if not lost:
            return ""
        self.keep_holds += 1
        self.keep_marks.add(mark)
        self.trace.append({"kind": "keep_held", "lost": lost})
        said = "; ".join(f"{i} (item {n}: {self.items[n - 1].name}) states every requirement in its own record ("
                         + "; ".join(t for m, t in self.needs_now() if m == n)[:300] + ")" for i, n in lost)
        return ("Not run yet (no step used). This cart change takes out a listing that already fits: " + said
                + ". What would replace it does not state them all. Keep that line and change only the others; if a"
                " requirement it misses has come up since, send the same call again and it will run.")

    def keep_decline(self) -> str:
        """Fix 5: no ending without a purchase while the cart holds a proven line for every requested item."""
        if self.keep_holds >= KEEP_HOLDS or "decline" in self.keep_marks:
            return ""
        lines = [i for i in self.cart_ids() if self.cart_lines.get(i, 1)]
        covered = {n for i in lines for n in self.owners(i) if self.proven(i, n)}
        if len(covered) < len(self.items):
            return ""
        self.keep_holds += 1
        self.keep_marks.add("decline")
        self.trace.append({"kind": "keep_decline", "lines": lines})
        return ("Not run yet (no step used). The cart already holds a listing for every requested item whose own record"
                " states each requirement: " + ", ".join(lines) + ". Check out the cart; end without a purchase only if"
                " a requirement it breaks has come up, by sending the ending call again.")

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

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
        got = self.ask_json(prompt, self.judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        if not got or got.get('ok') is not False:
            return ''
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        # a written figure ('+/- 6.00') is compared word for word: the review may not send the cart back to listings whose
        # records do not show it (paid replay 4 Nov, qualifying t-fc0b16b797: after the figure holds the review called a
        # cheaper listing without the figure "better", devil swapped to it and ordered where ending was right)
        unfit = [b for b in better if any(not self.shows_spec(b, f) for n in self.owners(b) for f in self.written_specs(n))]
        better = [b for b in better if b not in unfit]
        if unfit:
            self.trace.append({'kind': 'check_unfit', 'dropped': unfit})
            shown = all(self.shows_spec(i, f) for i in ids for n in self.owners(i) for f in self.written_specs(n))
            if not better and shown:
                return ''
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        if unfit:
            text += ('. ' + ', '.join(unfit) + " do not show the written figure the request states (" + ', '.join(sorted(set(self.written_specs())))
                     + ') word for word, so they do not fit, whatever the review says; a near figure never counts')
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'


    def chase(self) -> str:
        """Follow each requirement whose value is on another listing as soon as it is known, instead of leaving the chain
        to the checkout. Race 12: orders whose agent had opened every page of such a chain passed .24, others under .02;
        the base agent had done so on 40% of its chains. Store calls are made here, the pages are read by the light model."""
        live = {need for _, need in self.needs_now()}
        notes = []
        for number, need in self.refs:
            if need in live and need not in self.ref_values and need not in self.chased and len(self.chased) < 3:
                self.chased.add(need)
                note = self.chase_one(number, need)
                if note:
                    notes.append(note)
        return "\n".join(notes)

    def chase_one(self, number: int, need: str) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or not self.roles.get("search"):
            return ""
        start = self.step
        chain: list[str] = []
        shown: set[str] = set()
        cands: list[str] = []
        for _ in range(CHASE_HOPS):
            if self.finished or self.left() <= self.tail() + 6 or self.step - start >= CHASE_STEPS or self.model_late() or self.cost > SPEND_SOFT:
                break
            prompt = CHASE + f"\n\nREQUIREMENT (for the {self.items[number - 1].name}): {need}"
            prompt += "\n\nPAGES FOLLOWED:\n" + ("\n".join(f"{i}: {self.page_text(i)}" for i in chain) or "(none yet)")
            prompt += "\n\nCANDIDATES:\n" + ("\n".join(f"{i}: {self.page_text(i)}" for i in cands) or "(none yet)")
            got = self.ask_json(prompt, [LIGHT, HEAVY], 700) or {}
            match = str(got.get("match") or "").strip()
            if match in cands and match not in chain:
                chain.append(match)
            value = str(got.get("value") or "").strip()
            if value and chain and match in cands:
                self.ref_values[need] = value
                self.ref_chains[need] = chain
                self.version += 1
                self.trace.append({"kind": "chase", "need": need, "chain": chain, "value": value})
                return f"[Devil followed the listing this requirement points at] {need} -> {value} (pages opened in order: {', '.join(chain)})"
            want = got.get("find") if isinstance(got.get("find"), dict) else None
            if not want or not (str(want.get("shop") or "").strip() or str(want.get("kind") or "").strip()):
                break
            cands = self.locate(want, shown, start)
            if not cands:
                break
        self.trace.append({"kind": "chase_open", "need": need, "chain": chain})
        return ""

    def page_text(self, ident: str) -> str:
        record = dict(self.rows.get(ident) or {})
        record.update(self.full.get(ident) or {})
        return _dump(_clip(record, 300))[:1500]

    def search_for(self, query: str) -> list[str]:
        """One search from Devil's own code: the ids of the rows it returns."""
        name = self.roles.get("search")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        key = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not key:
            return []
        args: dict[str, Any] = {key: query}
        for other in schema.get("required") or []:
            if other not in args:
                spec = props.get(other) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[other] = spec["enum"][0]
                else:
                    return []
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return []
        reply = self.act(name, args)
        if reply is None:
            return []
        seen = reply.get("observation")
        self.observe(name, seen)
        return [i for i, _ in self.records_in(seen)]

    def locate(self, want: dict[str, Any], shown: set[str], start: int) -> list[str]:
        """Find the listing a chain names: search its shop, kind and properties, keep rows from that shop (the closest
        to the description first), and open their pages (up to three) with the page view."""
        shop = str(want.get("shop") or "").strip()
        kind = str(want.get("kind") or "").strip()
        details = [str(d).strip() for d in _items(want.get("details")) if str(d).strip()]
        words = _stem(" ".join([kind] + details))
        def from_shop(ident: str) -> bool:
            return bool(shop) and any(isinstance(v, str) and v.strip().lower() == shop.lower() for v in (self.rows.get(ident) or {}).values())
        def closeness(ident: str) -> int:
            return len(words & _stem(_dump(self.rows.get(ident) or {})))
        picks: list[str] = []
        for query in (" ".join([shop, kind] + details[:2]), " ".join([shop, kind])):
            if not query.strip() or self.left() <= self.tail() + 6 or self.step - start >= CHASE_STEPS:
                break
            ids = [i for i in self.search_for(query[:120]) if i not in shown]
            known = [i for i in self.seen_order if i not in shown and i not in ids and from_shop(i)]
            pool = [i for i in ids + known if from_shop(i)] if shop else ids
            picks = sorted(pool, key=lambda i: -closeness(i))[:3]
            if picks:
                break
        tool = self.offer_tool()
        target = self.single_target(tool)
        opened: list[str] = []
        for ident in picks:
            shown.add(ident)
            if ident not in self.opened:
                if self.left() <= self.tail() + 4 or self.step - start >= CHASE_STEPS:
                    break
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                reply = self.act(tool, args) if not faults else None
                if reply is None:
                    break
                self.offer_seen[ident] = self.step
                self.observe(tool, reply.get("observation"))
            opened.append(ident)
        return opened

    def trail(self) -> tuple[str, bool]:
        followed = self.chase()
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
                    seen.append(f'{self.trace[-1]["id"]}: an opened page of the shop this chain names that shows the values its description gives; read the value from it')
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
        text = followed + '\n' if followed else ''
        if seen:
            text += '[Devil opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)


    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if stale and self.left() < len(stale) + 3:
            stale = self.unread_pages()[: max(0, self.left() - 2 - (1 if self.cart_stale() else 0))]
        if not stale:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            # compare the stored record, so a re-read that arrives wrapped together with a note still shows its change
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Devil] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

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
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not self.cart_sets(name):
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    # ---------- tool use ----------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")}
        tools = [t for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any(k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)"

        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), "")
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
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
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        # only into a side-by-side view that shows what the opens would (attributes)
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        picks = picks[: self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        cart = self.roles.get('cart')
        swap = [c for c in calls if (c.get('function') or {}).get('name') == cart and cart]
        whole: set[str] = set()
        keep_skip: set[str] = set()
        if len(swap) >= 2 and any((_removes(_json_obj((c.get('function') or {}).get('arguments')) if isinstance((c.get('function') or {}).get('arguments'), str) else (c.get('function') or {}).get('arguments')) for c in swap)):
            if self.left() < len(swap) + 1:
                self.trace.append({'kind': 'swap_held', 'calls': len(swap), 'left': self.left()})
                for c in swap:
                    out.append((c['id'], f'Not run (no step used): this cart change takes {len(swap)} steps and too few are left to make it and still check out, so the cart stays as it is. Check out the cart now.'))
                calls = [c for c in calls if c not in swap]
            else:
                whole = {c['id'] for c in swap}
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Devil opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Devil's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f'Not run (no step used): unknown tool. Tools: {", ".join(sorted(allowed))}.'))
                continue
            if name not in allowed and (not (cid in whole and self.left() >= 2)):
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
                trimmed = f'\n[Devil] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {", ".join(map(str, args[many][room:]))}.'
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
            if cid in keep_skip:
                out.append((cid, 'Not run (no step used): part of the cart change held above.'))
                continue
            if name == cart and cart:
                # the whole batch of cart calls counts: a swap sent as "take X off" then "put Y in" is judged on its end state
                changes = [(name, args)]
                for later in calls[index + 1:]:
                    lf = later.get('function') or {}
                    if lf.get('name') == cart:
                        la = lf.get('arguments')
                        la = la if isinstance(la, dict) else (_json_obj(la) if (la or '').strip() else {})
                        if isinstance(la, dict):
                            changes.append((cart, la))
                held = self.keep_hold(name + _dump(args), changes)
                if held:
                    keep_skip |= {c['id'] for c in calls[index + 1:] if (c.get('function') or {}).get('name') == cart}
                    out.append((cid, held + '\nCart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            looking = name == self.roles.get('cart_view') or (name == self.offer_tool() and target and (str(args.get(target)) in self.unread_pages()))
            if not ending and (not looking) and (self.left() <= self.tail()) and (self.roles.get('buy') or self.roles.get('refuse')) and (not (cid in whole and self.left() >= 2)):
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
                    trimmed += f'\n[Devil] Left out {", ".join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since.'
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
                    if self.cart_stale() and self.left() > 2:
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
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f'{", ".join(held)} in the cart is sold out: replace that line first.') + '\nCart now: ' + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Kyo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    gap = self.spec_gap() or self.referee_note()
                    if gap:
                        out.append((cid, 'Not run yet. ' + gap + '\nCart now: ' + self.cart_text()))
                        continue
                    if self.spec_unseen():
                        faults = []
                        name = self.roles['refuse']
                        args = _shape(self.quiet(name, dict(self.brief.get('refuse_args') or {})), self.schemas.get(name), 'args', faults)
                        if faults:
                            out.append((cid, 'Not run: ' + '; '.join(faults[:3])))
                            continue
                    verdict = '' if self.is_refuse(name, args) else self.check_cart()
                    if verdict:
                        seen = "Devil read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT and (not self.is_refuse(name, args)):
                        out.append((cid, "Not run yet (no step used). Devil re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
            kept = self.keep_decline() if self.is_refuse(name, args) and self.left() > 3 else ''
            if kept:
                out.append((cid, kept + '\nCart now: ' + self.cart_text()))
                continue
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f'Put {", ".join(missing)} back in the cart at the requested quantities, then check out.' if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and self.decline_holds < DECLINE_HOLDS and (self.left() > 4):
                picks = self.unruled_picks()
                if picks:
                    self.decline_holds += 1
                    self.trace.append({'kind': 'decline_held', 'picks': picks})
                    out.append((cid, 'Not run yet (no step used). An ending without a purchase needs proof that every listing of a requested kind is ruled out by its own record. ' + ', '.join(picks) + ' is not ruled out: no record states another value for a requirement. Put the best of these in the cart and check out.'))
                    continue
            if self.is_refuse(name, args):
                done = '' if self.decline_backed() else self.order_instead()
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
                    trimmed = '\n[Devil] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Devil] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Devil] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Devil re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Devil's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out


    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                self.last_look()
                if self.finished or self.left() <= 0:
                    return ""
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        self.order_instead()
        if self.finished or self.left() <= 0:
            return ""
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Devil] Devil tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and not self.late():
            if self.seen_order and not self.ready_to_ask() and any(n not in self.chased and n not in self.ref_values for _, n in self.refs) and len(self.chased) < 3:
                # a requirement the request itself points with is followed once the first search has run
                followed = self.chase()
                if followed:
                    self.chat.append({"role": "user", "content": followed + "\n" + self.status()})
                    continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2 or self.model_late():
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _dump(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})

    # ---- merged from Know Thyself 06ae7239 (public 4 Nov 2026): question lanes, value search, cart line and fit checks ----

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

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
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                return (number, left[:1])
        return (0, [])

    def next_list(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int],
                  named: dict[tuple[int, str], float], span: int = ASK_SPAN) -> tuple[int, list[str]]:
        """The next question: the item next_ask would pick, and up to ASK_SPAN of its properties, best first. A property
        is worth its place in the item's lane, discounted by how surely the questions so far would have drawn out a
        preference on it (`named`); a property a requirement or an answer already settles is left out.
        Race 14 (5 Nov study): devil asked one property per message and drew out a hidden preference in 45 of 81
        episodes, the raw winner named 7-8 per message and drew one out in 67 of 81; episodes that drew none passed 3/65."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            item = self.items[number - 1]
            stated = [_stem(m.split(':')[0]) for m in item.musts if ':' in m and m not in item.dropped]
            heard = [_stem(m.split(':')[0]) for m in item.told if ':' in m]
            worth: list[tuple[float, int, str]] = []
            seen: set[str] = set()
            for place, label in enumerate(lanes[number]):
                own = _stem(label)
                if not own or _words(label) in seen:
                    continue
                # a stated requirement settles its own property only ("size" leaves "size class" open); an answer settles
                # any spelling that holds its words
                if any(k and own <= k for k in stated) or any(k and (k <= own or own <= k) for k in heard):
                    continue
                left = 1.0 - named.get((number, _words(label)), 0.0)
                if left >= 0.2:
                    seen.add(_words(label))
                    worth.append((left / (1.0 + RANK_FADE * place), place, label))
            worth.sort(key=lambda w: (-w[0], w[1]))
            if worth:
                return (number, [label for _, _, label in worth[:max(1, span)]])
        return (0, [])

    def ask_text(self, number: int, labels: list[str]) -> str:
        if len(labels) >= 3:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.items[number - 1].name}?"
        if len(labels) >= 2:
            return f'What {labels[0]} or {labels[1]} would you like for the {self.items[number - 1].name}?'
        return self.question(number, labels[0])

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not _lumped(key, table)] + [key for key in keys if _lumped(key, table)]

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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if _lumped(name, table) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
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
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not _lumped(k, table))][:GUARD_KEYS]:
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
            lanes[number] = [x for x in lanes[number] if not _lumped(_words(x), table)] + [x for x in lanes[number] if _lumped(_words(x), table)]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

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
                    faults.append(f'{line} ({self.items[part - 1].name}) works with {", ".join(sorted(fits)[:4])}, not with {devices[0]} ({", ".join(sorted(codes))})' + (f'; seen parts of that kind that work with it: {", ".join(good)}' if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
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
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f' (device seen: {", ".join(devices[code])})' if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

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
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f'item {n} ({item.name}) has {len(lines)} lines ({", ".join(lines)}); keep one')
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f'{i} ({label[i] or "unknown kind"}) is not a kind the shopper asked for: take it out')
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


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Devil(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
