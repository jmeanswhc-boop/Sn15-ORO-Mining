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

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42             # model turns of the tool loop at least; a longer step cap allows more (Kyo.loop_cap)
ALL_CALLS = 60              # model calls at least; a longer step cap allows more (Kyo.call_cap)
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_STOP = 0.085          # no model call goes out above this spend: the code finishes alone (one run key pays for every task)
SPEND_LIGHT = 0.035         # above this spend the late checks (cart check, decline review, swaps) try the light model first
# each task has 300 s in all on the network: Kyo keeps its own clock from the start of agent_main
DEADLINE = 275.0            # seconds Kyo plans to use, leaving room for one slow call and the finish. Race 15: Halo v2
                            # ran its tool loop to 255 s and called models to 274 s with no unfinished task in 180;
                            # love v20 at 250 used 37.5 steps a task against 47-48 for the top agents
CLOSING = 50.0              # with less time than this left, the tool loop stops and the episode heads for its finish
ASK_BASE = 70.0             # the interview's and the late code reviews' time gates count from here (the tool loop's floor up to build 11)
WALL = 290.0                # seconds from the start after which no store or model call may still be waiting (the network stops a task at 300 s)
LAST_CALL = 28.0            # with less time than this left, no model call starts: the code finishes alone
QUESTION_TIME = 45.0        # the interviewer asks only while more time than ASK_BASE plus this is left
LOGISTICS = {"options", "condition", "model", "warranty", "warranty type"}
LISTS = False               # list questions (a 2-property probe on a bundle, then up to LIST_KEYS): off, see interview
LIST_KEYS = 7               # properties an item's first question names together when the request allows 3+ messages
LISTS2 = True               # property lists on an order of two products short of messages (2-3 free): see Kyo.scarce_lists
LISTS2_KEYS = 8             # properties each of those lists names, all of one product
LISTS_ALL = True            # property lists on every order (any product count, any message cap), settled the way the
                            # answer shows (Kyo.settle_list); one "one at a time" reply turns the rest into single
                            # questions. False: lists only where scarce_lists allows
RECOVER_FLOOR = 40.0       # an ending the store turns down with a change still gets a short round while this much time is left
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 10             # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows per cart line whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words is not on the listings: drop words (keep the product noun) or browse the item's category instead; no rows never shows that the store lacks the item. Stores read a query's words differently: some count only its first two words, some need every word to match, some match any one of them; Kyo tells you once your own searches show which. So lead with the two most selective words (a required value and the product noun), keep a query to about three words, and avoid a bare number or a one-letter word among the first two; to see other rows, change the leading words. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may or may not read: browse the item's category, sliced by brand or a price ceiling when the store allows, and read the rows' descriptions.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants of the same word are fine, and a value that holds the required value's words together and in the same order counts, whatever words stand before or after them (a value "<A> <B>" meets a requirement "<B>"). The same words split apart by another word name another thing and do not count (a value "<B> <A> <C>" does not meet a requirement "<B> <C>"). A value in other words counts only as a last resort: a listing stating the required words always wins, and only when no listing you have read states them does a value that plainly means the same thing count (a synonym, never a near kind; an abbreviation or code is not the word it stands for, and sharing one word is not enough). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list attribute that includes it) or in the title or description counts. A number of another kind does not (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value, that requirement is met by the category.
- A requirement can point at another listing instead of giving a value (the same value as, a value unlike, or a value near the one on another listing that the request describes). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. Open every page of that chain before checking out. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule exactly as the page states it.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request that gives "<value>" as the "<property>" gives "<property>: <value>"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request's own words ("<property>: <relation> the <property> of the <kind> <shop> sells with <feature>", where <relation> is the request's own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in the request's own words, or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it as a must of that item in the request's own words ("<property>: <relation> the <other item> in this order") and leave it out of lookups.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about. Give these "first": true.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, how much of it is needed or used, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them. Name a hinted property by a listed key of that item when one fits the hint (a hint about who will wear or use it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry; about how much: the key for its size, volume or pack); a property no listing of the item carries comes after the item's three most carried keys. Give these "hint": true.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, avoid keys seen on only one or two of them. Pack sizes, counts and quantities per pack are properties like any other.
Rank every item on its own: what the request asks to check, then its hinted properties, then its listed keys. Give every item at least two properties: an item with no listed keys of its own category gets the properties a buyer of that kind of product chooses between.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), model names, warranty, condition and other selling terms rather than the product itself. With several items, take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property's name, no example values, no brackets, no second property.
Choose COUNT, best first (fewer only when the lists hold fewer usable keys). Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "first": true only for a property the request asks you to check, "hint": true only for a hinted property}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least <number> <unit>", "<value>", "not <value>"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least <number> <noun>", not "at least <number>").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value holding the required value's words together and in order counts whatever stands before or after them, but the same words split apart by another word do not count, a value in other words counts only when it plainly means the same thing (a synonym, never a near kind) and no listing shown states the required words, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
Category: when the CATEGORIES section shows a store category whose name carries the item's own words while a cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value holding the required value's words together and in order counts whatever stands before or after them, but the same words split apart by another word do not count, a value in other words counts only when it plainly means the same thing (a synonym, never a near kind) and no listing shown states the required words, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's own value as stated on that last page, or "", "rule_value": when that page states a rule that gives a value for this requirement, the value the rule gives, else ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

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


def _junk_key(name: str) -> bool:
    """A key name that reads as a store code, not a property a shopper names: a word without a vowel, a one-letter word,
    or a word holding a digit ("units hb", "fmlt skin care", "screen resolution e g 1920 x 1080")."""
    return any(len(w) == 1 or not re.search(r"[aeiouy]", w) or re.search(r"\d", w) for w in _words(name).split())


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


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
        cut = value[: schema["maxLength"]]
        # a text cut at the limit keeps whole words: a cut word becomes a token no listing carries
        value = cut.rsplit(" ", 1)[0] if " " in cut.strip() and not value[schema["maxLength"]].isspace() else cut
        value = value.rstrip()
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


_FIELD = re.compile(r"([a-z][a-z0-9_]{1,40}):(?:\s(.*)|$)", re.S)


def _text_value(value: str) -> Any:
    """One field of a text-line record: a run of "name=value" pairs joined by semicolons is a record of its own (a
    listing's attributes), a bare number a number, a bare true/false/none that value; anything else the text."""
    text = value.strip()
    pieces = [p.strip() for p in text.split(";")]
    if len(pieces) >= 2 and sum("=" in p for p in pieces) >= 2 and "=" in pieces[0]:
        pairs: dict[str, Any] = {}
        last = None
        for piece in pieces:
            name, sign, rest = piece.partition("=")
            if sign and name.strip() and name.strip() not in pairs:
                last = name.strip()
                pairs[last] = rest.strip()
            elif last is not None and piece:
                pairs[last] += "; " + piece
        return pairs
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _text_record(text: str) -> dict[str, Any] | None:
    """A listing a store writes as one text line of "name: value" fields joined by " | " (race 15's store: every search
    row and page came so, and a reader of structured records saw no listing at all): its fields as a record. A piece
    without a "name: " head belongs to the field before it (a title holding the separator); text that does not open
    with a field, or holds fewer than three, stays text."""
    if not isinstance(text, str) or text.count(" | ") < 2:
        return None
    record: dict[str, Any] = {}
    last = None
    for piece in text.split(" | "):
        found = _FIELD.fullmatch(piece.strip())
        if found and found.group(1) not in record:
            last = found.group(1)
            record[last] = found.group(2) or ""
        elif last is not None:
            record[last] += " | " + piece
        else:
            return None
    if len(record) < 3:
        return None
    return {name: _text_value(value) for name, value in record.items()}


def _records(node: Any, depth: int = 0) -> Any:
    """A store reply with its text-line listings read as records (see _text_record); a one-field wrapper around such
    a listing (a page given as {"value": "<the line>"}) is that record. Structured replies come back unchanged."""
    if depth > 6:
        return node
    if isinstance(node, str):
        record = _text_record(node)
        return node if record is None else record
    if isinstance(node, list):
        return [_records(value, depth + 1) for value in node]
    if isinstance(node, dict):
        out = {key: _records(value, depth + 1) for key, value in node.items()}
        if len(out) == 1:
            (key, value), = out.items()
            if isinstance(value, dict) and isinstance(node[key], str):
                return value
        return out
    return node


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


def _counted(text: str) -> tuple[str, ...]:
    """A query's words as a store counts them: letter and digit runs (hyphenated words whole), lower case, a plural
    s dropped; one-character pieces carry nothing in the stores seen."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower())
    return tuple(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if len(w) >= 2)


def _tokens(text: Any) -> set[str]:
    """A text's words for matching a description against a page: like _counted, but a digit group's comma is
    dropped, a decimal stays whole and one-character pieces count (a count of 2, a unit letter)."""
    flat = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text).lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", flat)}


def _held(want: tuple[str, ...], value: Any) -> str:
    """How a value holds a requirement's value words: "whole" when they stand together in order (words before or after
    them, or other list elements, do not matter), "split" when one element has them all but another word stands
    between them, "" otherwise."""
    text = str(value)
    pieces = [p for p in re.split(r"[,;/|]", text) if p.strip()]
    size = len(want)
    for piece in [text] + pieces:
        got = _counted(piece)
        if any(got[i:i + size] == want for i in range(len(got) - size + 1)):
            return "whole"
    return "split" if any(set(want) <= set(_counted(p)) for p in pieces) else ""


def _sentences(record: Any) -> str:
    """A listing's own short sentences about itself (outside its attribute block): a line that sums the listing up
    names the properties that set it apart. Long free text is left out, it names nearly everything."""
    if not isinstance(record, dict):
        return ""
    out = []
    for value in record.values():
        if isinstance(value, str):
            text = value.strip()
            if 25 <= len(text) <= 400 and text.endswith(".") and len(text.split()) >= 6:
                out.append(text.lower())
    return " ".join(out)


def _said(name: str, value: Any, text: str, plain: set[str] | frozenset[str] = frozenset()) -> str:
    """The spelling under which a sentence names a property (the key's words, or the key without a one- or two-letter
    store prefix), "*" when it only names the listing's value for it, "" when it names neither. `plain`: values the
    record gives under several keys, which name no property in particular."""
    if not text:
        return ""
    words = name.split()
    forms = [name] + ([" ".join(words[1:])] if len(words) >= 2 and len(words[0]) <= 2 and len(" ".join(words[1:])) >= 4 else [])
    for form in forms:
        if re.search(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", text):
            return form
    seen = str(value).strip().lower()
    if len(seen) >= 3 and seen not in plain and re.search(r"(?<![a-z0-9])" + re.escape(seen) + r"(?![a-z0-9])", text):
        return "*"
    return ""


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
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
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
        self.list_misses: dict[int, int] = {}
        self.swap_picks: dict[str, list[str]] = {}
        self.twins_of: dict[str, list[str]] = {}
        self.loss_carts: dict[str, set[str]] = {}
        self.split_held = False
        self.sold_blocks = 0
        self.passed_carts: list[tuple[int, float, dict[str, int]]] = []
        self.cheap_held = False
        self.asked_items: set[int] = set()
        self.key_distinct: dict[tuple[str, str], int] = {}
        # (category, key) -> (opened listings whose own sentence names it, the spelling that sentence uses)
        self.key_said: dict[tuple[str, str], tuple[int, str]] = {}
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.credit_out = 0
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
        self.price_moved = False
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
        # the change the store gave when it turned an ending down, and whether the next decline was held for it once
        self.end_change = ""
        self.end_held = False
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
        self.chased: set[str] = set()
        self.links: dict[str, str] = {}
        self.sightings: dict[str, int] = {}
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
        self.split_marks: set[str] = set()
        # the text values each search argument was given, for telling the free-text argument apart
        self.arg_values: dict[tuple[str, str], set[str]] = {}
        # every search call's arguments in order, for the code's own searches on a tool that requires a mode argument
        self.search_calls: list[dict[str, Any]] = []
        self.wider_held = False
        self.paged_obs: list[Any] = []
        # what the store's keyword search does with a query's words, learned from Kyo's own searches (no extra calls):
        # first-page results per (tool, other arguments), the longest shared lead proven to change results, lead
        # lengths seen to give identical results, the learned cut and word rule, and the hints already given
        self.word_log: list[tuple[tuple[str, str], tuple[str, ...], int, tuple[str, ...], int]] = []
        self.lead_proven: dict[tuple[str, str], int] = {}
        self.lead_same: dict[tuple[str, str], dict[int, int]] = {}
        self.word_cut: int | None = None
        self.word_kind: str | None = None
        self.rule_said: set[str] = set()
        self.search_ends: dict[str, tuple[int, Any]] = {}
        # the code's own review of a late decline (once), and the store's short texts' words (re-read as records come in)
        self.gate_done = False
        self.searched_at = -1
        self.word_mark: tuple[int, int] = (-1, -1)
        self.word_bag: set[str] = set()

    # ---------- plumbing ----------
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
        if talk and self.msg_used < self.msg_limit and self.gate_due("order"):
            return 2
        # until the store has turned an ending down once, a second step stays back for the retry such a refusal needs
        return 2 if self.refused_at < 0 and (self.roles.get("buy") or self.roles.get("refuse")) else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP:
                return None
            if self.time_left() < LAST_CALL:
                self.out_of_time = True
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                if status == 402:
                    # out of credit: waiting cannot help, so the next model is tried at once; a second 402 in the task
                    # stops the model calls (Arena 13854: a lone 402 cost its task 16 s median, the next call served)
                    break
                # rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.time_left() - WAITS[tries] < LAST_CALL:
                    self.out_of_time = True
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.credit_out += 1
                if self.credit_out >= 2:
                    self.no_credit = True
                    return None
        return None

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
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        # a call may wait only as long as the task's clock allows (Arena runs 12077-12219: no call failed, the longest took 30 s)
        _llm.timeout = int(max(8, min(50, self.time_left() - 10)))
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
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or self.cost >= SPEND_STOP:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        if name == self.roles.get("search"):
            self.search_calls.append(dict(args))
            for key, value in args.items():
                if isinstance(value, str) and value.strip():
                    self.arg_values.setdefault((name, key), set()).add(value.strip().lower())
        for _ in range(4):
            wall = WALL - (time.monotonic() - self.t0)
            if wall < 4:
                return None
            _env.timeout = int(max(4, min(50, wall)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                if "observation" in data:
                    data["observation"] = _records(data["observation"])
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
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0
            # questions first when holding those messages back leaves fewer than three: races 6-7 graded their steps of
            # reporting the basket's total or getting a yes before the order at caps of 4-5 (every pass sent it), while
            # races 8-14 graded none of the message steps read from a story remark (on one race-14 task all 16 passes
            # left it unsent) and paid only for hidden needs; the step's message still goes out at the gate when one is left
            if self.msg_limit - self.reserve < 3:
                self.reserve = 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, self.checker(), 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (value on the page it points at: {self.ref_values[m]}" + (f"; a rule stated on that page gives: {self.ref_rules[m]}" if m in self.ref_rules else "") + ")" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = brief.get("pick")
        if pick == "cheapest":
            parts.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace("  ", " "))
        else:
            parts.append("pick: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(brief.get("wishes")):
            parts.append(f"wish (tie-breaker only): {wish}")
        for rule in _items(brief.get("rules")):
            parts.append(f"rule from the request: {rule}")
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        secs = self.time_left()
        if secs < CLOSING + 50:
            text += f" Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
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

    def label_fit(self, label: str, item: Item) -> int:
        """How well a store category label fits a requested item: 3 the label is the item's shelf or name, 2 every
        word of the label is among the item's words (a short label such as an abbreviation), 1 they share a word of
        four letters or more, 0 not at all."""
        mine = _stem(item.shelf or "") | _stem(item.name)
        theirs = _stem(label)
        if not theirs:
            return 0
        if _words(label) in (_words(item.shelf or ""), _words(item.name)):
            return 3
        if theirs <= mine:
            return 2
        if {w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4}:
            return 1
        return 0

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            summing = _sentences(record)
            counts: dict[str, int] = {}
            for value in block.values():
                counts[str(value).strip().lower()] = counts.get(str(value).strip().lower(), 0) + 1
            plain = {v for v, n in counts.items() if n >= 2}
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}, "said": 0, "form": ""})
                slot["n"] += 1
                form = _said(name, value, summing, plain)
                if form:
                    slot["said"] += 1
                    if form != "*":
                        slot["form"] = form
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        matched = {label for label in groups if any(self.label_fit(label, item) for item in self.items)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            group["keys"] = self.merge_keys(group["keys"])
            kept: list[tuple[str, dict[str, Any]]] = []
            # most carried first; a key every opened listing shares stays in (one value on a few cards still varies in
            # the store). Races 11-14, single products: the plan model choosing from this table (love v20) named the
            # hidden property in its first 1/3 questions 29/46% of the time, any code order of the same keys 13-15/39-41%;
            # know-thyself's free ranking by what buyers of the product choose between, 39/59%
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values or not any(v.strip() for v in values):
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
                self.key_distinct[(label, name)] = len(set(slot["per"].values()))
                # a question names the property by the spelling a listing's sentence used, else by the spelling most
                # listings carry (the store's own name for it; the shortest when tied)
                spellings = slot.get("names") or [(name, slot["n"])]
                common = min(spellings, key=lambda pair: (-pair[1], len(pair[0])))[0]
                self.key_said[(label, name)] = (slot.get("said", 0), slot.get("form") or common)
        return table

    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Stores spell one property several ways across listings (a shorter or longer key, a run-together word).
        Two keys are one property when one key's words are all in the other's and their values agree on a listing
        both carry or share a value, or when they share a word and agree on every one of at least three listings
        both carry. The merged key counts every listing carrying either and is named by its longest spelling (the
        one a question can name without also naming a neighbour)."""
        def flat(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text)

        names = sorted(keys, key=lambda n: (-len(n.split()), -keys[n]["n"], n))
        merged: dict[str, dict[str, Any]] = {}
        for name in names:
            slot = keys[name]
            mine = _stem(name)
            target = None
            for big, spec in merged.items():
                theirs = _stem(big)
                both = [i for i in slot["per"] if i in spec["per"]]
                agree = sum(1 for i in both if flat(slot["per"][i]) == flat(spec["per"][i]))
                share = bool({flat(v) for v in slot["values"]} & {flat(v) for v in spec["values"]})
                if (mine <= theirs or theirs <= mine) and (agree or share) and agree == len(both):
                    target = big
                    break
                if mine & theirs and len(both) >= 3 and agree == len(both):
                    target = big
                    break
            if target is None:
                merged[name] = {"n": slot["n"], "values": list(slot["values"]), "per": dict(slot["per"]),
                                "said": slot.get("said", 0), "form": slot.get("form", ""), "names": [(name, slot["n"])]}
                continue
            spec = merged[target]
            spec["said"] = spec.get("said", 0) + slot.get("said", 0)
            spec.setdefault("names", [(target, spec["n"])]).append((name, slot["n"]))
            # the shortest spelling a listing's own sentence used names the merged property best
            forms = [f for f in (spec.get("form"), slot.get("form")) if f]
            spec["form"] = min(forms, key=len) if forms else ""
            for ident, text in slot["per"].items():
                spec["per"].setdefault(ident, text)
            for value in slot["values"]:
                if value not in spec["values"]:
                    spec["values"].append(value)
            spec["n"] = len(spec["per"])
        return merged

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any(self.label_fit(label, item) for item in self.items)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

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
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
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
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
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

    def more_pages(self, name: str, args: dict[str, Any], observation: Any, most: int = 4) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn, `most` pages at most); a long rest is named so the agent can narrow
        the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        tail_note = ""
        # race 15, cheapest tasks: orders whose every line came from a search of at most 49 matches (seven pages of
        # seven, read in full) passed .163 of 1,073, other orders .038 of 1,249
        long_note = (f"\n[Kyo] This search matched {count} listings, more than Kyo reads, and the pick counts every fitting listing:"
                     f" search again with one more required value word (the required values plus the product noun) until a search"
                     f" matches at most {7 * shown} listings, which Kyo reads to the end; or page on with the value the last page gives.")
        # a rest of up to four pages is read when the steps allow: the cheapest fitting listing is often on a later page
        if 4 < pages <= 6 and most >= 4 and self.pages_read + pages <= AUTO_PAGES + 2 and self.left() - self.tail() - pages >= 14:
            # a result of at most seven pages is read to its end (the pick counts every fitting listing): race 15, failing cheapest orders
            # on tasks other agents passed, the passing line sat in such a partly read search in 60 of 564
            pass
        elif pages > 4 and self.pages_read + 2 <= AUTO_PAGES and self.left() - self.tail() - 2 >= 12:
            # a long rest (a capped or any-word match): the next two pages, where other agents found the cheaper fit,
            # then the hint to narrow
            pages = min(2, most)
            tail_note = long_note
        elif most < pages <= 4 and self.pages_read + most <= AUTO_PAGES and self.left() - self.tail() - most >= 12:
            # a bounded caller (the code's own searches for an order of several products): its first pages, then the hint
            pages = max(0, most)
            tail_note = long_note
        elif pages > 4 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search again leading with other selective words (a required value and the product noun) or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return tail_note
        return ("\n[Kyo read " + ("the next pages" if tail_note else "the rest") + " of this search itself, one step a page]\n" + "\n".join(texts) + tail_note)

    # ---------- the store's search rules, learned from Kyo's own searches ----------
    def query_field(self, name: str) -> str | None:
        """The free-text argument of a search tool (its words), apart from the paging argument."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        shape = self.shapes.get(name) or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and k != shape.get("next_arg")]
        if len(texts) <= 1:
            return texts[0] if texts else None
        # several text arguments: the one the agent's own searches filled with the most varied text (most distinct values,
        # then most words); none before a search shows it
        def weight(key: str) -> tuple[int, float]:
            values = self.arg_values.get((name, key)) or set()
            return (len(values), sum(len(v.split()) for v in values) / max(1, len(values)))
        best = max(texts, key=weight)
        return best if weight(best)[0] > 0 else None

    def search_extras(self, name: str, field: str) -> dict[str, Any] | None:
        """The arguments besides its words that a search the code builds must send, when the tool's schema requires
        more than the words (a mode argument beside them): each takes the value of the agent's latest search that set the
        words; a two-way choice that only searches without words have set takes the other value (they used the way that
        needs no words). None when a required argument cannot be filled that way: no search is built then (Arena worlds:
        3 of the 9 search tools the logs show require a two-way mode beside the words, and in those worlds no search the
        code built was ever sent, 354 episodes: the schema check refused each one)."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        nxt = (self.shapes.get(name) or {}).get("next_arg")
        out: dict[str, Any] = {}
        for key in schema.get("required") or []:
            if key in (field, nxt):
                continue
            worded = [a[key] for a in self.search_calls if key in a and isinstance(a.get(field), str) and a[field].strip()]
            if worded:
                out[key] = worded[-1]
                continue
            options = (props.get(key) or {}).get("enum")
            bare = {_dump(a[key]) for a in self.search_calls if key in a and not (isinstance(a.get(field), str) and a[field].strip())}
            rest = [o for o in options if _dump(o) not in bare] if isinstance(options, list) and len(options) == 2 and len(bare) == 1 else []
            if len(rest) != 1:
                return None
            out[key] = rest[0]
        return out

    def search_context(self, name: str, args: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, ...], bool] | None:
        """(tool and its other arguments, the query's counted words, whether this is a first page) for a keyword search."""
        if name != self.roles.get("search"):
            return None
        field = self.query_field(name)
        text = args.get(field) if field else None
        if not isinstance(text, str) or not text.strip():
            return None
        nxt = (self.shapes.get(name) or {}).get("next_arg")
        given = args.get(nxt) if nxt else None
        first = given in (None, "", 0, 1) or (isinstance(given, str) and given.strip() in ("", "0", "1"))
        rest = _dump(sorted((k, str(v)) for k, v in args.items() if k not in (field, nxt)))
        return (name, rest), _counted(text), first

    def learn_words(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Compare a first-page keyword search with earlier ones on the same tool and other arguments: results that stay
        the same when only words after the first k change show the store reads k words; a longer query that returns
        fewer rows shows every word must match, more rows that any may. A hint goes to the agent once per finding."""
        found = self.search_context(name, args)
        if not found or not found[2] or not isinstance(observation, dict):
            return ""
        context, words, _ = found
        rows = tuple(i for i, _ in self.records_in(observation))
        shape = self.shapes.get(name) or {}
        total = observation.get(shape.get("count")) if shape.get("count") else None
        total = total if isinstance(total, int) and not isinstance(total, bool) else len(rows)
        if not words:
            return ""
        for ctx, old, old_total, old_rows, _ in self.word_log:
            if ctx != context or old == words:
                continue
            shorter = min(len(old), len(words))
            shared = 0
            while shared < shorter and old[shared] == words[shared]:
                shared += 1
            if total != old_total or rows[:6] != old_rows[:6]:
                self.lead_proven[context] = max(self.lead_proven.get(context, 0), shared)
            elif shared and shared == shorter and len(set(rows)) >= 3:
                same = self.lead_same.setdefault(context, {})
                same[shared] = same.get(shared, 0) + 1
            if shared == shorter and total != old_total and (self.word_cut is None or max(len(old), len(words)) <= self.word_cut):
                longer_more = (total > old_total) == (len(words) > len(old))
                self.word_kind = "any" if longer_more else "all"
        self.word_log.append((context, words, total, rows, self.step))
        proven = self.lead_proven.get(context, 0)
        # a cut needs two query pairs that agree past it (one pair can agree by chance, when the extra word sits on
        # every matching listing) and a pair sharing one word fewer that differs
        cuts = sorted(k for k, n in self.lead_same.get(context, {}).items() if k > proven and n >= 2)
        self.word_cut = cuts[0] if cuts and cuts[0] == proven + 1 else None
        hints = []
        if self.word_cut and f"cut{self.word_cut}" not in self.rule_said:
            self.rule_said.add(f"cut{self.word_cut}")
            lead = "word" if self.word_cut == 1 else f"{self.word_cut} words"
            hints.append(f"this store reads only the first {lead} of a query (searches that share them return the same rows, whatever follows): "
                         f"lead with the most selective words (a required value and the product noun) and change those words to see other rows")
        if self.word_kind and self.word_kind not in self.rule_said:
            self.rule_said.add(self.word_kind)
            hints.append("a listing must match every counted word here, so no rows means one word is not on the listings: drop or swap a word"
                         if self.word_kind == "all" else
                         "a listing matching any one word comes back here, so more words bring more rows, not fewer: search one value word with the product noun")
        hints += self.cap_check(name, args, observation)
        return "".join(f"\n[Kyo, from your searches] {h}." for h in hints)

    def cap_check(self, name: str, args: dict[str, Any], observation: Any) -> list[str]:
        """A price-limit argument the store ignored: a returned row costs more than the limit sent."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        nxt = (self.shapes.get(name) or {}).get("next_arg")
        # a number argument set to the budget or to a price seen works as a price limit
        marks = [float(self.budget)] if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else []
        marks += [p for p in (self.price_of(i) for i in self.rows) if p is not None]
        limits = [(k, v) for k, v in args.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k in props and k != nxt
                  and any(abs(float(v) - m) < 0.005 for m in marks)]
        if not limits or f"cap:{name}" in self.rule_said:
            return []
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        for key, limit in limits:
            if any(p is not None and p > limit for p in prices):
                self.rule_said.add(f"cap:{name}")
                return [f"this call's {key} limit was not applied (rows above it came back), so check each row's own price"]
        return []

    def word_guard(self, name: str, args: dict[str, Any]) -> str:
        """Once the store is shown to read only the first k words, a first-page search whose first k words repeat an
        earlier search's returns those same rows: it is not sent."""
        if not self.word_cut:
            return ""
        found = self.search_context(name, args)
        if not found or not found[2]:
            return ""
        context, words, _ = found
        lead = words[: self.word_cut]
        for ctx, old, _, _, step in self.word_log:
            if ctx == context and old[: self.word_cut] == lead and len(lead) == self.word_cut:
                return (f"Not run (no step used): this store reads only the first {self.word_cut} word(s), and \"{' '.join(lead)}\" was already searched at step {step}; "
                        "its rows are above. Lead with other words.")
        return ""

    def page_guard(self, name: str, args: dict[str, Any]) -> str:
        """A next-page call for a search whose last page was already shown, or past its reported matches, returns nothing."""
        nxt = (self.shapes.get(name) or {}).get("next_arg")
        if not nxt or name != self.roles.get("search") or args.get(nxt) in (None, "", 0, 1):
            return ""
        key = _dump(sorted((k, str(v)) for k, v in args.items() if k != nxt))
        seen = self.search_ends.get(key)
        if not seen:
            return ""
        count, after = seen
        value = args.get(nxt)
        if after is None:
            return "Not run (no step used): the store showed this search's last page already (it gave no next-page value); search with other words instead."
        if isinstance(value, int) and not isinstance(value, bool) and isinstance(count, int) and value >= count > 0:
            return f"Not run (no step used): this search matched {count} listings, so there is nothing at {nxt}={value}."
        return ""

    def note_end(self, name: str, args: dict[str, Any], observation: Any) -> None:
        shape = self.shapes.get(name) or {}
        nxt, cnt, after = shape.get("next_arg"), shape.get("count"), shape.get("next")
        if not nxt or not after or name != self.roles.get("search") or not isinstance(observation, dict):
            return
        key = _dump(sorted((k, str(v)) for k, v in args.items() if k != nxt))
        count = observation.get(cnt) if cnt else None
        self.search_ends[key] = (count if isinstance(count, int) and not isinstance(count, bool) else -1, observation.get(after))

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items) or self.time_left() < CLOSING + 40:
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches return only the rows their words select,"
                " so a requirement's value finds only the listings that word selects, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int, until: float = 0.0) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps (and
        stopping once less than `until` seconds are left)."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.time_left() < until:
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
        never read, get read so the last check can weigh them against the cart's choice: the rows carrying most of the
        requirements' words first, the cheapest first among equals."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        # three a line, cheapest first, can leave out a fitting row priced between them and the line (Arena: build 04's
        # paging recovered 4 of 9 "not the best option" targets; chidori reads every cheaper row): each line gets up to
        # half the steps past the finish's and 8 more (3 at least), the rows carrying the requirements' values first
        room = max(CHEAPER_OPENS, (self.left() - self.tail() - 8) // 2)
        lanes: list[list[str]] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            lanes.append(self.value_rank(pool)[:room])
        # the lines take turns, so a long order's later lines get their rows read too; the first three a line keep the
        # 4 steps beside the finish's, the rest 8 (a swap the check asks for needs its steps), and none is read with
        # less than CLOSING + 20 s left (the check and a swap turn need that time)
        first = list(dict.fromkeys(lane[rank] for rank in range(CHEAPER_OPENS) for lane in lanes if rank < len(lane)))
        more = list(dict.fromkeys(lane[rank] for rank in range(CHEAPER_OPENS, room) for lane in lanes if rank < len(lane)))
        return self.fill_facts(first, self.tail() + 4, CLOSING + 20) + self.fill_facts(more, self.tail() + 8, CLOSING + 20)

    def held_as(self, ident: str, prop: str, want: tuple[str, ...]) -> tuple[str, str, str] | None:
        """(how, key, value) for the read attributes of a listing whose keys name the property: "whole" when one of
        them holds the wanted words together, else "split" when one holds them apart; None otherwise."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        found = [(_held(want, v), str(k), str(v)) for k, v in block.items() if _names(_stem(prop), _stem(k))]
        return next((f for f in found if f[0] == "whole"), None) or next((f for f in found if f[0] == "split"), None)

    def line_of(self, ident: str, number: int) -> bool:
        """Whether a cart line belongs to item `number`: its category label fits that item best (always with one item)."""
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return True
        label = str((self.rows.get(ident) or {}).get(cat_key) or "")
        fits = [self.label_fit(label, it) for it in self.items]
        return max(fits) == 0 or fits[number - 1] == max(fits)

    def split_lines(self) -> str:
        """A cart whose lines do not match the requested items one to one: an item without a line, or an item bought as
        several listings. Race 15: each of the 486
        paid baskets held one line per requested product, and 39 orders holding more lines than products were refused
        as wrong; an item of quantity 2 is one listing at quantity 2, not two listings at 1."""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        missing = self.items_missing()
        if missing:
            # the short side of the same rule: an order missing a requested product earns nothing
            return ("The cart has no line for " + ", ".join(f"item {n} ({self.items[n - 1].name}, quantity {self.items[n - 1].qty})" for n in missing)
                    + ". An order missing a requested item earns nothing: add the best fitting listing of each missing item at its quantity, then check out.")
        if len(cart) <= len(self.items):
            return ""
        crowded = []
        for number, item in enumerate(self.items, 1):
            mine = [i for i in cart if self.line_of(i, number)]
            if len(mine) > 1:
                crowded.append(f"item {number} ({item.name}, quantity {item.qty}): {', '.join(mine)}")
        what = "; ".join(crowded) or f"{len(cart)} lines for {len(self.items)} requested items: {', '.join(cart)}"
        return ("Each requested item is bought as one listing at its full quantity, never spread over several listings. "
                f"Now: {what}. Keep the one best listing of each item at the requested quantity, remove the other lines, then check out.")

    def cart_total(self, ids: list[str]) -> float | None:
        prices = [self.price_of(i) for i in ids]
        if any(p is None for p in prices):
            return None
        return sum(p * self.cart_lines.get(i, 1) for p, i in zip(prices, ids))

    def cheaper_passed(self) -> str:
        """Under a cheapest pick: a cart that passed the cart check earlier, under the same requirements and store
        changes, cost less than the cart now going to checkout, and its lines can still be bought. Race 15, cheapest
        tasks without a store change: orders whose final line cost more than a line of its kind carted earlier passed
        .014 (n=211), orders ending on the cheapest carted line .056 (n=1,295)."""
        if self.brief.get("pick") != "cheapest":
            return ""
        now = self.cart_ids()
        total = self.cart_total(now)
        if total is None:
            return ""
        for version, cost, lines in self.passed_carts:
            if (version == self.version and cost < total - 1e-9 and len(lines) == len(now) and set(lines) != set(now)
                    and all(self.buyable(i) and not self.sold_out(i) for i in lines)):
                listed = ", ".join(f"{i} x{n}" for i, n in lines.items())
                return (f"An earlier cart passed the check under the same requirements and cost less ({cost:g} against {total:g} now): {listed}. "
                        "The pick rule is the cheapest: put that cart back and check out, unless a line of it no longer fits.")
        return ""

    def merge_split(self) -> bool:
        """The cart still spreads an item over several listings after the hold: Kyo keeps that item's first carted line
        (the cheapest under a cheapest pick) at the item's quantity and removes the others (no model call)."""
        cart = self.roles.get("cart")
        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if not self.sold_out(i)}
        drop: list[str] = []
        for number, item in enumerate(self.items, 1):
            mine = [i for i in lines if i not in drop and self.line_of(i, number)]
            if len(mine) < 2:
                continue
            if self.brief.get("pick") == "cheapest":
                mine.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            for extra in mine[1:]:
                drop.append(extra)
            lines[mine[0]] = item.qty
        if not drop or len(lines) - len(drop) != len(self.items):
            return False
        keep = {i: n for i, n in lines.items() if i not in drop}
        calls = self.cart_calls(keep, drop) if cart else None
        if not calls or len(calls) + 2 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.book_cart(cart, args, reply.get("observation"))
        self.trace.append({"kind": "merged_split", "kept": keep, "dropped": drop})
        return True

    def swap_sold(self) -> bool:
        """At a forced finish a cart line the store called sold out cannot be bought: Kyo puts the sold-out review's first
        replacement in its place at the same quantity (no model call). False when a sold-out line has no replacement
        or the cart cannot be changed in the steps left."""
        lost = [i for i in self.cart_ids() if self.sold_out(i)]
        if not lost:
            return True
        cart = self.roles.get("cart")
        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in lost}
        for ident in lost:
            if ident not in self.swap_picks and self.time_left() > LAST_CALL + 10:
                # the sold-out review has not run for this line yet (the refusal came late): it runs now
                self.replacements([ident])
            pick = next((p for p in self.swap_picks.get(ident, []) if not self.sold_out(p) and p not in lines), None)
            if not pick:
                return False
            lines[pick] = self.cart_lines.get(ident, 1)
        calls = self.cart_calls(lines, lost) if cart else None
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.book_cart(cart, args, reply.get("observation"))
        self.trace.append({"kind": "sold_swap", "lost": lost, "cart": list(lines)})
        return not any(self.sold_out(i) for i in self.cart_ids())

    def twin_back(self) -> bool:
        """A sold-out cart line had a same-title listing (the same product from another seller) that was read and can be
        bought, but the line now in its place is another listing: at checkout Kyo puts the same-title listing back at
        that line's quantity, once per sold-out line (under a cheapest pick only when it is not dearer). Race 15, every
        fetched agent: a sold-out line replaced by a same-title listing passed .69 of 187 orders, by any other .57 of 757."""
        cart_role = self.roles.get("cart")
        if not cart_role:
            return False
        for lost, before in list(self.loss_carts.items()):
            cart = self.cart_ids()
            twins = [t for t in self.twins_of.get(lost, []) if not self.sold_out(t) and self.buyable(t)]
            added = [i for i in cart if i not in before]
            if lost in cart or not twins or len(added) != 1 or any(t in cart for t in self.twins_of.get(lost, [])):
                continue
            new, pick = added[0], twins[0]
            if self.title_of(new) == self.title_of(lost):
                continue
            mine, theirs = self.price_of(new), self.price_of(pick)
            if theirs is None or (self.brief.get("pick") == "cheapest" and (mine is None or theirs > mine + 1e-9)):
                continue
            lines = {i: self.cart_lines.get(i, 1) for i in cart if i != new}
            lines[pick] = self.cart_lines.get(new, 1)
            prices = {i: self.price_of(i) for i in lines}
            total = None if None in prices.values() else sum(prices[i] * q for i, q in lines.items())
            if isinstance(self.budget, (int, float)) and total is not None and total > self.budget + 1e-9:
                continue
            calls = self.cart_calls(lines, [new])
            if not calls or len(calls) + 2 > self.left():
                continue
            del self.loss_carts[lost]
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart_role), "args", faults)
                if faults:
                    return False
                reply = self.act(cart_role, args)
                if reply is None or self.finished:
                    return False
                self.book_cart(cart_role, args, reply.get("observation"))
            self.trace.append({"kind": "twin_back", "lost": lost, "out": new, "in": pick})
            return pick in self.cart_ids()
        return False

    def items_missing(self) -> list[int]:
        """Requested items no cart line belongs to (by the category label; a line of no known category counts for
        every item), on an order of several products whose cart holds fewer lines than items."""
        cart = self.cart_ids()
        if len(self.items) < 2 or len(cart) >= len(self.items):
            return []
        # one line serves one item: each line goes to the item whose label it fits best among those still open (a line
        # whose label fits two items equally covers only one of them), so a short cart always shows what it lacks
        cat_key = self.category_key()
        open_items = list(range(1, len(self.items) + 1))
        for ident in cart:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            best = max(open_items, key=lambda n: self.label_fit(label, self.items[n - 1]), default=None)
            if best is not None:
                open_items.remove(best)
        return open_items

    def short_texts(self, ident: str) -> list[str]:
        """The record's short texts (not its id or category), the one with most words first: the title among them."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        cat_key = self.category_key()
        texts = [str(v) for k, v in record.items() if isinstance(v, str) and k != cat_key and v != ident and 0 < len(v) <= 160]
        return sorted(texts, key=lambda t: -len(t.split()))

    def priced(self, ident: str) -> str:
        price = self.price_of(ident)
        return ident if price is None else f"{ident} at {price:g}"

    def split_hold(self) -> str:
        """Once per cart, before the last check (no step, no model call, whatever the spend): a cart line whose attribute
        for a requirement's property holds the value's words split apart by another word, while listings of its category
        hold the value whole. Races 8-14: values with words before or after the wanted ones were paid (56 listings,
        "<A> <B>" for "<B>"); values with the wanted words apart never were (64 read listings, 244 orders, 0 paid).
        A value of one word is never split, and limits, exclusions and closed lists carry words no listing writes."""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.split_marks:
            return ""
        self.split_marks.add(mark)
        cat_key = self.category_key()
        cart = self.cart_ids()
        cheap = self.brief.get("pick") == "cheapest"
        for number, need in self.needs_now():
            prop, _, value = need.partition(":")
            want = _counted(value)
            if len(want) < 2 or not prop.strip():
                continue
            for line in cart:
                held = self.held_as(line, prop, want) if self.line_of(line, number) else None
                if not held or held[0] != "split":
                    continue
                kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
                pool = [i for i in self.seen_order if i not in cart and self.buyable(i) and self.affordable(i)
                        and (not kind or str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind)]
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
                read = [i for i in pool if (self.held_as(i, prop, want) or ("",))[0] == "whole"][:4]
                rows = [i for i in pool if i not in self.full and any(_held(want, t) == "whole" for t in self.short_texts(i))][:3]
                if not read and not rows:
                    continue
                self.trace.append({"kind": "split_hold", "line": line, "need": need, "read": read, "rows": rows})
                text = (f"{line}'s {held[1]} reads \"{held[2][:60]}\": the words of the requirement \"{need.strip()}\" are there but another word"
                        " stands between them, so it names another thing and does not meet the requirement.")
                if read:
                    text += " Read listings of its category that hold the value whole: " + ", ".join(map(self.priced, read)) + "."
                if rows:
                    text += " Unread rows whose own texts hold it whole: " + ", ".join(map(self.priced, rows)) + " (read their attributes before carting)."
                pick = "the cheapest of them" if cheap else "one of them"
                return text + f" Replace {line} with {pick} that meets every other requirement too, then check out."
        return ""

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
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
        cap = self.compare_room()
        if cap < 2:
            self.widen_singly()
            return
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
            if not any(str(i) in _dump(seen) for i in picks):
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            else:
                self.compare_ok = max(self.compare_ok, len(picks))
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
                        if any(str(i) in _dump(reply.get("observation")) for i in picks[:2]):
                            self.compare_ok = max(self.compare_ok, 2)

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            # four cards per item when steps allow (a single item, or a long order's step cap): read off two cards the
            # ranking misses the shopper's property about twice as often
            room = 4 if len(self.items) == 1 or self.max_steps >= 60 else 2
            lanes.append(pool[: max(0, min(room, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[: min(12, 4 * len(lanes)) if self.max_steps >= 60 else min(8, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        # the properties to ask about come from each item's own listings: listings of another kind (a page the request
        # points at) show none of them, so an early start waits for two of each item's own (the interview reads two of
        # its own for an item that has none). Love v20's step bound stays, and the questions never wait past the time
        # they need: Arena, a 103-step bundle waited for every item's two listings until its question time was gone
        covered = self.own_opened() >= 2
        if opened >= 3 * len(self.items) + 1 and covered:
            return True
        if opened and self.time_left() < ASK_BASE + QUESTION_TIME + 40 + 20 * max(0, len(self.items) - 2):
            return True
        return self.step >= max(8, self.max_steps // 4)

    def own_opened(self) -> int:
        """The fewest opened listings any item has of its own kind (by the category label); a large number when the
        store's rows carry no category."""
        cat_key = self.category_key()
        if not cat_key:
            return 99
        counts = []
        for item in self.items:
            counts.append(sum(1 for ident in self.full if self.label_fit(str((self.rows.get(ident) or {}).get(cat_key) or ""), item) > 0))
        return min(counts) if counts else 99

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

    def scarce_lists(self, count: int) -> bool:
        """Whether the interview asks in property lists: only an order of exactly two products whose request states a
        message cap leaving two or three messages free for questions (count: msgs_free at the interview's start, the
        messages the request's steps at the order or the ending take already held back). Every other order (one
        product, three or more, four or more messages free, no stated cap) keeps single questions.
        Races 8-14, two-product orders: where the shopper answers lists (races 8 and 14), agents asking mostly in lists
        drew every hidden need .28 of the time at a cap of 3 (race 14), single questions .05. Where the shopper wants
        one question at a time (races 9-13), these orders passed about 0 either way at a cap of 2-3, while with two
        messages or more a product single questions passed .08-.25; on single products and on three or more products
        a list loses more under that shopper than it gains under the other"""
        stated = _count(self.brief.get("messages"))
        return LISTS2 and len(self.items) == 2 and stated is not None and stated >= 0 and 2 <= count <= 3

    def interview(self) -> str:
        """Ask the shopper and read each answer before choosing the next question. One plan call ranks each item's
        properties (what the request asks to check, the story's hints, then what buyers of that product choose between
        and what most listings carry), and that order stands. On an order of several
        products with three or more messages the first question to an item names two properties as a probe; an answer
        that reveals nothing or wants one question at a time turns the rest into single questions. The questions take
        turns between the items, an item whose answer revealed a preference waiting until the others had their turn.
        An order of exactly two products whose request leaves two or three messages for questions (scarce_lists) asks
        in property lists instead: every message names up to LISTS2_KEYS properties of one product, the products in
        turn A-B and the third list to the product with the fewest revealed needs (A when tied); a list leads with its
        product's best property not asked yet; only a "one at a time" answer turns the rest into single questions.
        With LISTS_ALL every order asks that way, whatever its product count or cap, and settle_list decides which of a
        list's properties count as asked (race 15: lists of 5-8 properties drew a need .24-.29 per message, single
        questions .15-.20; a "one at a time" fallback costs one message where the shopper wants single questions)."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0 or self.time_left() <= ASK_BASE + QUESTION_TIME or not self.roles.get("ask"):
            return ""
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            said, form = self.key_said.get((label, name), (0, name))
            listing.append(f"  {index}. {form} (on {carried}): " + " | ".join(values))
        # a list question names several properties of one item. Races 8-14: lists drew more needs only where the shopper
        # answers lists (no "one at a time" reply); where it asks for one at a time, a list costs the message, most of all
        # on single-product tasks. So lists only for orders of several products with 3+ messages, the first a 2-property
        # probe; any "one at a time" reply turns the rest into single questions
        # Arena checks of builds 06-10: a probe answered "one at a time" cost a bundle's guard twice (love v20 kept both),
        # and Algorithm-Analysis finds lists losing per message in races 9-12; the reader is unknown before asking, so
        # single questions for every product count (LISTS keeps the probe path for a race that answers lists)
        lists = LISTS and self.msg_limit >= 3 and len(self.items) >= 2
        # the one exception (scarce_lists): two products with two or three messages for questions, where single
        # questions leave a product with one question at most
        scarce = not lists and (LISTS_ALL or self.scarce_lists(count))
        probed = False
        reasked = False
        examples: set[str] = set()
        want = count * (LIST_KEYS if lists else 1) + len(self.items) + 2
        if scarce:
            # two lists' worth from the plan; the category's own keys (item_keys) top up each queue for a third
            want = 2 * LISTS2_KEYS + len(self.items) + 2
        prompt = PLAN.replace("COUNT", str(want)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        # know-thyself's room for the plan. Arena runs of race 8-14 tasks: 29 of 115 plan calls of agents using this
        # prompt came back without text (both models at once in 3 of love's 5); a repeat of the same prompt gets the same
        # empty reply, so one reworded try goes to the light model while time allows
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        if not _items(plan.get("ask")) and self.time_left() > ASK_BASE + QUESTION_TIME + 30:
            plan = self.ask_json(prompt + "\n\nReply now with that JSON object alone, beginning with {.", [LIGHT], 3500) or {}
        queues: dict[int, list[str]] = {}
        order: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per label: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            queue = queues.setdefault(number, [])
            if all(_words(label) != _words(other) for other in queue):
                queue.append(label)
            if number not in order:
                order.append(number)
        bare = [n for n in range(1, len(self.items) + 1) if not queues.get(n) and not self.item_keys(n, table, 1)]
        if bare and self.cat_ids(bare):
            # an item with no property to ask yet (none of its own listings opened): two of its own rows are read so it
            # gets its question too
            self.fill_facts(self.cat_ids(bare), 12)
            table = self.key_table()
        for number in range(1, len(self.items) + 1):
            # the plan's own order stands (what the request asks to check, the story's hints, then its ranking of the
            # listed keys), and the category's keys the plan did not name close the queue, so no message is left unused
            # for want of a property (with lists, up to two lists' worth of them: a product's second list reaches past
            # the plan's picks). No carried-key guard: replayed on 407 recorded single-product plans, lifting a much
            # more carried key into the first three gained no hidden need and lost 3
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 2 * LISTS2_KEYS if scarce else 12):
                if all(_words(name) != _words(other) for other in queue):
                    queue.append(name)
                    self.topup.add(name)
            if queue and number not in order:
                order.append(number)
        for number, queue in queues.items():
            if number > len(self.items) or not queue:
                continue
            # a key that reads as a store code goes behind the plain property names (Arena: "units hb" led lists)
            queue[:] = [k for k in queue if not _junk_key(k)] + [k for k in queue if _junk_key(k)]
            # the plan's first property keeps the lead; the variant-telling key (title_lead) takes the second place
            # (race 15 list place 2: reveal .39, places 4 and later .17-.23)
            lead = self.title_lead(number)
            if lead and not any(_stem(lead) <= _stem(k) or _stem(k) <= _stem(lead) for k in queue[:2]):
                queue[:] = [k for k in queue if _stem(k) != _stem(lead)]
                queue.insert(1, lead)
                self.trace.append({"kind": "title_lead", "item": number, "key": lead})
        if len(self.items) >= 3:
            # races 8-14: the first-named product of a 3+ product order carries a hidden need most often (.50), a
            # product the request ties to another (fits, goes with, matches: .52 vs .35) next, then the last (.39),
            # then the middle ones (.31)
            tied = [n for n in range(2, len(self.items) + 1) if self.tied_item(n)]
            rank_items = [1] + [n for n in tied if n != len(self.items)] + [len(self.items)] + [n for n in range(2, len(self.items)) if n not in tied]
            order.sort(key=lambda n: rank_items.index(n) if n in rank_items else len(rank_items))
        self.trace.append({"kind": "plan", "lists": lists, "scarce": scarce, "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        before_told = [len(it.told) for it in self.items]
        found: set[int] = set()
        last_item: int | None = None
        while self.msgs_free() > 0 and not self.finished and self.time_left() > ASK_BASE + QUESTION_TIME:
            if scarce:
                # every product's first list before any product's second; then the product with the fewest revealed
                # needs (none first), the one after the last asked when tied: A-B-A
                counts = {n: self.revealed(n) for n in order}
                fewest = min(counts.values(), default=0)
                pick = self.next_pick(order, queues, {n for n, c in counts.items() if c > fewest}, last_item)
            else:
                pick = self.next_pick(order, queues, found, last_item)
            if pick is None:
                break
            number, label = pick
            labels = [label]
            if lists:
                # while the shopper takes them, every question names an item's next strongest properties, best first
                size = LIST_KEYS if probed else 2
                while queues.get(number) and len(labels) < size:
                    more = queues[number].pop(0)
                    if all(_stem(more) != _stem(other) for other in labels):
                        labels.append(more)
            if scarce:
                # the product's strongest properties not asked yet (next_pick dropped the asked and the settled ones),
                # best first; a spelling inside one already in the list is skipped, as next_pick skips it once asked.
                # Properties the message's length leaves out go back to the head of the queue.
                # (a list whose answer revealed nothing or one need puts its unanswered properties first in the
                # queue: settle_list)
                while queues.get(number) and len(labels) < LISTS2_KEYS:
                    more = queues[number].pop(0)
                    if not any(_stem(more) <= _stem(other) for other in labels):
                        labels.append(more)
                kept, text = self.fit_list(number, labels)
                queues.setdefault(number, [])[:0] = labels[len(kept):]
                labels = kept
            elif len(labels) == 1:
                text = self.question(number, labels[0], self.samples(number, labels[0], table) if labels[0] in examples else None)
            else:
                text = self.list_question(number, labels)
            batch = self.send([(number, label, text)])
            if not batch:
                queues.setdefault(number, [])[:0] = labels[1:]
                break
            last_item = number
            self.asked_items.add(number)
            if self.msg_refused:
                # the store refused the message (its real limit is lower than the one read): no more questions
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            # one answer: the light model reads it (the heavy one when the light one fails)
            heard = self.hear(batch, (LIGHT, HEAVY))
            before = len(told)
            self.take_heard(heard, told, number if scarce else None)
            if scarce:
                if len(told) > before:
                    found.add(number)
                if _items(heard.get("crowded")) and len(told) == before:
                    # the shopper takes one question at a time: single questions from here on, this list's properties
                    # first, strongest first (one its answer settled is dropped by next_pick)
                    scarce = False
                    queues[number][:0] = labels
                    self.trace.append({"kind": "one_at_a_time", "item": number, "labels": labels})
                    continue
                self.settle_list(number, labels, told[before:], queues)
                self.trace.append({"kind": "list", "item": number, "labels": labels, "new": len(told) - before, "next": queues.get(number, [])[:4]})
                continue
            if len(labels) > 1 and not _items(heard.get("crowded")):
                if len(told) == before:
                    # races 8-12: lists drew fewer needs per message than single questions, unlike race 14: a list answered
                    # without a preference turns the rest into single questions, its later properties asked singly on the
                    # item's next turn
                    lists = False
                    queues[number][:0] = labels[1:]
                    self.asked_labels.append((number, labels[0]))
                    self.trace.append({"kind": "list_miss", "item": number, "labels": labels})
                    continue
                probed = True
            if (len(labels) == 1 and len(self.items) == 1 and _items(heard.get("crowded")) and len(told) == before
                    and len(label.split()) >= 2 and not reasked):
                # a single item: an answer that wants a simpler question gets the property's head word once
                reasked = True
                queues[number].insert(0, label.split()[-1])
                self.trace.append({"kind": "head_word", "label": label})
                continue
            if (len(labels) == 1 and _items(heard.get("crowded")) and len(told) == before and label not in examples
                    and len(self.samples(number, label, table)) >= 2):
                # know-thyself: an answer that wants a simpler question gets the same property once more with two of
                # its values as examples
                examples.add(label)
                queues[number].insert(0, label)
                self.trace.append({"kind": "examples", "label": label})
                continue
            if len(labels) > 1 and _items(heard.get("crowded")) and len(told) == before:
                # the shopper takes one question at a time: single questions from here on, the strongest property first
                lists = False
                queues[number][:0] = labels
                self.trace.append({"kind": "one_at_a_time", "item": number, "labels": labels})
                continue
            if len(told) > before:
                found.add(number)
                self.asked_labels.append((number, labels[0]))
                # an answer states one preference at most: the list's other strongest properties come back on the item's
                # next turn (the property the answer settled is dropped by next_pick)
                queues[number][:0] = labels[1:3]
            else:
                self.asked_labels.extend((number, one) for one in labels[:3])
            # properties named late in a list are answered less often: they get a second turn after the item's other keys
            queues[number].extend(one for one in labels[3:] if one not in queues[number])
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if len(self.items) == 1 and any(not t.startswith("dropped ") for t in told) and not pointed:
            summary += self.reveal_search(1)
        elif len(self.items) > 1:
            # every product count (single questions or lists): each item whose answers revealed a requirement (not one
            # read off another listing) gets its search, while 3 steps stay back for each item after it (hiro reads about
            # 5 more listings after its questions; love ended with 10-13 steps unused)
            due = [n for n, it in enumerate(self.items, 1)
                   if [t for t in it.told[before_told[n - 1]:] if t not in it.dropped]
                   and not any((n, t) in self.refs for t in it.told[before_told[n - 1]:])]
            for index, number in enumerate(due):
                summary += self.reveal_search(number, len(due) - index - 1)
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def settle_list(self, number: int, labels: list[str], new: list[str], queues: dict[int, list[str]]) -> None:
        """Settle what a property list's answer showed (Halo v2, published code). The shopper answers about the first
        property of the list it holds a preference on, at most one per reply. A revealed need settles its property and
        every one named before it; the properties named after it lead the product's next list. An answer that reveals
        nothing settles only the first property (a preference there would have been answered); the rest come back in
        reverse order, and after a second such answer in a row for the product they go behind its properties not yet
        named. Race 14's lenient reader: list place 1 drew the need .97-.99, places 2-3 .47-.62, 4th and later .22-.41."""
        hit = -1
        for need in new:
            if need.startswith("dropped "):
                continue
            prop = _stem(re.sub(r"^item \d+: ", "", need).split(":")[0])
            best = max(range(len(labels)), key=lambda i: len(prop & _stem(labels[i])), default=-1)
            if best >= 0 and prop & _stem(labels[best]) and (hit < 0 or best < hit):
                hit = best
        misses = 0 if hit >= 0 else self.list_misses.get(number, 0) + 1
        self.list_misses[number] = misses
        done = labels[: hit + 1] if hit >= 0 else labels[:1]
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        self.asked_labels.extend((number, one) for one in done)
        rest = [one for one in queues.get(number, []) if one not in ahead]
        queues[number] = rest + ahead if misses and misses % 2 == 0 else ahead + rest

    def next_pick(self, order: list[int], queues: dict[int, list[str]], found: set[int], last_item: int | None) -> tuple[int, str] | None:
        """The next question: the item after the one asked last that still has a property queued, items whose answers
        revealed nothing yet first. A property an answer has already settled for that item is not asked again."""
        for number, queue in queues.items():
            if number <= len(self.items):
                item = self.items[number - 1]
                told = [_stem(t.split(":")[0]) for t in item.told]
                stated = [_stem(t.split(":")[0]) for t in item.musts]
                # a property an answer settled is not asked again under any spelling that holds its words; a property
                # the request states is skipped among the code's own top-up keys (the plan already saw the request)
                queue[:] = [label for label in queue
                            if not any(k and k <= _stem(label) for k in told)
                            and not (label in self.topup and any(k and (k <= _stem(label) or _stem(label) <= k) for k in stated))]
                # a spelling of a property already asked about (its words inside the other's) is not asked again
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any(_stem(label) <= a for a in asked)]
        live = [n for n in order if queues.get(n)]
        if not live:
            return None
        # every product gets a first question before any product gets a second; then products with nothing revealed yet
        pool = [n for n in live if n not in self.asked_items] or [n for n in live if n not in found] or live
        start = (order.index(last_item) + 1) if last_item in order else 0
        for step in range(len(order)):
            number = order[(start + step) % len(order)]
            if number in pool:
                return number, queues[number].pop(0)
        return None

    def revealed(self, number: int) -> int:
        """The needs an answer or a note has revealed for an item and still in force."""
        item = self.items[number - 1]
        return sum(1 for need in item.told if need not in item.dropped)

    def title_lead(self, number: int) -> str | None:
        """The item's property that most opened listings of its kind carry and whose value their titles state: the
        property that tells the product's variants apart. Race 15, first question to each product: the plan model's
        first property named a hidden need .22 of 89 (love v20), this key .18; they agreed on 2%, so the two together
        named it .37 (Halo v2 .33, Noppe .34, Shikra .29 of 178-197)."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        keys: dict[str, list[int]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record or (cat_key and self.label_fit(str(record.get(cat_key) or ""), item) <= 0):
                continue
            title = self.title_of(ident)
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if not name or name in LOGISTICS:
                    continue
                entry = keys.setdefault(name, [0, 0])
                entry[0] += 1
                words = [w for w in re.findall(r"[a-z0-9.]+", str(value).lower()) if len(w) > 1][:3]
                if title and words and any(w in title for w in words):
                    entry[1] += 1
        settled = " ".join(item.musts).lower()
        pool = [k for k, (n, _) in keys.items() if n >= 2 and not _junk_key(k) and k not in settled]
        if not pool:
            return None
        return max(pool, key=lambda k: keys[k][0] * (0.5 + keys[k][1] / keys[k][0]))

    def item_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], limit: int = 4) -> list[str]:
        """The most carried keys of the category that matches an item (its label or the words of its name)."""
        item = self.items[number - 1]
        settled = " ".join(item.musts).lower()
        # only the category that fits this item best: in a bundle another item's shelf can share a word with it
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        picked: list[str] = []
        # the table's own order: the most carried first
        for label, name, carried, values in table:
            near = best > 0 and fits[label] == best
            said, form = self.key_said.get((label, name), (0, name))
            # a key listing purchase options, or a selling term, names no property a shopper keeps a preference on
            # (exact names only: "motorcycle model" or "chipset model" are real properties)
            if name in LOGISTICS or form in LOGISTICS:
                continue
            if near and (carried >= 2 or said) and name not in settled and form not in settled and form not in picked:
                picked.append(form)
        return picked[:limit]

    def cat_ids(self, numbers: list[int]) -> list[str]:
        """Two unopened buyable rows of each given item's own kind."""
        cat_key = self.category_key()
        if not cat_key:
            return []
        out: list[str] = []
        for number in numbers:
            item = self.items[number - 1]
            own = [i for i in self.seen_order if i not in self.full and self.buyable(i)
                   and self.label_fit(str((self.rows.get(i) or {}).get(cat_key) or ""), item) > 0]
            out += own[:2]
        return out

    def tied_item(self, number: int) -> bool:
        """The request ties this item to another of the order: one of its musts names another item (the brief writes
        such a relation as a must naming the other item)."""
        item = self.items[number - 1]
        for other in self.items:
            if other is item:
                continue
            words = {w for w in _stem(other.name) if len(w) >= 4}
            if words and any(words & _stem(must) for must in item.musts if must not in item.dropped):
                return True
        return False

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples and len(samples) >= 2 else "?")

    def samples(self, number: int, label: str, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """Two values the item's listings show for a property, none the request already states."""
        item = self.items[number - 1]
        settled = " ".join(item.musts).lower()
        for category, name, _, values in table:
            form = self.key_said.get((category, name), (0, name))[1]
            if _words(label) in (_words(name), _words(form)) and self.label_fit(category, item) > 0:
                return [v for v in values if v.strip() and v not in settled and len(v) <= 30][:2]
        return []

    def list_question(self, number: int, labels: list[str]) -> str:
        return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.items[number - 1].name}?"

    def fit_list(self, number: int, labels: list[str]) -> tuple[list[str], str]:
        """The longest head of a property list whose question the message tool takes whole, and that question: the
        last properties drop off rather than the text being cut at the tool's maxLength (a cut drops the product's
        name and the question mark). One property left is asked as a single question."""
        kept = list(labels)
        while len(kept) > 1:
            text = self.list_question(number, kept)
            args = self.message_args(text)
            if args is not None and text in args.values():
                return kept, text
            kept.pop()
        return kept, self.question(number, kept[0])

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
            self.msg_refused = isinstance(answer, dict) and bool(answer.get("error"))
            if self.msg_refused:
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str], about: int | None = None) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary. `about`:
        the item a list asked about, which takes every need, drop and free property the reading gives no item number."""
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or about or 1), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or about or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or about or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = (GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                # what a reply to the gate's own call asks for (earlier answers are the interviewer's to read)
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
                break
            if self.left() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name], "args", faults)
            if faults:
                break
            if name == self.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        self.price_moved = got.get("price_moved") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            changed.append(f"budget now {budget}")
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(":")[0]) == _words(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
            self.rearm()
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

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
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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
            cat_key = self.category_key()
            if len(self.items) > 1 and cat_key:
                # an order of several products: each cart line's own read rivals priced below it come first (races 11-14,
                # love's failed cheapest orders: 5 cheaper paid listings already read sat past the pooled rows shown)
                def kind(ident: str) -> str:
                    return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()

                front: list[str] = []
                for line in skip:
                    price = self.price_of(line)
                    if price is None:
                        continue
                    front += [i for i in order if i in self.full and i not in skip and i not in front and kind(i) == kind(line)
                              and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price][:3]
                order = front + [i for i in order if i not in front]
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def late_check_due(self) -> bool:
        """A cart holding a line for every item, not yet checked under the requirements now in force, while the loop's
        end is near (within 25 s of CLOSING) and steps remain to act on a finding."""
        ids = self.cart_ids()
        if not ids or not self.roles.get("buy") or self.items_missing() or self.check_key(ids) in self.checked:
            return False
        if self.cost >= SPEND_HARD or self.left() <= self.tail() + 2:
            return False
        return CLOSING < self.time_left() <= CLOSING + 25

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        depts = self.dept_lines(ids)
        if depts:
            prompt += "\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + depts
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if self.verdicts[key] and not lead:
            total = self.cart_total(ids)
            if total is not None:
                self.passed_carts.append((self.version, total, {i: self.cart_lines.get(i, 1) for i in ids}))
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def dept_lines(self, ids: list[str]) -> str:
        """Each cart line's store category, and per item the categories seen whose names share the item's words: a
        neighbouring category (a near kind of product on another shelf) shows up next to the right one."""
        key = self.category_key()
        if not key:
            return ""
        labels: list[str] = []
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(key) or "").strip()
            if label and label not in labels:
                labels.append(label)
        out = []
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(key) or "").strip()
            if label:
                out.append(f"{ident}: category {label!r}")
        for number, item in enumerate(self.items, 1):
            near = sorted((label for label in labels if self.label_fit(label, item)), key=lambda label: -self.label_fit(label, item))
            if near:
                out.append(f"item {number} ({item.name}): categories seen whose names share its words: " + ", ".join(repr(l) for l in near[:4]))
        return "\n".join(out)

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 6 and self.cost < SPEND_HARD and self.time_left() > CLOSING:
            self.trails += 1
            records = []
            # the opened pages first (a chain is made of them; an early link fell past the newest sixty records), up to
            # 45 of them, then rows
            newest = list(reversed(self.seen_order))
            for ident in [i for i in newest if i in self.opened][:45] + [i for i in newest if i not in self.opened]:
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    if str(chain.get("rule_value") or "").strip():
                        self.ref_rules[need] = str(chain["rule_value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 3 and self.left() > 5 and self.time_left() > CLOSING
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def link_page(self, page: Any, said: set[str], own: set[str]) -> bool:
        """Whether an opened page is the listing a requirement's words describe: one of its attributes has its property
        and its value named there, and no attribute whose property is named there holds another value (the page that
        shares the shop but not the described property is a near miss). `own` is the requirement's own property,
        which the pointed-at page holds with whatever value."""
        facts = _fact_block(page) if isinstance(page, dict) else None
        hit = False
        for key, value in (facts or {}).items():
            prop = {w for w in _tokens(_words(key)) if len(w) >= 3}
            if not prop or not prop <= said or prop <= own:
                continue
            flat = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(value))
            pieces = [_tokens(p) for p in re.split(r"[,;/|]", flat) if _tokens(p)]
            if any(p <= said for p in pieces):
                hit = True
            else:
                return False
        return hit

    def chase_links(self, observation: Any) -> str:
        """A search result while a requirement takes its value from another listing that nobody has found yet: a row
        whose shop (a short field of the row, not its category nor the item's own kind words, that the requirement
        writes with a capital) and whose category the requirement names is that listing's candidate. When such a row
        shows up unopened a second time (the loop model has passed it over once), Kyo opens it: side by side when the
        store can, else one by one, three per search and six in all, until a page holds the property the requirement
        describes the listing by (link_page). The loop model had skipped the right row of a named shop and spent its
        steps on that shop's near-miss pages; the chain's later links stay with the model, which reads a page's words."""
        rows = self.records_in(observation)
        for ident, _ in rows:
            self.sightings[ident] = self.sightings.get(ident, 0) + 1
        live = {need for _, need in self.needs_now()}
        pending = [(n, need) for n, need in self.refs if need in live and need not in self.ref_values and need not in self.links]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        if not pending or not tool or not target or self.finished or len(self.chased) >= 6:
            return ""
        many, joiner = self.compare_target(), self.roles.get("compare")
        cat_key = self.category_key()
        seen: list[str] = []
        for number, need in pending:
            item = self.items[number - 1]
            kind = _tokens(item.name + " " + item.shelf)
            own = _tokens(need.split(":", 1)[0]) if ":" in need else set()
            said = _tokens(need.split(":", 1)[1] if ":" in need else need) - own
            cands: list[str] = []
            # the rows seen before come first; the shop's other unopened rows of that kind in this result join them, so
            # the open does not narrow what the model would have looked at (Arena, a pointer task: the model opened four
            # of the shop's rows side by side, the code-run open took one and the right page went unread)
            later: list[str] = []
            for ident, row in rows:
                if ident in self.opened or ident in self.chased:
                    continue
                named = False
                for key, value in row.items():
                    if key == cat_key or not isinstance(value, str) or value == ident:
                        continue
                    words = _tokens(value)
                    if not words or len(value.split()) > 3 or not words <= said or words <= kind:
                        continue
                    # a name: the requirement writes it with a capital, as shops and brands are written
                    spot = re.search(r"\b" + re.escape(value.strip().lower()) + r"\b", need.lower())
                    if spot and need[spot.start()].isupper():
                        named = True
                # the requirement names the pointed-at listing's kind: a row of another category is another listing
                label = _tokens(row.get(cat_key) or "") if cat_key else set()
                if named and label <= said:
                    (cands if self.sightings.get(ident, 0) >= 2 else later).append(ident)
            cands = (cands + later)[: min(3, 6 - len(self.chased))] if cands else []
            if not cands or self.left() < self.tail() + 8 or self.time_left() < CLOSING + 30 or self.finished:
                continue
            if many and joiner and not self.compare_bare() and len(cands) > 1:
                plan = [(joiner, {many: cands[: self.compare_room()]})]
            else:
                plan = [(tool, {target: ident}) for ident in cands]
            for name, args in plan:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(name), "args", faults)
                if faults:
                    break
                self.chased.update(cands if name == joiner else [args.get(target)])
                reply = self.act(name, args)
                if reply is None or self.finished:
                    break
                page = reply.get("observation")
                self.observe(name, page)
                seen.append(self.show(page)[:1500 * (len(cands) if name == joiner else 1)])
                found = next((i for i, r in self.records_in(page) if i in cands and self.link_page(r, said, own)), None)
                if found:
                    self.links[need] = found
                    break
        if not seen:
            return ""
        self.trace.append({"kind": "chase", "opened": sorted(self.chased), "links": dict(self.links)})
        # the pages only: a shop's near-miss pages can share every property this matching reads, so which one the
        # requirement describes stays the model's call
        return ("\n[Kyo opened rows of the shop a requirement's value points at; find among them the one it describes]\n"
                + "\n".join(seen))

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
        if not stale or self.left() < len(stale) + 3:
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
            self.loss_carts.setdefault(ident, set(self.cart_ids()))
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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

    def title_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        key = next((k for k in record if "title" in str(k).lower()), None) or next((k for k in record if str(k).lower() in ("name", "product_name")), None)
        text = str(record.get(key) or "") if key else (self.short_texts(ident) or [""])[0]
        return re.sub(r"\s+", " ", text).strip().lower()

    def same_titled(self, ident: str) -> list[str]:
        """Buyable listings with the sold-out line's own title: the same product from another seller, inside the budget.
        Race 15, every fetched agent: a sold-out line replaced by such a listing passed .69 of 187 orders, by any other
        listing .57 of 757. When none was seen, one store search for the title runs (no model call)."""
        want = self.title_of(ident)
        if not want:
            return []
        others = sum((self.price_of(i) or 0.0) * self.cart_lines.get(i, 1) for i in self.cart_ids() if i != ident)
        qty = self.cart_lines.get(ident, 1)

        def found() -> list[str]:
            return [i for i in self.seen_order if i != ident and not self.sold_out(i) and self.buyable(i) and self.title_of(i) == want
                    and not (isinstance(self.budget, (int, float)) and self.price_of(i) is not None
                             and others + self.price_of(i) * qty > self.budget + 1e-9)]
        got = found()
        search = self.roles.get("search")
        field = self.query_field(search) if search else None
        extras = self.search_extras(search, field) if search and field else None
        if got or extras is None or self.left() < self.tail() + 4 or self.time_left() < LAST_CALL + 25:
            return got
        words = re.findall(r"[A-Za-z0-9]+", want)[:8]
        faults: list[str] = []
        args = _shape({**extras, field: " ".join(words)}, self.schemas.get(search), "args", faults)
        if faults or not words or (search + _dump(args)) in self.read_cache:
            return got
        reply = self.act(search, args)
        if reply is not None and not self.finished:
            self.observe(search, reply.get("observation"))
            self.read_cache[search + _dump(args)] = (self.version, self.step, self.show(reply.get("observation")))
            got = found()
        self.trace.append({"kind": "twin_search", "lost": ident, "found": got[:4]})
        return got

    def replacements(self, lost: list[str]) -> list[str]:
        # a sold-out line leaves the order short whatever the spend: the review runs up to the hard gate
        twins = list(dict.fromkeys(i for ident in lost for i in self.same_titled(ident)))
        for ident in lost:
            self.twins_of[ident] = twins[:4]
        if self.cost > SPEND_HARD:
            for ident in lost:
                self.swap_picks[ident] = twins[:4]
            return twins[:4]
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            for ident in lost:
                self.swap_picks[ident] = twins[:4]
            return twins[:4]
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.checker(), 1200)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        # the same product from another seller leads (under a cheapest pick only when it is not dearer than the review's best)
        if twins and (self.brief.get("pick") != "cheapest" or not ranked
                      or (self.price_of(twins[0]) or float("inf")) <= (self.price_of(ranked[0]) or float("inf"))):
            ranked = twins + [i for i in ranked if i not in twins]
        else:
            ranked = ranked + [i for i in twins if i not in ranked]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits, "twins": twins[:4]})
        for ident in lost:
            self.swap_picks[ident] = ranked[:4]
        return ranked[:4]

    def audit(self, limit: float = SPEND_SOFT, keep: int = 5) -> str:
        if self.audited or self.cost > limit:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named and self.time_left() > LAST_CALL + 35:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again. Races
            # 8-14: ranked by the requirements' words they carry, the paid row was among the first four unread rows in
            # half of love's declines that had it in their search rows; the review's own picks join that ranking
            asked = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and str(i) not in self.full and not self.sold_out(str(i))]
            # chidori's last pass before a decline: one search per item built from its requirements' values (races 13-14:
            # chidori declined a payable task 7 and 11 times less often than love); its unread rows join the ranking
            found: list[str] = []
            if self.searched_at != self.version and self.left() >= keep + 6 and self.time_left() > LAST_CALL + 45:
                self.searched_at = self.version
                for number in range(1, min(2, len(self.items)) + 1):
                    if self.left() < keep + 6:
                        break
                    # a bundle's item searches under the word its own rows carry, two extra pages at most (as after the questions)
                    many = len(self.items) > 1
                    hit = self.value_search(number, keep + 4, self.item_noun(self.items[number - 1]) if many else None, 2 if many else 4)
                    if hit:
                        found += [i for i in self.unread_rows() if i in hit[1]][:4]
                self.trace.append({"kind": "audit_search", "found": found[:8]})
            rows = self.value_rank(list(dict.fromkeys(asked + found + self.unread_rows()[:4])))
            if rows and self.open_rows(rows, keep):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def store_words(self) -> set[str]:
        """The words of the store's own short texts (titles, attribute values) over every record seen."""
        mark = (len(self.rows), len(self.full))
        if mark == self.word_mark:
            return self.word_bag
        bag: set[str] = set()

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, str):
                if len(node) <= 80:
                    bag.update(_counted(node))
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node[:40]:
                    walk(value, depth + 1)

        for record in list(self.rows.values()) + list(self.full.values()):
            walk(record, 0)
        self.word_mark, self.word_bag = mark, bag
        return bag

    def need_words(self, number: int | None = None) -> list[tuple[int, list[str]]]:
        """Each requirement in force (of one item, or all) as the words of its value that the store's own short texts
        also use: limit words and the like fall away without a word list, since listings do not carry them."""
        known = self.store_words()
        out = []
        for item_no, need in self.needs_now():
            if number is not None and item_no != number:
                continue
            value = need.split(":", 1)[1] if ":" in need else need
            words = list(dict.fromkeys(w for w in _counted(value) if w in known))
            if words:
                out.append((item_no, words))
        return out

    def value_rank(self, ids: list[str]) -> list[str]:
        """Listings ordered by how much of the requirements' value words their records carry (a share per requirement,
        summed), the cheapest first among equals under a cheapest pick; ties keep the given order."""
        needs = self.need_words()
        cheap = self.brief.get("pick") == "cheapest"

        def score(ident: str) -> tuple[float, float]:
            text = set(_counted(_dump(self.full.get(ident) or self.rows.get(ident) or {})))
            got = sum(sum(w in text for w in words) / len(words) for _, words in needs)
            price = self.price_of(ident) if cheap else None
            return (-got, price if price is not None else 0.0)

        return sorted(ids, key=score)

    def affordable(self, ident: str) -> bool:
        price = self.price_of(ident)
        cap = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        if price is None or not cap:
            return True
        if len(self.items) == 1 and self.brief.get("budget_scope") != "each":
            price *= max(1, self.items[0].qty)
        return price <= cap

    def unread_rows(self) -> list[str]:
        """Buyable, affordable rows of a requested kind whose page was never read, the ones carrying most of the
        requirements' words first."""
        cat_key = self.category_key()
        pool = [i for i in reversed(self.seen_order) if i not in self.full and self.buyable(i) and self.affordable(i)
                and (not cat_key or any(self.label_fit(str((self.rows.get(i) or {}).get(cat_key) or ""), it) > 0 for it in self.items))]
        return self.value_rank(pool)

    def decline_gate(self) -> str:
        """A decline no review has looked at (it came with few steps or little time left, or the code is ending the
        episode). Races 8-14 single-product tasks: love declined although a fitting listing existed 15 times more than
        the race leaders, 7 of 9 such race 11-14 declines with exactly 5 steps left, past the review's step gate. The
        code reviews the listings seen (opening the unread rows that carry most of the requirements' words); a fit the
        review names is carted and checked out by the code only when an independent check passes the whole cart."""
        if self.audited or self.gate_done or self.completed or not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        if self.left() < len(self.items) + 2 or self.time_left() < LAST_CALL + 30 or self.cost > SPEND_HARD:
            return ""
        self.gate_done = True
        self.audit(SPEND_HARD, len(self.items) + 2)
        self.trace.append({"kind": "decline_gate", "leads": list(self.leads)})
        if not self.leads or self.time_left() < LAST_CALL + 15:
            return ""
        self.confirmed = self.leads[0]
        return self.complete_cart()

    def owns(self, label: str, item: Item) -> bool:
        """A store category label belongs to this item of the order: it fits the item, and no other item better (a
        tablet's label also shares a word with a tablet case's)."""
        fit = self.label_fit(label, item)
        return fit > 0 and all(self.label_fit(label, other) <= fit for other in self.items if other is not item)

    def item_noun(self, item: Item) -> str:
        """The product word a code-built search for an item of a several-product order leads with: of the shelf label's
        words, then the name's, the one that most sets the item's own rows (their category label, see owns) apart from
        the other rows seen, outside the category field; a word another item of the order also names only when no other
        is left; the label's and the later word among near equals; in the spelling those rows use. The name's last word
        when no row tells (no category field, or no rows of another kind seen to compare with). A word every listing
        carries (a store writing the same word into each title) sets nothing apart."""
        word_re = r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
        raw = re.findall(word_re, item.name)
        cat_key = self.category_key()
        others = {w for it in self.items if it is not item for w in _counted(it.name + " " + it.shelf)}
        mine: list[set[str]] = []
        rest: list[set[str]] = []
        forms: dict[str, dict[str, int]] = {}
        for record in self.rows.values():
            own = bool(cat_key) and self.owns(str(record.get(cat_key) or ""), item)
            bag: set[str] = set()
            for key, value in record.items():
                if key == cat_key or not isinstance(value, str) or len(value) > 300:
                    continue
                for word in re.findall(word_re, value):
                    counted = _counted(word)
                    if counted:
                        bag.add(counted[0])
                        if own:
                            spell = forms.setdefault(counted[0], {})
                            spell[word.lower()] = spell.get(word.lower(), 0) + 1
            (mine if own else rest).append(bag)
        if not cat_key or not mine or not rest:
            return raw[-1] if raw else ""
        best: tuple[tuple[bool, float, int, int], str] | None = None
        for rank, phrase in enumerate((item.shelf, item.name)):
            for place, word in enumerate(re.findall(word_re, phrase or "")):
                counted = _counted(word)
                if not counted or re.fullmatch(r"\d[\d.,/]*", word):
                    continue
                share = sum(counted[0] in bag for bag in mine) / len(mine)
                apart = share - sum(counted[0] in bag for bag in rest) / len(rest)
                score = (counted[0] not in others, round(apart, 1), -rank, place)
                if share > 0 and apart > 0 and (best is None or score > best[0]):
                    best = (score, counted[0])
        if not best:
            return raw[-1] if raw else ""
        spell = forms.get(best[1]) or {best[1]: 1}
        return max(spell, key=lambda w: spell[w])

    def reveal_search(self, number: int, rest: int = 0) -> str:
        """After an answer revealed a requirement on an item: one search the code builds from the requirements'
        value words, the revealed ones first, and the item's noun (races 8-14: the leaders first saw the paid listing near
        the top of such a search; the light model's own query after a reveal sometimes kept a dropped value or left the new
        one out), cut to the words the store is known to read; then the unread rows carrying most of those words are read.
        On an order of several products each item with a revealed requirement gets one (`rest`: the items still to search
        after it, 3 steps kept for each), with the noun its own rows carry, fewer pages read and the result in short lines."""
        keep = 3 * rest
        if self.left() < 16 + keep or self.time_left() < ASK_BASE + 40:
            return ""
        item = self.items[number - 1]
        many = len(self.items) > 1
        # a name's last word can be one every listing carries, or none: a bundle's item takes the word its own rows carry.
        # Under a cheapest pick the rest of a search can be 4 pages: a bundle's item reads 2, its other items need steps too
        got = self.value_search(number, 15 + keep, self.item_noun(item) if many else None, 2 if many else 4)
        if not got:
            return ""
        text, found, shown, shorter, (pages, hints, paged) = got
        rows = [i for i in self.unread_rows() if i in found]
        cat_key = self.category_key()
        if many and cat_key:
            # a bundle: the rows of this item's own kind (another item's rows may carry the same value words)
            rows = [i for i in rows if self.owns(str((self.rows.get(i) or {}).get(cat_key) or ""), item)]
        rows = rows[:2]
        read = self.fill_facts(rows, 12 + keep)
        self.trace.append({"kind": "reveal_search", "query": text, "rows": len(found), "read": rows[: len(read)]} | ({"item": number} if many else {}))
        if many:
            # one item of several: short lines (about 1,500 characters), not the whole result, so the summary holds every
            # item's search; the store's word rules, said once, are kept whole
            search = self.roles.get("search") or ""
            listed = self.value_rank(list(dict.fromkeys(i for obs in pages for i, _ in self.records_in(obs))))
            plan, last = self.paging(search, pages[0]), self.paging(search, pages[-1])
            count = f" of {plan[0]} matches" if plan and plan[0] > len(listed) else ""
            if count and last and last[3] is not None:
                count += f", next page {last[2]}={_dump(last[3])}"
            # each read row's attributes, those naming one of the item's required properties first
            props = [_stem(need.split(":")[0]) for n, need in self.needs_now() if n == number]
            facts = []
            for ident in rows[: len(read)]:
                block = _fact_block(self.full.get(ident)) or self.full.get(ident) or {}
                block = dict(sorted(block.items(), key=lambda kv: not any(_names(p, _stem(str(kv[0]))) for p in props)))
                facts.append(f"{ident}: " + _dump(_clip(block, 60))[:260])
            out = (f"\n[Kyo searched \"{text}\" for item {number}'s new requirement (one step a search"
                   + (f", {len(pages) - 1} more page(s)" if len(pages) > 1 else "") + f"): {len(listed)} rows{count}, those carrying most of the requirements' words first]\n"
                   + "\n".join(self.row_line(i) for i in listed[:5]) + self.cheapest_here(pages)
                   + ("\n[Kyo read (one step each)] " + " | ".join(facts) if facts else ""))
            if len(out) > 1500:
                out = out[:1500] + "...(cut)"
            return out + hints + ("" if len(pages) > 1 else paged)
        out = f"\n[Kyo searched \"{text}\" for the new requirement" + (" after a longer query matched nothing" if shorter else "") + " (one step a search)]\n" + shown
        if read:
            out += "\n[Kyo read the rows carrying most of the requirements' words (one step each)]\n" + "\n".join(read)
        return out

    def value_search(self, number: int, keep: int, noun: str | None = None, most: int = 4) -> tuple[str, set[str], str, bool, tuple[list[Any], str, str]] | None:
        """One search for an item built from its requirements' value words that the store's own texts use, the newest
        revealed value first, and the item's noun (the name's last word unless `noun` is given), with one shorter try when
        nothing matches; `keep` steps stay, `most` further pages at most are read. Returns the query, the listing ids its
        pages showed, the text shown, whether the shorter try was used, and its parts (the pages' results, the store's
        word hints, the further pages' text)."""
        search = self.roles.get("search")
        field = self.query_field(search) if search else None
        extras = self.search_extras(search, field) if search and field else None
        if not search or not field or extras is None:
            return None
        item = self.items[number - 1]
        known = self.store_words()
        raw = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", item.name)
        noun = noun or (raw[-1] if raw else "")
        told = [t for t in reversed(item.told) if t not in item.dropped]
        values: list[str] = []
        newest = 0
        figures: set[str] = set()
        for index, need in enumerate(told + [m for m in item.musts if m not in item.dropped]):
            # a requirement whose value is read off another listing names that listing, not a value to search for
            value = "" if (number, need) in self.refs else need.split(":", 1)[1] if ":" in need else need
            after_figure = False
            for word in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", value):
                counted = _counted(word)
                # a figure and the word right after it (its unit) say a measure, which a keyword search does not match
                # as a limit means it
                if re.fullmatch(r"\d[\d.,/]*", word) or after_figure:
                    figures.add(word)
                after_figure = bool(re.fullmatch(r"\d[\d.,/]*", word))
                if counted and counted[0] in known and all(_counted(w) != counted for w in values + [noun]):
                    values.append(word)
            if index == 0 and told:
                newest = len(values)
        # races 12-14: a code search sent for a measured limit alone ("at least <figure> <unit>") replaced reads that
        # found the fit (another line of ours lost a single-product task that way): only figures kept, no search
        if not values or not noun or all(v in figures for v in values):
            return None
        # the newest revealed value is a plain word value the listings read so far never write: a search without it
        # would only repeat the stated values (Arena, a two-product order: "polyester" was on no page read yet, the
        # code's search went out without it and the model carted a blend), so the model searches for it itself
        if told and newest == 0 and (number, told[0]) not in self.refs and not re.search(r"\d", told[0].split(":", 1)[-1]):
            return None
        # races 10-14, real searches whose first page held a paid listing: a value word first .30-.48, the product kind
        # then a value .09-.47 (r14 .30 vs .09); a bare figure first is weak (r12 .24, r14 .07), so then the noun leads
        lead = newest > 0 and re.fullmatch(r"\d[\d.,/]*", values[0]) is None
        plain = [v for v in values if v not in figures]
        if len(self.items) == 1 and len(plain) >= 2:
            # races 11, 13 and 14, every agent, within the same task: a search led by two value words (no product word
            # among the first two) listed a paid listing on its first page more often than a value word with the product
            # word (r14 .42 vs .32, r13 .49 vs .43, r11 .51 vs .47); one value word alone did worse, so the noun follows
            # the values and the shorter try is the newest value with the noun
            words = (plain + [noun] + [v for v in values if v in figures])[: max(2, self.word_cut or 4)]
            # a query no listing matches every word of gets one shorter try
            tries = [words, [plain[0], noun]]
        else:
            words = values[:newest] + [noun] + values[newest:] if lead else [noun] + values
            words = words[: max(2, self.word_cut or 4)]
            # a query no listing matches every word of gets one shorter try: the lead value and the noun
            tries = [words] + ([words[: (newest if lead else 1) + 1]] if len(words) > 2 else [])
        seen: Any = None
        for words in tries:
            text = " ".join(words)
            faults: list[str] = []
            args = _shape({**extras, field: text}, self.schemas.get(search), "args", faults)
            if faults or (search + _dump(args)) in self.read_cache or self.word_guard(search, args) or self.left() < keep:
                return None
            reply = self.act(search, args)
            if reply is None or self.finished:
                return None
            seen = reply.get("observation")
            if self.records_in(seen):
                break
            self.observe(search, seen)
            self.note_end(search, args, seen)
            self.learn_words(search, args, seen)
            self.read_cache[search + _dump(args)] = (self.version, self.step, self.show(seen))
        self.observe(search, seen)
        self.read_shape(search, seen)
        self.note_end(search, args, seen)
        hints = self.learn_words(search, args, seen)
        paged = self.more_pages(search, args, seen, most)
        shown = self.show(seen)
        self.read_cache[search + _dump(args)] = (self.version, self.step, shown)
        self.recent = (self.recent + [search + _dump(args)])[-3:]
        pages = [seen] + self.paged_obs
        found = {i for obs in pages for i, _ in self.records_in(obs)}
        return text, found, shown + hints + paged + self.cheapest_here(pages), len(tries) > 1 and words is tries[-1], (pages, hints, paged)

    def open_rows(self, rows: list[str], keep: int = 5) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by
        one; `keep` steps stay for the cart and the finish (fewer rows are opened rather than none)."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            # more rows when steps and time are plentiful: a decline is wrong whenever a fitting row was left unread
            room = 4 if self.left() >= 15 and self.time_left() > ASK_BASE + 30 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
        else:
            return False
        plan = plan[: max(0, self.left() - keep)]
        if not plan:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
            if name == joiner and isinstance(args.get(many), list) and len(args[many]) > 1:
                if any(str(i) in _dump(reply.get("observation")) for i in args[many]):
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                else:
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

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

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

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
            # an item still without a cart line may need one listing read before it is carted (Arena builds 05-12: 12 of
            # 143 episodes had two or more turns without a call while the attributes view was withheld, 4 of them lost)
            if len(self.cart_lines) < len(self.items):
                keep.add(self.facts_tool())
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
                # a removal without a count takes off one unit (the store's default), not the whole line
                have = have - (count if count else 1)
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def sync_cart(self, observation: Any, name: str | None = None, args: dict[str, Any] | None = None) -> str:
        """Read the cart from a cart tool's own reply when it lists the cart's lines (listing records, each with one
        small count): "synced" when that list was read and is now Kyo's record, "unreadable" when such a list is there
        but its counts cannot be told apart from other small numbers, "none" when the reply carries no such list."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(node, list):
                if all(isinstance(r, dict) for r in node):
                    lists.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)

        if not isinstance(observation, dict) or self.ident(observation):
            return "none"
        walk(observation, 0)
        touched = [v for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v in self.rows]
        with_ids = [rows for rows in lists if rows and any(self.ident(r, True) for r in rows)]
        # the list that carries the listing just changed, else any list of listing records; an empty list is the
        # emptied cart only when it is the reply's one list
        rows = next((r for r in with_ids if any(self.ident(x, True) in touched for x in r)), with_ids[0] if with_ids else None)
        if rows is None:
            if len(lists) == 1 and not lists[0]:
                rows = lists[0]
            else:
                return "none"
        idents = [self.ident(r, True) for r in rows]
        if not all(idents):
            return "unreadable"
        key = self.count_key
        if not key and rows:
            keys = [k for k in rows[0] if k != self.price_key
                    and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) and 0 <= r[k] <= 99 for r in rows)]
            if len(keys) > 1 and name and args is not None:
                # several small numbers on each line: the one that matches Kyo's own arithmetic for the changed line
                saved = dict(self.cart_lines)
                self.note_cart(name, args)
                expected = dict(self.cart_lines)
                self.cart_lines = saved
                keys = [k for k in keys if all(r.get(k) == expected.get(i, 0) for r, i in zip(rows, idents) if i in touched)]
            if len(keys) != 1:
                return "unreadable"
            key = keys[0]
            self.count_key = key
        out: dict[str, int] = {}
        for row, ident in zip(rows, idents):
            count = row.get(key) if key else None
            if not isinstance(count, int) or isinstance(count, bool):
                return "unreadable"
            if count > 0:
                out[ident] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return "synced"

    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Record a cart call: from the reply's own line list when it can be read; a store that lists the cart after
        every change it made changed nothing when a reply has no list of listings at all (a refusal); otherwise
        (no list, or one that cannot be read) Kyo's own arithmetic."""
        self.cart_touched = True
        self.cart_at = self.step
        state = self.sync_cart(observation, name, args)
        if state == "synced":
            return
        if state == "none" and self.cart_shows_lines:
            self.trace.append({"kind": "cart_unchanged", "args": _clip(args, 60)})
            return
        self.note_cart(name, args)

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
        """How many listings one side-by-side call may carry: once a call has failed, what the store has shown to work
        (a stated limit can be wrong), else the schema's limit, else what has worked."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        cap = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        if self.compare_bad:
            # step down from the failed size (halving at most), never below what has worked or below two
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if cap:
            return cap
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
        self.join_failed = False
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                if self.join_failed:
                    out.append((cid, "Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own."))
                else:
                    out.append((cid, "Shown in the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): too few steps are left for that; use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], "args", faults)
            if faults:
                out.append((cid, "Not run (no step used): " + "; ".join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, "Not run (no step used): send the finishing call on its own, after reading your other results."))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            blocked = "" if ending else (self.word_guard(name, args) or self.page_guard(name, args))
            if blocked:
                out.append((cid, blocked))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get("cart") and mark in self.recent:
                out.append((cid, "Not run (no step used): you just made this same call and nothing changed; its result is above."))
                continue
            readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and cached[0] == self.version:
                out.append((cid, f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.roles.get("compare") and self.shown_already(name, str(args.get(target))):
                out.append((cid, f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, "Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                                + "\n".join(f"{i}: {self.record_of(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and self.end_change and not self.end_held and self.left() >= 3 and self.time_left() > LAST_CALL + 10:
                # the store answered an ending with a change (a new budget, a dropped requirement, a price): a listing may
                # fit now, so the next decline waits once for a search and a checkout under the change
                self.end_held = True
                self.trace.append({"kind": "end_change_held", "change": self.end_change[:200]})
                out.append((cid, "Not run yet (no step used). The store turned down the last ending and changed the task: " + self.end_change
                            + ". Such a change usually lets a listing fit. Search each item again under it (with the new budget, without a dropped requirement), put the best fitting listing in the cart at the requested quantity and check out. End without a purchase only if that search shows that nothing fits; then send the ending call again."))
                continue
            if self.is_buy(name, args) and any(self.sold_out(i) for i in self.cart_ids()) and self.sold_blocks < 2:
                # a checkout holding a line the store called sold out earns nothing at any time left: Kyo swaps the line
                # itself when the sold-out review named a replacement, else the call is held back
                if not self.swap_sold():
                    self.sold_blocks += 1
                    gone = [i for i in self.cart_ids() if self.sold_out(i)]
                    out.append((cid, f"Not run (no step used): {', '.join(gone)} in the cart is sold out, so this order would fail. Replace that line with a fitting listing in stock, then check out." + "\nCart now: " + self.cart_text()))
                    continue
            if self.is_buy(name, args) and self.loss_carts and self.time_left() > LAST_CALL + 5:
                self.twin_back()
            if ending and self.left() > 2 and self.time_left() > ASK_BASE - 25:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    split = self.split_lines()
                    if split and self.split_held and not self.items_missing() and self.merge_split():
                        split = ""
                    elif split and self.split_held:
                        split = ""
                    if split:
                        self.split_held = True
                    elif not self.cheap_held:
                        split = self.cheaper_passed()
                        self.cheap_held = bool(split)
                    if split:
                        out.append((cid, "Not run yet (no step used). " + split + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                    # a line holding the value's words split apart is wrong whatever the spend: held once per cart
                    split = self.split_hold()
                    if split:
                        out.append((cid, "Not run yet. " + chain + seen + split + "\nCart now: " + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                elif self.time_left() > CLOSING + 15:
                    # a held ending needs one more model turn to act on the reason; without that time the decline goes through
                    if self.left() > 5:
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held and self.time_left() > CLOSING + 15:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.time_left() > CLOSING + 15:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and not self.audited:
                # a decline past the review's gates (few steps or little time left) gets the code's own review
                done = self.decline_gate()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                if self.finished:
                    out.append((cid, "Not run: the episode is over."))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and self.time_left() > LAST_CALL + 12:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    # a size limit only when every id is a listing the store has shown (an unknown id fails a call too)
                    if all(str(i) in self.rows for i in args[many]):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if name == self.roles.get("search") and not ending:
                # the result's shape (rows, match count, next page) is named once per tool
                self.read_shape(name, observation)
                self.note_end(name, args, observation)
                paged = self.learn_words(name, args, observation) + paged + self.chase_links(observation)
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open") and not self.facts_from[name]:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                self.book_cart(name, args, observation)
                self.cart_said = _dump(observation)
                self.recent = []
                # lines taken off by a zero or negative count (unless the reply still lists them), and the fullest
                # cart built so far, which is offered back when a late decline would throw it away
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.rows:
                        if counts[0] <= 0 and not _in_rows(observation, value):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
                    self.rearm()
            elif self.is_refuse(name, args) and not reply.get("finished"):
                self.rearm()
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
                    if ending and not reply.get("finished"):
                        self.end_change = meaning
                        self.end_held = False
                        if self.is_buy(name, args) and self.price_moved and not self.note_sold:
                            text += ("\n[Kyo] A changed price alone does not end the task: when the cart's total still fits the budget and each line"
                                     " still meets its requirements, check out again with the cart unchanged; keep the line in the cart.")
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
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
        # an order missing a requested product earns nothing (Arena builds 05-13: 7 of 127 orders, all lost, one where a
        # decline was the answer): such a cart goes to the decline review, which may complete it, and else to the ending
        partial = holding and bool(refuse) and bool(self.items_missing())
        if partial:
            self.trace.append({"kind": "partial_cart", "missing": self.items_missing(), "cart": self.cart_ids()})
            holding = False
        if holding and buy and any(self.sold_out(i) for i in self.cart_ids()) and not self.swap_sold():
            # an order holding a line the store called sold out earns nothing: without a replacement it is not placed
            self.trace.append({"kind": "sold_line_kept", "cart": self.cart_ids()})
            holding = False
            partial = True
        if buy and not partial and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
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
                    self.note_sold = False
                    change = self.observe(buy, reply.get("observation"))
                    if self.note_sold or any(line in self.spoken(reply.get("observation")) for line in self.sold_lines):
                        # a forced checkout refused for a sold-out line marks that line, as a refused checkout of the loop
                        # does, so the next finish does not order it again
                        self.find_sold()
                    if change:
                        self.end_change, self.end_held = change, False
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse and not holding:
            # the loop ran out of time or turns with no cart: the code's own review before ending without a purchase
            done = self.decline_gate()
            if self.finished or self.left() <= 0:
                return ""
            if done and again:
                return "the checkout of a listing the review found was refused: " + done
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    change = self.observe(refuse, reply.get("observation"))
                    if change:
                        self.end_change, self.end_held = change, False
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
                self.chat[index]["content"] = content[:300] + "...(older result shortened; listings in it:\n" + "\n".join(self.row_line(i) for i in ids) + ")"

    def row_line(self, ident: str) -> str:
        """A listing in one short line as its newest record shows it: id, its wordiest short text, price, stock."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        parts = [ident, (self.short_texts(ident) or [""])[0][:70]]
        price = self.price_of(ident)
        if price is not None:
            parts.append(f"price {price:g}")
        if self.stock_key and self.stock_key in record:
            parts.append(f"{self.stock_key} {str(record[self.stock_key])[:20]}")
        return " | ".join(p for p in parts if p)

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
        self.set_caps()
        self.work(self.loop_cap, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            # a change that came with the refusal (a new budget, a dropped requirement) may let a listing fit: that round
            # runs on a shorter floor, since without it the next ending throws the change away
            floor = RECOVER_FLOOR if self.end_change else CLOSING
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.time_left() > floor:
                todo = ("\nThe change may let a listing fit now: search each item again under it, cart the best fitting listing at the requested quantity"
                        " and check out. End without a purchase only if that search shows nothing fits." if self.end_change and not self.cart_ids() else "")
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish." + todo + "\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01, LAST_CALL + 8 if self.end_change else CLOSING)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def loop_floor(self, floor: float) -> float:
        """The tool loop runs until CLOSING (Arena: a quarter of build 08-11's tasks stopped at the old 70 s floor with time
        unused); with nothing carted and no decline review yet, it stops at the old floor so the code's own review of the
        coming decline still has its time."""
        if floor == CLOSING and not self.cart_lines and not self.audited and not self.gate_done and self.roles.get("refuse"):
            return ASK_BASE
        return floor

    def work(self, loop_cap: int, spend_cap: float, floor: float = CLOSING) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.time_left() > self.loop_floor(floor)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if self.late_check_due():
                # the checkout's own cart check needs ASK_BASE-25 s left; a loop that runs out of time first ends in a
                # forced checkout nobody checked (Arena 13854: 12 such orders, none passed; checked orders .64). So with
                # the end near and every item carted, Kyo checks the cart itself while the agent still has turns to act
                verdict = self.check_cart(None, SPEND_HARD)
                self.trace.append({"kind": "late_check", "ok": not verdict})
                if verdict:
                    self.chat.append({"role": "user", "content": "[Kyo] Time is short, so Kyo checked the cart now (no step used). " + verdict
                                      + "\nFix only what this names (the review is advice: re-read a record when it seems wrong), then check out within your next turns.\n" + self.status()})
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or self.cost >= SPEND_STOP:
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
                names = ", ".join(t["function"]["name"] for t in self.offered())
                self.chat.append({"role": "user", "content": f"Call one of the tools to continue. The tools open to you now: {names}; a step needing any other tool is no longer possible, so cart the best fitting listing seen for each item and check out.\n" + self.status()})
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


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
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
