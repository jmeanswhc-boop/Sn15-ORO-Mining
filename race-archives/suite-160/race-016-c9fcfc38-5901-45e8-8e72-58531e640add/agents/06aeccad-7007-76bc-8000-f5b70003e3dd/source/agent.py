"""newTerry: a shopping agent that buys for one shopper in an online store, acting only through the store's tool calls.
It reads the shopper's request and the store's tools (name, description, parameter schema) and has a model write a brief:
the items with quantity and firm requirements, the budget, the pick rule (any, cheapest, most or least), the message and
step limits, the steps along the way the request asks for, and which tool does each job (search, open, offer, compare,
ask, cart, cart_view, buy, refuse). Argument names, id keys, price and stock keys and paging are learned from the schemas
and from the results, never assumed. A main model then chooses store calls under the GUIDE prompt; newTerry checks every
call against its schema before a step is spent, and keeps a record of each listing (search rows and opened pages).
Once enough listings are open, an interviewer asks the shopper about hidden needs on the properties those listings
carry and on hints in the request, and turns the answers into requirements. newTerry reads further result pages for a
cheapest pick, follows requirements that point at another listing page by page, keeps its own record of the cart, and
reads store notes (budget, requirement, price and stock changes) into the ledger. Before checkout it re-reads offers,
looks at the cart, replaces sold-out lines and has a second model check the basket against the requirements and the
pick rule; before a no-purchase ending it audits the listings seen for a fit. Steps the request asks for (an approval,
a message) are done by a gate just before the order or the ending. Everything runs inside the request's step cap,
LOOP_CALLS and ALL_CALLS model calls, the SPEND_SOFT and SPEND_HARD cost caps and a wall clock (CLOCK_DEEP, CLOCK_WRAP,
CLOCK_KILL); when the loop stops, finish checks out a held cart or ends without a purchase.
Entry point: agent_main(problem_data), which returns the trace of the episode."""
from __future__ import annotations

import json
import random
import re
import signal
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
BUNDLE = 8
SPEND_SOFT = 0.055
SPEND_HARD = 0.070
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
CREDIT_WAIT = 150
CREDIT_MISSES = 3
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150_000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
DEEP_PAGES = 3
DEEP_MORE = 1
DEEP_RESERVE = 6
CLOCK_DEEP = 245.0
CLOCK_WRAP = 265.0
CLOCK_KILL = 285.0
CALL_MIN = 8.0
LLM_TIMEOUT = 50

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are newTerry, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once newTerry knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by newTerry's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. newTerry carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through newTerry's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that holds the required value as a whole phrase, its words together and in the same order, counts ("deep teal" meets teal, "extra roomy fit" meets roomy), but the same words split apart by another word do not ("14k rose gold" does not meet 14k gold). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. newTerry adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- Words that only say how, where or by whom the product will be used (to be worn on the right hand, a gift for a nephew) are not a property when no listing you open records them in any form: do not search for them and never end without a purchase over them. When a record gives the same property twice under different key spellings with different values, either value meets the requirement.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and newTerry reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour, pattern or finish (whichever a listed key carries and the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type or style, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, leave out keys seen on only one or two.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value.
- An answer that turns away everything except one value ("anything with the colour apart from V gets binned", "don't show me anything where the colour comes in other than V", "nothing but V") asks for that value: the need is "colour: V" with V copied word for word, never "not V". When V is itself a comparison or points at another listing ("other than 2 litres", "within 1 inch of the one on the page for ..."), the need is that comparison, copied whole."""


PLAN_MANY = PLAN.replace(", and each question is answered only about the first property it names, so every question costs one property.",
                         ". One message names several properties of one product; the shopper answers about one of them, mostly the first, so put the likeliest first and keep a long, well ranked list behind it.")
HEAR_MANY = HEAR.replace('- "free": the property of each question the shopper said they do not mind about.',
                         '- "free": the property of each question the shopper said they do not mind about. When a question names several properties and its answer names none of them, leave "free" out for it.')


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value that holds the required value as a whole phrase, its words together and in the same order, counts ("deep teal" meets teal, "extra roomy fit" meets roomy), but the same words split apart by another word do not ("14k rose gold" does not meet 14k gold), units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement that only says how, where or by whom the product will be used (worn on the right hand, a gift for a nephew) and that no listing record shown states in any form counts as met, since the store does not record it; when a record gives the same property twice under different key spellings with different values, either value meets it. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses (other than the use requirements above)."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value that holds the required value as a whole phrase, its words together and in the same order, counts ("deep teal" meets teal, "extra roomy fit" meets roomy), but the same words split apart by another word do not ("14k rose gold" does not meet 14k gold), limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement that only says how, where or by whom the product will be used (worn on the right hand, a gift for a nephew) and that no listing record shown states in any form counts as met, since the store does not record it; when a record gives the same property twice under different key spellings with different values, either value meets it; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "values": [its shop and each property value named for it, word for word, without the property names]}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown. A record is a link only when it states every value named for that listing, under the property's name or in its text; one that lacks any of them is a different listing (e.g. asked for the lamp sold by Shop Q that has colour red and base marble, values ["Shop Q", "red", "marble"]: a Shop Q lamp in red that never mentions marble is not it). A page that sends the reader to another listing for this property does not end the chain, even when it also shows a value of its own: follow it."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


def _dump(value: Any) -> str:
    """Serialise a value as compact JSON (no spaces after separators, non-ASCII characters kept).
    Falls back to str(value) when the value cannot be encoded."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


class _Late(BaseException):
    """Raised by the SIGALRM handler when a bounded proxy call overruns its time limit.
    It derives from BaseException, so except Exception blocks inside the client do not swallow it."""
    pass


def _ring(signum: int, frame: Any) -> None:
    """SIGALRM handler installed by _bounded; it only raises _Late to break out of a stalled post."""
    raise _Late()


def _bounded(client: Any, path: str, body: dict[str, Any], limit: float) -> Any:
    """POST body to path through a proxy client with a hard cap of limit seconds (never under 1).
    Arms a real interval timer that raises _Late; returns the client's verbose result, or None when the timer fired.
    Where no SIGALRM handler can be set (not the main thread, no such signal) it posts without a cap.
    The timer is cleared and the previous handler restored in every case."""
    try:
        old = signal.signal(signal.SIGALRM, _ring)
    except (ValueError, OSError, AttributeError):
        return client.post_verbose(path, body)
    result = None
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL, max(1.0, limit))
            result = client.post_verbose(path, body)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except _Late:
        result = None
    finally:
        signal.signal(signal.SIGALRM, old)
    return result


_LINK_SKIP = frozenset("a an and any as at by for from go goes has have in is it its of on or page pages point points product products listing listings same sold that the their them this those to wherever whatever what which with you your one ones other another not but".split())


def _link_words(text: str) -> set[str]:
    """Split text into a set of lowercase words for matching the values named along a pointer chain.
    Punctuation becomes spaces and a digit glued to letters is split (12inch gives 12 and inch); _words splits camel case.
    Plurals are trimmed: words ending in sses, or in hes or xes past 4 letters, lose es; other words over 3 letters lose a final s unless it is ss."""
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", re.sub(r"[^\w\s]", " ", str(text)))
    out = set()
    for word in _words(text).split():
        if word.endswith("sses") or (len(word) > 4 and word.endswith("es") and word[-3] in "hx"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _key_words(value: Any, depth: int = 0) -> set[str]:
    """Gather the link words of every dict key inside a nested record (to depth 4, first 50 entries of a list).
    Scalar values add nothing, so the set holds only property names; trail passes it to _states_all."""
    out: set[str] = set()
    if depth > 4:
        return out
    if isinstance(value, dict):
        for key, inner in value.items():
            out |= _link_words(key) | _key_words(inner, depth + 1)
    elif isinstance(value, list):
        for inner in value[:50]:
            out |= _key_words(inner, depth + 1)
    return out


def _states_all(values: Any, records: list[Any], names: set[str] | frozenset[str] = frozenset()) -> bool:
    """Check that the records state every value named for one link of a pointer chain.
    Values with a digit are skipped; for the others, words of two or more letters outside _LINK_SKIP must appear in the records.
    A value fails when all its words are missing, or when the missing words are not all property names (the names set)."""
    have = _link_words(" ".join(_dump(r) for r in records if r))
    for value in _items(values):
        value = str(value or "")
        if re.search(r"\d", value):
            continue
        want = {w for w in _link_words(value) if len(w) > 1 and w not in _LINK_SKIP}
        miss = want - have
        if miss and (miss == want or not miss <= names):
            return False
    return True


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    """Copy a nested value with every string longer than limit cut to limit characters plus an ellipsis.
    Dicts and lists keep their structure down to depth 7; anything deeper is replaced by three dots."""
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
    """Parse a model reply into one JSON object, or return None.
    Strips a Markdown code fence, tries the whole text, then the span from the first opening brace to the last closing one.
    A reply that parses to anything but a dict, or is not a string, gives None."""
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
    """Normalise a key or label to lowercase words: camel case is split and runs of spaces, underscores, dots and hyphens become one space.
    This is the shared form for comparing attribute keys, property names and category labels."""
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


_ONLY_EXCEPT = r"(?:apart from|other than|except for|except|besides|save for|bar)"
_ONLY_END = r"(?=\s+(?:gets?|goes(?! with)|go(?! with)|will|would)\b|\s+(?:is|are)\s+(?:out|binned|not|no)\b|\s*,\s*(?:because|since|as|so|cos)\b|\s*[.!?;]?\s*$)"
_ONLY_ANY = re.compile(r"\b(?:anything|everything|any)\b.*?\b" + _ONLY_EXCEPT + r"\s+(.+?)" + _ONLY_END, re.I | re.S)
_ONLY_NONE = re.compile(r"\bnothing(?: else)?\s+(?:but|apart from|other than|except for|except|besides|save for)\s+(.+?)" + _ONLY_END, re.I | re.S)
_TIMING = re.compile(r"deliver|arriv|dispatch|turnaround|lead time|deadline|\bship\w*\s*(?:date|day|time|by|before|within|speed)", re.I)
_ONLY_BUT = re.compile(r"\b(?:anything|everything)(?: else)?\s+but\s+(.+?)" + _ONLY_END, re.I | re.S)
_ONLY_REFUSE = re.compile(r"\b(?:bin\w*|don'?t|do not|won'?t|will not|never|reject\w*|refus\w*|no room|no good|no use|nothing|not|toss\w*|chuck\w*|throw\w*|thrown|no thanks?|no thank you|nope|no-go|no deal|deal-?breakers?|turn\w* (?:it |them |that )?down|say no|rule\w* (?:it |them )?out|out of the question|off the table|veto\w*|elsewhere|a (?:firm |flat |polite |hard |straight )?no)\b", re.I)
_ONLY_EASY = re.compile(r"\b(?:mind|care|fine|okay|ok|happy|great|works?|welcome|open to|suits?|fuss\w*|bother\w*|picky|either way|no (?:problem|bother|worries|matter|preference))\b|\b(?:but|except|besides|apart from|other than)\s+(?:anything|everything|any)\b", re.I)


def _only_values(text: str) -> list[str]:
    """Find the values an answer asks for by turning away everything else, one sentence at a time.
    Matches the nothing but V form (_ONLY_NONE) or the anything apart from V form (_ONLY_ANY), V running up to an ending such as gets binned, a because clause or the sentence end.
    An _ONLY_ANY match counts only when the rest of the sentence has a refusal word (_ONLY_REFUSE) and no easygoing word (_ONLY_EASY).
    Returns the distinct values of at most 70 words, with quotes and end punctuation trimmed."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text or "")):
        hit = _ONLY_NONE.search(sentence)
        if not hit:
            for form in (_ONLY_BUT, _ONLY_ANY):
                hit = form.search(sentence)
                rest = sentence[:hit.start(1)] + " " + sentence[hit.end(1):] if hit else ""
                if hit and (not _ONLY_REFUSE.search(rest) or _ONLY_EASY.search(rest)):
                    hit = None
                if hit:
                    break
        if hit:
            value = hit.group(1).strip(" ,.;!?\"'")
            if value and len(value.split()) <= 70 and value not in out:
                out.append(value)
    return out


def _overlap(a: str, b: str) -> bool:
    """True when the word sets of a and b are both non empty and one holds the other.
    Used to match a planned question label against an attribute key or an ask first entry."""
    x, y = set(a.split()), set(b.split())
    return bool(x and y and (x <= y or y <= x))


def _stem(text: str) -> frozenset[str]:
    """Words of a text as a frozenset after _words, with a final s dropped from words over 3 letters.
    Lets queries, labels and property names match whatever their plural form."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Decide whether an attribute key names a requirement's property; both arrive as stem sets.
    True when they are equal, when every property word is in the key, or when a key of two or more words sits inside the property."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    """The JSON Schema type names a schema allows: its type string, or the strings in its type list."""
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    """Test a value against one JSON Schema type name; booleans never pass as integer or number.
    A type name it does not know accepts any value."""
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
    """Coerce tool arguments toward their JSON schema and collect what still breaks it in faults, each tagged with its path (where).
    Safe conversions: an integral float or a numeric string to a number, a number to a string, true or false text to a boolean, a lone value to a one element list.
    An enum value is mapped to the allowed spelling when only its case differs; a string over maxLength is cut silently.
    Reported faults: a type it cannot convert, a value outside the enum, minLength, minimum and maximum, minItems and maxItems, an argument not allowed under additionalProperties false, and a missing required key.
    Recurses into array items and object properties and returns the coerced value; every caller skips the store call when faults is not empty."""
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
    """Treat a value as a list: a list as it is, a non empty string or dict as a one element list, anything else as empty."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _count(value: Any) -> int | None:
    """Read a whole number from an int, an integral float or a digit string (an optional .0 tail allowed).
    Booleans, negatives written as text and anything else give None."""
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
    """Decide whether a value could be a listing id: a string of 3 to 64 characters with no space once stripped.
    Unless named says its key is an id key, it must also contain a digit."""
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Pick the key that holds listing ids in a list of rows, or None.
    A candidate must be id like in every row and, with several rows, unique across them.
    Keys whose name contains id rank first, then the key with the longer mean value."""
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
    """Search a result, down to depth 5, for a dict inside a list whose string values include ident.
    After a cart call it tells whether a line is still shown in the cart."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    """Find a listing's attribute block: the nested dict of node with the most scalar entries (at least two strings or numbers).
    Returns those scalar entries, or None; a record that has one is treated as an opened page."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def _cut(text: str, limit: int) -> str:
    """Shorten text to at most limit characters at a word boundary; a single word longer than that is cut hard."""
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0]
    return head if head else text[:limit]


class Item:
    """One requested product in the ledger: its name, the store's category label (shelf) and the quantity.
    musts are the firm requirements from the request, told the ones the shopper or a store note added later,
    dropped the ones given up since, and free the properties the shopper said they do not mind about."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Keep the brief's name, shelf, quantity and musts for one item; told, dropped and free start empty."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class NewTerry:
    """One episode of newTerry in one store: the brief, the ledger of items, the listing records and the chat with the main model.
    It also tracks steps and messages used, its own record of the cart, store notes and the caches that save repeated reads.
    run plays the episode; the trace it returns records each decision newTerry took along the way."""
    def __init__(self, problem: dict[str, Any]):
        """Set up an episode from the problem's environment: the binding sent with every store call and the policy view.
        Takes the request text (query), keeps tools that carry a function name, and maps each tool to its parameter schema (an empty object schema when it has none).
        The step cap is max_steps, or 30 when that is missing or not positive; the wall clock starts here (t0).
        Everything else starts empty: listing rows and opened pages, cart lines, notes, interview state, the read and paging caches and the trace."""
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
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.credit_misses = 0
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
        self.spare: list[tuple[int, str]] = []
        self.many = False
        self.crowded = False
        self.led: set[tuple[int, str]] = set()
        self.rank: dict[tuple[int, str], int] = {}
        self.missed: dict[int, list[str]] = {}
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
        self.hits: list[list[str]] = []
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
        self.part_holds = 0
        self.part_seen: set[str] = set()
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
        self.chains: dict[frozenset[str], tuple[int, str | None, Any]] = {}
        self.widened = False
        self.first_deep = False
        self.t0 = time.monotonic()
        self.paged_obs: list[Any] = []

    def left(self) -> int:
        """Steps still available under the step cap: max_steps minus the steps used so far."""
        return self.max_steps - self.step

    def clock(self) -> float:
        """Seconds of wall clock time since the episode object was built."""
        return time.monotonic() - self.t0

    def msgs_free(self) -> int:
        """Messages the interviewer may still send: the limit minus those sent, minus the reserve still owed to gate steps."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """A stable text key for the recorded cart (sorted listing and units pairs), used to see whether the cart changed."""
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """The tools that show a listing's page: the offer tool (or the open tool when there is no offer tool) and the compare tool."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single listing tool that shows attributes, or None.
        Prefers a tool already seen returning an attribute block; else the open tool unless it proved to show none.
        The tool must take exactly one id argument (single_target)."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the compare tool was seen showing attributes: True, False, or None while untested."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """True when the compare tool should not be used for attributes: it showed none,
        or it is untested while the facts tool already showed attributes or an offer tool exists."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        """True when the request has steps for this stage (order or end) that the gate has not yet cleared for the current cart and requirement version."""
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps to keep for finishing: 2 when an order step needs a message, messages remain and the order gate is still due; otherwise 1."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def panel(self, limit: float = SPEND_SOFT) -> list[str]:
        """Models a judging call may use at the episode's cost so far: HEAVY then LIGHT while the cost is within limit and HEAVY_ROOM still fits under SPEND_HARD,
        LIGHT alone up to SPEND_HARD, and none past it."""
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Get one chat reply, trying the models in order; returns the message or None.
        Every model try counts toward ALL_CALLS, and nothing is sent once that cap is reached.
        Only transient statuses are retried: no status, 402, 408, 425, 429 and 5xx; no status or 408 gets one retry, the others up to len(WAITS).
        Each pause is the WAITS entry with up to 20 percent jitter either way; retrying stops when the waits would pass WAIT_TOTAL (CREDIT_WAIT for a 402) or the clock CLOCK_WRAP.
        Other failures move on to the next model. A 402 still failing after its retries returns None and counts one credit miss; CREDIT_MISSES in a row set no_credit, and any good reply clears the count."""
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    self.credit_misses = 0
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > (CREDIT_WAIT if status == 402 else WAIT_TOTAL) or self.clock() + WAITS[tries] > CLOCK_WRAP:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.credit_misses += 1
                self.no_credit = self.credit_misses >= CREDIT_MISSES
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Send one chat completion through the proxy at temperature 0 and return (message, status).
        For models under the anthropic/ prefix with more than one message, the first and last messages get an ephemeral cache_control mark.
        Tools, when given, go with tool_choice auto.
        With less than CALL_MIN seconds left before CLOCK_KILL the call is skipped as late; otherwise it is bounded by LLM_TIMEOUT or the time left, whichever is less.
        The reported usage cost is added to self.cost; a reply with neither content nor tool calls counts as a failure with status 0.
        An error status comes from the client error or from an error object in the body; each failure is logged to the trace."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        room = CLOCK_KILL - self.clock()
        if room < CALL_MIN:
            self.trace.append({"kind": "late", "model": model, "at": round(self.clock(), 1)})
            return None, "late"
        _llm.timeout = int(min(LLM_TIMEOUT, room))
        result = _bounded(_llm, "/inference/chat/completions", body, room)
        if result is None:
            self.trace.append({"kind": "late", "model": model, "at": round(self.clock(), 1), "cut": True})
            return None, "late"
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
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """Send a one message prompt to each model in turn until a reply parses as one JSON object (_json_obj).
        Gives up with None after an environment 402 (broke), a model 402 (no_credit) or once ALL_CALLS is reached."""
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Make one store tool call through the environment endpoint, with the binding and the next step number.
        On success it updates the step counter and the finished flag and returns the response data;
        the string arguments of a page tool call that returned no error are marked as opened listings.
        A 409 is resynced: an expected N step number is adopted, a complaint about a reused step skips ahead one,
        and a finished, finalized or quarantined episode marks finished.
        Statuses 400 to 404 and 422 give up at once (402 also sets broke); there are at most 4 attempts, and None means the call failed."""
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

    def read_request(self) -> None:
        """Turn the request and the tool list into the brief (BRIEF prompt) and set the episode up from it.
        The store's opening note is added when present; LIGHT answers first, and HEAVY is asked when the brief lacks items, roles or a buy role.
        Roles that name an unknown tool are dropped.
        When one tool both checks out and ends without a purchase and the fixed arguments do not tell the two apart, a short prompt asks for buy_args and refuse_args.
        Builds the items (quantity 1 when unstated; one generic item when the brief has none) and the pointer requirements (refs) from lookups, matched to a must by property name.
        Then runs unmix and sets the budget (positive numbers only) and the message limit (3 when unstated but an ask tool exists).
        The request's step limit lowers max_steps when it lies between 10 and the cap.
        Process steps keep their stage (choose, order or end; order by default) and message count;
        reserve is the largest message count of an order or end step, at most the message limit."""
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
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
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
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
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Separate words that only describe a pointed at listing from real requirements on the product.
        For each item that has pointer requirements and other musts, the UNMIX prompt (HEAVY first) names the musts that describe that other listing.
        Those are removed and appended to the first pointer requirement as (that listing: ...), and refs is updated to the joined text."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def ledger(self) -> str:
        """Write the requirements in force as bullet lines for the status block and the review prompts.
        Per item: name, store label, quantity, the live requirements (musts and told, minus dropped) and what the shopper does not mind.
        A requirement whose value was already read from the page it points at carries that value.
        Then the budget with unit and scope, the pick rule, in stock only, wishes, rules from the request and a quiet decline when asked.
        Each process step is listed with the moment newTerry does it (before choosing, at the order, at a no purchase ending).
        Once the attribute tool is known it is named, with the page tools that show no attributes and the tool that shows a listing's page."""
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
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
            when = {"choose": "before choosing", "order": "at the order (newTerry does it when you check out)", "end": "at an ending without a purchase (newTerry does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        """The status block sent after every turn: steps used and left, a warning, pending lookups, review candidates, recent changes and the ledger.
        The warning grows sharper as steps run out: FINAL STEPS within tail plus one, few steps at 6 or fewer (or items plus 5 before any checkout),
        running low at 12 before any checkout, and a push to cart when half the steps are gone with nothing carted.
        Pointer requirements still unread are listed (at most three) only until the first checkout call; only the last four notes are repeated."""
        left = self.left()
        text = f"[newTerry] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
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

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """Return the listing id of a record, or None; a dict needs at least 3 keys.
        The learned id key is tried first when its value is id like.
        Otherwise only list members, the outermost dict and dicts holding nested dicts are tried with _learn_id;
        the key found must contain id or its value must already be a known listing."""
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
        """Decide whether a list holds listing rows: at least half its entries are dicts with 3 or more keys.
        True when every row carries an id under the current id key, or when _learn_id finds one, which then becomes the id key."""
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
        """Fold every listing record in a result into self.rows by id, newest values winning, and note the order ids were first seen.
        A boolean field turning from True to False marks the listing unavailable; turning back to True clears that mark.
        A record with an attribute block (_fact_block) is copied to self.full as an opened page; later partial records update that copy.
        Walks dicts and lists down to depth 6 and stops descending once a record is identified."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
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
        """Every string enum value found in the tool schemas, lowercased.
        category_key uses them to spot the row field that holds the store's category labels."""
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
        """Guess which row key holds the category label, or None.
        Candidates hold strings of 1 to 40 characters without three digits in a row, appear on at least 80 percent of rows and take at most 40 distinct values.
        The winner has the largest share of values found among the schema enum labels, then fewer distinct values, more rows, longer values."""
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
        """Table of the attribute keys seen on opened listings, grouped by category label, for the interview plan.
        Each full record adds its attribute block keys (normalised by _words) with the listings that carry them and their values.
        When some group labels match an item's shelf or name (same words or a shared stem of 4 or more letters), only those groups are kept; they are saved as key_groups.
        Skips keys with long values (mean over 40) or mostly slash separated variant lists, and all distinct long values carried by under half the group.
        A key that only repeats a shorter kept key's values is skipped as its twin.
        Returns up to 14 keys per group as (label, key, listings carrying it, sample values)."""
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
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        self.key_groups = groups
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            spread: set[str] = set()
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    if slot["n"] * 2 < group["n"]:
                        continue
                    spread.add(name)
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                shown = [_cut(v, 30) for v in slot["values"][:2]] if name in spread else [v[:40] for v in slot["values"][:4]]
                table.append((label, name, slot["n"], shown))
        return table

    def item_groups(self) -> list[list[str]]:
        """Group the seen listing ids by category label and return the largest groups, no more than one per item.
        Groups whose label matches an item's shelf or name come first; when none match, all groups compete on size."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        """The one string argument through which a tool takes a listing id, or None.
        It is the required free text string (or the only free text string), and the tool may require nothing else."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """A listing's price from its row under the learned price key, as a float or None.
        A nested price dict gives its first number, and a digit string is converted."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """False when a listing is sold out or its stock field (the learned stock key) is False or a number at or below 0."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The (id, record) pairs a result shows: the result itself when it is one record,
        else the rows of its lists and of lists one level down; a bare list is read as rows."""
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
        """Remember which record keys a view tool returned, per tool and per listing.
        The per listing set starts over when the requirement version changes; shown_already compares the two."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """True when this view already showed this listing every key it gives since the last version change,
        and the listing has not been marked gone; the repeat view is then answered from the record."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learn the layout of a tool's list result once per tool with the SHAPE prompt.
        It names the rows key, the total count key, the next page key and the parameter that takes it, and a row's price and stock keys.
        Only results with a list of 2 or more dicts are sent (three rows, clipped); each name must exist in the result, the row or the schema.
        The count key must hold an integer; the first price and stock keys learned become price_key and stock_key.
        A failed reading is cached as None, so the tool is not asked about again."""
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
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
        """(total count, rows shown, next page parameter, next page value) for a paged result,
        or None when the shape lacks one of them or the values have the wrong types."""
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
        """True when a result shows at least 3 priced rows in ascending price order."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Read the rest of a short paged search for a cheapest, most or least pick, so the model sees the whole set.
        Also records the search in chains (note_chain) and resets paged_obs, which later feeds cheapest_here.
        Nothing happens on a later page, without a next value, when every row is shown, or for a cheapest pick already in price order.
        When it would need more than 2 pages, pass AUTO_PAGES in all, or leave fewer than 12 spare steps, it only tells the model the counts and the parameter to page with.
        Otherwise it reads each page itself (one step each), absorbs it, caches its text in read_cache and returns the pages as text."""
        self.paged_obs = []
        self.note_chain(name, args, observation)
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[newTerry] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
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
            self.note_chain(name, call, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[newTerry read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def note_chain(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """For a cheapest pick, remember how far a paged search for a query was read, keyed by the stem set of the query.
        A plain first call counts 1 and a call with any other set argument (such as a page token) adds one.
        The next page parameter and value are kept while the total exceeds the rows shown."""
        if self.brief.get("pick") != "cheapest":
            return
        target = self.single_target(name)
        plan = self.paging(name, observation)
        if not target or not plan or not isinstance(args.get(target), str):
            return
        bag = _stem(args[target])
        later = any(k != target and v not in (None, "", 0, 1) for k, v in args.items())
        done = (self.chains.get(bag, (0, None, None))[0] + 1) if later else 1
        self.chains[bag] = (done, plan[2], plan[3] if plan[0] > plan[1] else None)

    def plain_query(self, item: Any) -> str:
        """A short search query for an item: its name plus the words of simple must values, at most 6 words.
        A value is used only when it has 3 words or fewer and no comparison or pointer words; words already present are not repeated."""
        words = item.name.split()
        for must in item.musts:
            if must in item.dropped:
                continue
            value = must.partition(":")[2].strip()
            if not value or len(value.split()) > 3 or re.search(r"\b(?:at least|at most|under|over|within|between|than|not|no|same|page|whatever|unlike|only|any|either|or|up to|less|more)\b", value, re.I):
                continue
            words += [w for w in value.split() if not (_stem(w) & _stem(" ".join(words)))]
        return " ".join(words[:6])

    def deep_read(self) -> str:
        """For a cheapest pick, read result pages of a plain query per item with the search tool, so cheaper rows become known.
        Reads up to DEEP_PAGES pages, DEEP_MORE more once the task has widened, continuing from what chains says was already read.
        Needs a known price key and a search tool that takes one text argument; does nothing after CLOCK_DEEP seconds.
        Keeps DEEP_RESERVE plus 2 steps per item (and the tail) in hand.
        Stops at a missing next page, a schema fault or a failed call; returns the store note changes those pages showed, joined by semicolons."""
        if self.brief.get("pick") != "cheapest" or not self.price_key or self.finished:
            return ""
        tool = self.roles.get("search")
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        if self.clock() > CLOCK_DEEP:
            return ""
        reserve = DEEP_RESERVE + 2 * len(self.items)
        want = DEEP_PAGES + (DEEP_MORE if self.widened else 0)
        changes: list[str] = []
        for item in self.items:
            query = self.plain_query(item)
            bag = _stem(query)
            if not bag:
                continue
            done, param, value = self.chains.get(bag, (0, None, None))
            read = 0
            while done < want and self.left() - self.tail() - 1 >= reserve and not self.finished:
                if done and (value is None or not param):
                    break
                faults: list[str] = []
                call = _shape({target: query, **({param: value} if done else {})}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, call)
                if reply is None:
                    break
                seen = reply.get("observation")
                change = self.observe(tool, seen)
                if change:
                    changes.append(change)
                plan = self.paging(tool, seen)
                done += 1
                read += 1
                param, value = (plan[2], plan[3] if plan[0] > plan[1] else None) if plan else (param, None)
                self.chains[bag] = (done, param, value)
                if not plan:
                    break
            if read:
                self.trace.append({"kind": "deep", "q": query, "read": read, "pages": done})
        return "; ".join(changes)

    def cheapest_here(self, observations: list[Any]) -> str:
        """For a cheapest pick, a hint listing up to 6 buyable priced rows from these results, cheapest first; needs at least 2 such rows."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[newTerry] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def adds(self, name: str, args: dict[str, Any]) -> list[str]:
        """The listing ids a cart call would add.
        Empty when the tool is not the cart tool, a mode argument takes a non first enum value (a removal), or a count is 0 or less."""
        if name != self.roles.get("cart"):
            return []
        spec = self.schemas.get(name, {}).get("properties") or {}
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                return []
            if not isinstance(value, (str, list, dict)) and (_count(value) or 0) <= 0 and _count(value) is not None:
                return []
        out: list[str] = []
        for value in args.values():
            for part in value if isinstance(value, list) else [value]:
                found = [v for v in (part.values() if isinstance(part, dict) else [part]) if isinstance(v, str) and v in self.rows]
                out += [v for v in found if v not in out]
        return out

    def lower_rows(self, ids: list[str], fresh: set[str]) -> list[str]:
        """Among newly seen ids (fresh), buyable rows of the same category as a carted id and priced below it; up to 4, cheapest first."""
        cat_key = self.category_key()
        if not cat_key:
            return []
        picks: list[tuple[float, str]] = []
        for line in ids:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            picks += [(self.price_of(i), i) for i in fresh if i not in ids and self.buyable(i) and self.price_of(i) is not None
                      and self.price_of(i) < price and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind]
        return [i for _, i in sorted(set(picks))[:4]]

    def first_add(self, name: str, args: dict[str, Any]) -> str:
        """Hold the first cart add of a cheapest pick, once, to read more result pages first (deep_read).
        When those pages show a store note, the add waits so the change can be applied.
        When they show buyable rows of the same kind priced below the added listing (lower_rows), the add waits and those rows are shown.
        Returns the text sent instead of running the call, or an empty string to let the add go ahead."""
        ids = self.adds(name, args)
        if self.first_deep or self.cart_touched or not ids or self.brief.get("pick") != "cheapest":
            return ""
        self.first_deep = True
        before, at = set(self.rows), self.step
        change = self.deep_read()
        if change:
            self.queued = []
            return "Reading more result pages showed a change: " + change + "\nApply the change, then add the listing that fits."
        lower = self.lower_rows(ids, set(self.rows) - before)
        self.trace.append({"kind": "deep_add", "steps": self.step - at, "lower": lower})
        if not lower:
            return ""
        lines = ", ".join(f"{i} ({self.price_of(i):g})" for i in ids if self.price_of(i) is not None)
        rows = "\n".join(_dump(_clip(self.rows.get(i), 120))[:500] for i in lower)
        return (f"newTerry read {self.step - at} more result pages ({self.step - at} steps; {self.left()} left). The pick is the cheapest that fits,"
                f" and these buyable listings of the same kind, priced below {lines}, were on those pages; you have not seen them. Their result rows:\n"
                + rows + "\nSkip a row that already shows a value breaking a requirement. Open the cheapest one that may fit; add it if it fits"
                " every requirement, otherwise add your listing.")

    def wider_hold(self) -> str:
        """Hold the first checkout of a cheapest pick once, when at least 14 plus 2 steps per item remain; no step is used.
        Asks for a few more searches under other names for the product, looking only at buyable rows priced below the cart's lines."""
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
        """Open listings whose attributes have not been read, one step each, with the attribute tool.
        Skips ids never seen as rows, already opened or sold out; stops when keep or fewer steps remain, at a fault or failure,
        or once that tool proves to show no attributes.
        Records whether the tool returned an attribute block; returns one text line per opened listing, up to 1500 characters."""
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
        """For a cheapest pick, open the unopened buyable rows of each cart line's category priced below it, two per line,
        at most CHEAPER_OPENS in all and once per cart and version, keeping tail plus 4 steps; returns their page text."""
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
        """Absorb a result, note the ids a search showed in hits, and read any loose note lines in it with take_note.
        A note that changes the task is queued for the chat; its meaning is returned (empty when there is none)."""
        self.absorb(observation)
        if name and name == self.roles.get("search"):
            self.hits.append([i for i, _ in self.records_in(observation)])
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """Open several listings before the interview so the key table has attributes to show.
        With a compare tool whose only required argument is an array and that is not bare, one step opens up to maxItems (4 by default) unopened ids shared over the item groups.
        Otherwise, or when the view shows rows without attributes, it falls back to widen_singly.
        Skipped with 12 or fewer steps left or fewer than 2 picks; when nothing new opened from more than 2 ids it tries again with the first two."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
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
        """Open up to 5 listings one at a time for the interview, keeping 22 steps in hand.
        Each item group offers up to 2 unopened buyable ids (fewer once 8 are open), cheapest first for a cheapest pick; groups take turns.
        Within a group, rows sharing the most words with the stated requirements' values come first."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        lanes: list[list[str]] = []
        stated = {w for _, need in self.needs_now() for w in re.findall(r"[a-z0-9]{3,}", need.split(":", 1)[-1].lower())}
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            pool.sort(key=lambda i: -len(stated & set(re.findall(r"[a-z0-9]{3,}", _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        """True when the interview should run now: not asked yet, an ask tool, free messages and more than 8 steps left.
        It starts once 3 listings per item plus one are opened, or from step max(8, a quarter of max_steps)."""
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Arguments of the ask tool for one message: the text goes to the required (or first) free text string parameter.
        Other required parameters take their first enum value; None when one has no enum or the result breaks the schema."""
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
        """Ask the shopper about unwritten preferences, once, and report the answers to the main model.
        First opens more listings (widen) and builds the key table; the PLAN prompt (HEAVY first) gets the request, what each item already settles,
        the ASK FIRST entries and the seen keys. Each label is cut to its first property (split at and, or, slash, comma, semicolon or bracket),
        kept at 5 words or fewer, and not repeated for an item. check_plan may swap a key that few listings carry for a common one.
        Unless the episode allows exactly 3 messages (many is False) PLAN_MANY is not used: the plan asks for one more property than there are free messages, all
        free messages but one go out first, one property each, and the answers are read with the HEAR prompt; the last message repeats a
        question the shopper found crowded, now with two sample values, or asks the next planned property.
        With exactly 3 messages (many is True) PLAN_MANY asks for BUNDLE properties per free message, and each message names up to BUNDLE
        properties of one item (bundle), since a shopper answers one property per message; HEAR_MANY reads the answers. A complaint about
        crowding puts that message's properties back at the front and turns every later message into one property.
        Why exactly 3: with 2 messages a refused bundle costs half of them; with 4 or more, single questions already reach most of
        the ranked list, so the gain is small, while the shoppers seen with 4 or 5 messages refused any bundle and cost one message.
        New requirements bump the version; the summary lists each question and answer, the new requirements,
        pointer requirements to resolve, and the notes seen meanwhile.
        Properties left unasked, plus one well carried key per item, stay in spare for spare_ask to use just before the order."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
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
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        self.many = self.msg_limit == 3
        prompt = (PLAN_MANY.replace("COUNT", str(BUNDLE * count + 1)) if self.many else PLAN.replace("COUNT", str(count + 1))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        try:
            ordered = self.check_plan(ordered, table)
        except Exception:
            pass
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        if self.many:
            pending = list(ordered) + [pair for pair in self.backfill(ordered, table) if pair not in ordered]
            listed: list[tuple[int, str]] = []
            self.rank = {pair: place for place, pair in enumerate(pending)}
            for _ in range(count):
                left = [pair for pair in pending if self.open_pair(pair)] or sorted([pair for pair in listed if self.open_pair(pair)], key=lambda pair: sum(1 for got in sent if got[0] == pair[0]))
                if self.msgs_free() <= 0 or not left:
                    break
                number, pack = self.bundle(left, listed)
                if not pack:
                    break
                self.led.add((number, pack[0]))
                batch = self.send([(number, " / ".join(pack), self.ask_many(number, pack))])
                if not batch:
                    break
                sent += batch
                pending = [pair for pair in pending if not (pair[0] == number and pair[1] in pack)]
                listed += [(number, label) for label in pack if (number, label) not in listed]
                heard = self.hear(batch, (LIGHT, HEAVY))
                before = len(told)
                self.take_heard(heard, told)
                if len(told) > before or len(pack) == 1:
                    self.missed.pop(number, None)
                else:
                    self.missed[number] = list(pack)
                if len(pack) > 1 and _items(heard.get("crowded")):
                    self.crowded = True
                    self.missed.pop(number, None)
                    self.led.discard((number, pack[0]))
                    pending = [(number, label) for label in pack] + pending
                    self.trace.append({"kind": "crowded", "pack": _clip(pack, 80)})
            if not sent:
                return ""
            self.spare = [pair for pair in pending + [pair for pair in listed if pair not in pending] if self.open_pair(pair)]
        else:
            head = ordered[: count - 1] if count >= 2 else ordered[:count]
            self.spare = ordered[len(head):] + self.backfill(ordered, table)
            batch = self.send([(number, label, self.question(number, label)) for number, label in head])
            if not batch:
                return ""
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            if count >= 2 and self.msgs_free() > 0:
                crowded =[batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
                again = None
                for number, text, _ in crowded:
                    label = next((lab for n, lab in head if n == number and self.question(n, lab) == text), None)
                    samples = self.samples(number, label) if label else []
                    if label and len(samples) >= 2:
                        again = (number, label, self.question(number, label, samples))
                        break
                if again is None and len(ordered) > len(head):
                    number, label = ordered[len(head)]
                    again = (number, label, self.question(number, label))
                    self.spare = [pair for pair in self.spare if pair != (number, label)]
                if again:
                    batch = self.send([again])
                    if batch:
                        sent += batch
                        self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "newTerry's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def own_groups(self, number: int, groups: dict[str, Any]) -> list[str]:
        """The key table groups that belong to an item: labels equal to its shelf or name,
        else labels sharing a stem of 4 or more letters with the shelf, then with the name; a lone group is taken when nothing matches."""
        item = self.items[number - 1]
        exact = [label for label in groups if _words(label) in (_words(item.shelf or ""), _words(item.name))]
        if exact:
            return exact
        for text in (item.shelf or "", item.name):
            long = {w for w in _stem(text) if len(w) >= 4}
            mine = [label for label in groups if {w for w in _stem(label) if len(w) >= 4} & long]
            if mine:
                return mine
        return list(groups) if len(groups) == 1 else []

    def check_plan(self, ordered: list[tuple[int, str]], table: list[tuple[str, str, int, list[str]]]) -> list[tuple[int, str]]:
        """Review the planned questions against the key table before any is sent.
        A label naming a seen key that fewer than half the item's listings carry (with at least 4 seen) is swapped for spare_key's choice,
        unless the item's name or musts hint at it or it matches an ask first entry; each swap is logged."""
        groups = getattr(self, "key_groups", None) or {}
        first = [_words(str(x)) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        out: list[tuple[int, str]] = []
        for index, (number, label) in enumerate(ordered):
            want = _words(label)
            mine = self.own_groups(number, groups)
            seen = sum(groups[g]["n"] for g in mine)
            holders: set[str] = set()
            known = False
            for g in mine:
                for key, slot in groups[g]["keys"].items():
                    if _overlap(key, want):
                        known = True
                        holders |= set(slot["per"])
            carried = len(holders)
            item = self.items[number - 1]
            hinted = {w for w in _stem(label) if len(w) >= 4} & {w for w in _stem(" ".join([item.name] + list(item.musts))) if len(w) >= 4}
            if known and seen >= 4 and carried * 2 < seen and not hinted and not any(_overlap(want, f) for f in first):
                spare = self.spare_key(number, mine, table, out + ordered[index + 1:], seen)
                if spare:
                    self.trace.append({"kind": "plan_swap", "item": number, "from": label, "to": spare, "on": [carried, seen]})
                    out.append((number, spare))
                    continue
            out.append((number, label))
        return out

    def spare_key(self, number: int, mine: list[str], table: list[tuple[str, str, int, list[str]]], taken: list[tuple[int, str]], seen: int) -> str | None:
        """A replacement question key for an item: the most carried key of its groups, held by at least half the listings seen and with two or more sample values.
        Keys already planned for that item and keys whose words all appear in the item's name or musts are passed over."""
        item = self.items[number - 1]
        own_text = " ".join([item.name] + list(item.musts)).lower()
        for _, name, carried, values in sorted((row for row in table if row[0] in mine), key=lambda row: -row[2]):
            if carried * 2 < seen or len(set(values)) < 2 or any(n == number and _overlap(name, _words(lab)) for n, lab in taken):
                continue
            if all(re.search(r"\b" + re.escape(w), own_text) for w in name.split()):
                continue
            return name
        return None

    def backfill(self, taken: list[tuple[int, str]], table: list[tuple[str, str, int, list[str]]]) -> list[tuple[int, str]]:
        """Reserve questions beyond the plan: for each item, the best carried key of its groups that is not already taken, chosen by spare_key."""
        groups = getattr(self, "key_groups", None) or {}
        extra: list[tuple[int, str]] = []
        for number in range(1, len(self.items) + 1):
            mine = self.own_groups(number, groups)
            key = self.spare_key(number, mine, table, taken + extra, sum(groups[g]["n"] for g in mine)) if mine else None
            if key:
                extra.append((number, key))
        return extra

    def spare_ask(self) -> str:
        """Use messages the shopper still allows just before an order, since an order that misses an unheard need cannot pass.
        Sends the next spare property one message at a time, while more than 2 steps remain and the clock is before CLOCK_DEEP, until an answer adds a requirement.
        A new requirement bumps the version and returns text that holds the order for one more look at the cart; otherwise returns an empty string and the order goes on."""
        told: list[str] = []
        lines: list[str] = []
        while self.spare and not told and not self.finished and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() > 2 and self.clock() < CLOCK_DEEP:
            if self.many:
                number, pack = self.bundle(self.spare, [])
                if not pack:
                    break
                self.led.add((number, pack[0]))
                self.spare = [pair for pair in self.spare if not (pair[0] == number and pair[1] in pack)]
                batch = self.send([(number, " / ".join(pack), self.ask_many(number, pack))], 2)
            else:
                number, label = self.spare.pop(0)
                batch = self.send([(number, label, self.question(number, label))], 2)
            if not batch:
                break
            lines += [f"Q: {q}\nA: {a}" for _, q, a in batch]
            self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        if not told:
            return ""
        self.version += 1
        self.trace.append({"kind": "spare_ask", "told": _clip(told, 120)})
        return "The shopper answered one more question before the order:\n" + "\n".join(lines) + "\nNew firm requirements: " + "; ".join(told) + ". Check the cart against them and change it if it does not meet them, then check out."

    def bundle(self, pending: list[tuple[int, str]], listed: list[tuple[int, str]]) -> tuple[int, list[str]]:
        """Choose the next message's item and properties: the item of the best-ranked open property, its open properties in
        rank order. After a message on that item told nothing, the later half of that message goes first (the shopper answered
        its first property they have a view on, so repeating the order would hit the same one); one property after a complaint."""
        open_pairs = [pair for pair in pending if self.open_pair(pair)]
        if not open_pairs:
            return 0, []
        number = open_pairs[0][0]
        pool = [label for n, label in open_pairs if n == number]
        pool += [label for n, label in listed if n == number and label not in pool and self.open_pair((n, label))]
        pool.sort(key=lambda label: self.rank.get((number, label), len(self.rank)))
        prev = [label for label in self.missed.get(number, []) if label in pool]
        if prev and not self.crowded:
            cut = len(prev) // 2
            pool = prev[cut:] + [label for label in pool if label not in prev] + prev[:cut]
        return number, pool[: 1 if self.crowded else BUNDLE]

    def open_pair(self, pair: tuple[int, str]) -> bool:
        """True while a property has been neither led in a message nor answered."""
        return pair not in self.led and not self.answered(pair)

    def answered(self, pair: tuple[int, str]) -> bool:
        """True when the shopper has stated a need on this property of the item or said they do not mind about it."""
        number, label = pair
        want = _words(label)
        item = self.items[number - 1]
        return any(_overlap(_words(need.split(":")[0]), want) for need in item.told) or any(_overlap(_words(free), want) for free in item.free)

    def ask_many(self, number: int, pack: list[str]) -> str:
        """Word one message about several properties of an item; a single property keeps the one-property wording."""
        if len(pack) == 1:
            samples = self.samples(number, pack[0])
            return self.question(number, pack[0], samples if len(samples) >= 2 else None)
        return f"For the {self.items[number - 1].name}, what " + ", ".join(pack[:-1]) + f" or {pack[-1]} would you like?"

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        """Word one question about a property of an item, ending in two example values when samples are given."""
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
        """Up to two example values of the key named label from the key table, leaving out values over 30 characters or already in the item's musts."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        """Send questions to the shopper with the ask tool, one step each; returns (item, question, answer text) per message sent.
        Stops when the episode ends, floor or fewer steps remain, the message limit is used, the arguments cannot be built, a call fails or the reply is an error.
        An answer that turns away all but some values gets a plain reading appended (_only_values).
        Every message and its reply join the done log, with the step, which the gate later reads."""
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
            plain = _only_values(words)
            if plain:
                words += " [plain reading: this answer turns away everything except " + " / ".join(plain) + ", so the requirement is: " + " / ".join(plain) + "]"
                self.trace.append({"kind": "only", "values": _clip(plain, 120)})
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Apply parsed answers to the items, appending each change to told for the summary.
        Needs become told requirements (pointing ones also join refs); drops mark live requirements on that property as dropped and set widened.
        Free properties are noted on the item, and a positive budget replaces the old one."""
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
                self.widened = self.widened or bool(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        """Parse the questions and answers with the HEAR prompt into needs, drops, free properties, crowded questions and a budget; {} on failure."""
        prompt = (HEAR_MANY if self.many else HEAR) + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Carry out the steps the request asks for at this stage (order or end) just before the ending call; returns (go on, text).
        Up to GATE_ROUNDS times, the GATE prompt sees the steps, ledger, cart, the numbered record of messages and calls, notes, messages and steps left, and the non ending tools.
        It answers with a verdict per step and the next call: a message goes through send (counted in gate_msgs), any other tool runs with act.
        When an order step needs messages, only record lines made after the last cart change prove it; with no proof and no call, the model is nudged once.
        If a reply after the gate has acted brings new needs or a hold, the version is bumped and (False, text) asks the main model to fix the cart.
        With no call left the stage is marked done for this cart and version; an unknown or ending tool, a fault or a failure also ends the rounds."""
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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, newTerry did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(newTerry: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, newTerry did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        """The readable text of a reply: a string as it is, else every string of two or more words within depth 4, joined and cut at 800 characters.
        With no such string, the JSON of the reply up to 400 characters."""
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

    def loose_lines(self, observation: Any) -> list[str]:
        """Note like text in a result that is not itself a listing: outer strings of 5 or more words, string lists,
        and strings of 6 or more words inside a dict that is not a listing; lines already read are left out."""
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
        """Read store or shopper notes with the NOTE prompt (LIGHT, then SPARE) and apply them.
        Ids it says are gone become unavailable; note_sold is set when something is gone or sold out, and a one line sold out note joins sold_lines.
        Only a change (or an unlabeled reply) goes on: a new positive budget, drops (exact text, else by property name) and adds update the items.
        A budget change or a drop sets widened; gone ids and price moves are listed among the changes.
        A change bumps the version, clears the repeat call memory and joins notes; returns the summary with the changes in brackets."""
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            self.widened = True
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
            self.widened = True
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
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        """The live requirements as (item number, requirement): musts and told, minus those dropped."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """The [evidence] line after a view, for up to 12 listings in the result that carry an attribute block.
        For each live requirement (first 8), it lists the block's keys whose names match its property (_names on stems), 3 keys at most.
        Empty when the result has no such listing or no key matches."""
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

    def cart_text(self) -> str:
        """The cart for prompts: the last cart view when it is newer than the last cart change, else newTerry's record of the cart lines."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by newTerry's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        """The listing ids in the cart: known ids found in a fresh cart view, else the recorded cart lines."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """The newest record of a listing as clipped JSON, the opened page before the row, up to 3000 characters."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        """Lines for other seen listings shown to the cart check, newest first, skipping the given and sold out ids.
        For a cheapest pick, opened listings come first, then by price."""
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        """Cache key for a cart check: the sorted ids plus the requirement version."""
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        """Independent check of the cart, or of a planned set of ids, against the requirements with the CHECK prompt (HEAVY first).
        Runs once per id set and version, and only while panel still offers a model for limit (SPEND_SOFT unless given).
        The prompt shows the ledger, the cart or the planned ids with their quantities, each listing's record (marked SOLD OUT when known),
        and for a cheapest, most or least pick the other listings seen.
        The verdict is stored (a reply that does not parse counts as not ok); on ok false the ids leave leads,
        and the text names up to 5 problems and up to 4 better listings; otherwise it returns an empty string."""
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.panel(limit) or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + self.timing_rule() + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, self.panel(limit), 3000)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self) -> tuple[str, bool]:
        """Close pointer requirements before checkout: requirements whose value comes from another listing and is not read yet.
        Up to 4 rounds while cost stays under SPEND_HARD: the TRAIL prompt gets the pending requirements and up to 60 records
        (OPENED pages or ROW rows, newest first) and follows each chain link by link.
        A link naming an opened listing that does not state all its values (_states_all) is a different listing, and the model is told to find the right one.
        A finished chain whose links were all opened gives the value, kept in ref_values (version bumped).
        Unopened row links are opened with the offer or open tool, up to 3 per round and 6 in all, if the steps allow.
        Returns the opened pages and read values as text, and whether to hold the checkout: links still missing, at most twice, more than 5 steps left."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 4500) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            names = set().union(*(_key_words(r) for r in self.full.values()))
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                wrong = next(((i, str(l.get("about") or "").strip() or ", ".join(map(str, _items(l.get("values"))))) for l, i in zip(links, ids) if i and i in self.opened and not _states_all(l.get("values"), [self.full.get(i), self.rows.get(i)], names)), None)
                if wrong:
                    lost.append(f"{need} -> find {wrong[1]} ({wrong[0]} does not state all of that, so it is a different listing)")
                    self.trace.append({"kind": "trail_wrong", "id": wrong[0], "about": _clip(wrong[1], 120)})
                    continue
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.left() < len(reach) + self.tail() + 1 or self.finished:
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
            text += "[newTerry opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def follow_parts(self) -> str:
        """Before check-out: the request asks to look at any listing a page points to. Stores tag each listing's title with a
        part code in brackets; a cart item's page that names another such code points to that listing. A pointed listing
        already seen in a search row is opened here (one step each, while steps last); one that was never seen is handed
        to the model once, to search by its code and open. Each code is handled once; returns '' when nothing is pending."""
        if self.part_holds >= 2 or self.finished:
            return ""
        code = re.compile(r"[\(\[](?:[Pp]art\s+)?([A-Z]{2,4}\d{4,7})[\)\]]")
        tagged: dict[str, set[str]] = {}
        for ident in set(self.rows) | set(self.full):
            for found in code.findall(_dump(self.rows.get(ident)) + " " + _dump(self.full.get(ident))):
                tagged.setdefault(found, set()).add(ident)
        heads = {re.match(r"[A-Z]+", c).group(0) for c in tagged}
        if not heads:
            return ""
        known: list[str] = []
        unknown: list[tuple[str, str, str]] = []
        for ident in self.cart_ids():
            record = self.full.get(ident) or self.rows.get(ident)
            if not isinstance(record, dict):
                continue
            text = _dump(record)
            own = set(code.findall(text))
            for found in dict.fromkeys(re.findall(r"\b([A-Z]{2,4}\d{4,7})\b", text)):
                if found in own or found in self.part_seen or re.match(r"[A-Z]+", found).group(0) not in heads:
                    continue
                ids = tagged.get(found) or set()
                if ids & self.opened:
                    continue
                self.part_seen.add(found)
                if ids:
                    pick = sorted(ids)[0]
                    if pick not in known:
                        known.append(pick)
                else:
                    at = text.find(found)
                    unknown.append((ident, found, text[max(0, at - 40): at + len(found) + 20]))
        tool = self.offer_tool()
        target = self.single_target(tool)
        opened: list[str] = []
        for ident in known[:4]:
            if not tool or not target or self.left() < self.tail() + 4:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            self.observe(tool, reply.get("observation"))
            opened.append(ident)
        unknown = unknown[:3]
        self.trace.append({"kind": "parts", "opened": opened, "unknown": [c for _, c, _ in unknown]})
        if not unknown or self.left() < 2 * len(unknown) + self.tail() + 3:
            return ""
        self.part_holds += 1
        return ("The request asks you to look at the page of anything you buy and at any listing a page points to. "
                + " ".join(f"The page of {i} points to {c} (its words: ...{w}...)." for i, c, w in unknown)
                + " Search the store for " + ", ".join(c for _, c, _ in unknown) + " (a listing carries it in its title; it may sit in another category than the cart item, so search by that code alone),"
                + " open the page of the listing whose title carries it, and do not put it in the cart. Then check out.")

    def offer_tool(self) -> str | None:
        """The tool for a listing's current offer: the offer role, else the open role."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read the offer of each cart line not read since two steps ago or since the last refusal, one step each.
        Needs those lines plus 3 steps; reports lines whose plain fields (price, stock and the like) changed, bumping the version when any did."""
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
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """True when a listing is marked unavailable or gone."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """When a cart line has sold out, mark it gone at this step, drop it from leads, bump the version and suggest replacements.
        With 6 or fewer steps left it says to add the first replacement now; for a cheapest pick, to search below its price before swapping;
        otherwise to swap in the first one; with none found, to search for another fit."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[newTerry] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """After a checkout refusal that hints at a sold out line: with one cart line, mark it unavailable;
        with several, re-read their offers to find it. Nothing happens when a line is already known sold out."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        """Ask the SWAP prompt which other seen listings (up to 25 records, newest first) fit the sold out line's item; skipped above SPEND_SOFT.
        Keeps known ids that are not lost, gone or carted, ranks them by price for a cheapest pick and returns at most 4."""
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
        prompt = SWAP + self.timing_rule() + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.panel(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        """Before a no purchase ending, ask once (AUDIT prompt) whether a seen listing fits after all; skipped above SPEND_SOFT.
        When none is named, it opens the rows the audit suggests and the top search rows, then asks again.
        Named listings become lead_items and leads; returns advice to open and buy them, or an empty string."""
        if self.audited or self.cost > SPEND_SOFT:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            rows += [i for i in self.top_rows() if i not in rows]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        """Run the AUDIT prompt over up to 30 seen records that are not sold out (OPENED pages or ROW rows, newest first) with the ledger.
        The lighter model reads first, with the heavier one as its fallback, since most audits end in the same decline."""
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
        prompt = AUDIT + self.timing_rule() + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.panel()[::-1], 4000)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """The audit's fits per item: known ids that are not sold out, at most 4 per item; items left empty are dropped."""
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for k, v in fits.items():
            ids = [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4]
            key = self.item_key(k)
            named[key] = (named.get(key) or []) + [i for i in ids if i not in (named.get(key) or [])]
        return {k: v[:4] for k, v in named.items() if v}

    def item_key(self, key: Any) -> str:
        """The item number an audit answer used as a key: a plain number, the one number inside a label such as
        item 2, or item 1 when the order has a single item and the key holds no number (a model sometimes copies the
        template's words). Any other key is kept as it was."""
        number = _count(key)
        if number is None:
            digits = re.findall(r"\d+", str(key))
            number = int(digits[0]) if len(digits) == 1 else (1 if len(self.items) == 1 else None)
        return str(number) if number is not None and 1 <= number <= len(self.items) else str(key)

    def timing_rule(self) -> str:
        """An extra reading rule for the cart check, the audit and the sold-out swap, returned only when a requirement in
        force is about when the order must arrive, ship or be delivered. Store records rarely state delivery times, so
        such a requirement counts as met unless a record does state one; otherwise a story line such as a deadline in the
        request could make the checks refuse a cart that fits. With no such requirement the prompts stay as they were."""
        needs = [m for item in self.items for m in item.musts + item.told if m not in item.dropped]
        if not any(_TIMING.search(str(m).split(":")[0]) and "power" not in str(m).lower() for m in needs):
            return ""
        return ("\nExtra reading rule: a requirement on when the order must arrive, ship or be delivered (a date, a day or a deadline) counts as met "
                "unless a record shown states a delivery or shipping time for that listing, since this store does not record delivery times; "
                "never fail a listing or a cart on that requirement alone.")

    def top_rows(self) -> list[str]:
        """Up to 6 unopened buyable rows from the first 3 of each of the last 4 searches, newest first, within the budget when priced."""
        out: list[str] = []
        for hits in reversed(self.hits[-4:]):
            for ident in hits[:3]:
                price = self.price_of(ident)
                if ident in out or ident in self.full or not self.buyable(ident):
                    continue
                if price is not None and isinstance(self.budget, (int, float)) and price > self.budget:
                    continue
                out.append(ident)
        return out[:6]

    def open_rows(self, rows: list[str]) -> bool:
        """Open audit candidates: one compare call (up to compare_room ids) when compare shows attributes, else up to 3 single opens.
        Needs those calls plus 3 steps; True when any new page with attributes was read."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:3]]
        else:
            return False
        if self.left() < len(plan) + 3:
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
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Plan the cart tool calls that leave exactly lines (id to units) in the cart after taking out drop; None when the store cannot express it.
        A parameter holding an array of line objects takes the whole cart in one call.
        Otherwise one call per line: the id goes to an argument that carried ids before, the required string or the only string.
        Removal uses the mode enum's second value, a count of 0 when the tool sets counts, or a negative count when the minimum allows it.
        A line already carted at another count can be changed only by a tool that sets counts."""
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
        """Build and check out a confirmed cart without the main model, once.
        Runs only when the second refusal block confirmed a lead (confirmed) that is not sold out, and cart and buy tools exist.
        Takes one lead per item from lead_items (one per category), keeps carted lines of other kinds, drops the rest, and needs a line per item.
        Reads missing attributes, checks the planned cart (up to SPEND_HARD), sets the cart and checks out through the order gate;
        after a refusal it re-reads offers, checks again and tries once more.
        Returns a report of the calls and the store's replies, or an empty string when any condition fails."""
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
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
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
        return ("Not run (newTerry acted instead): the independent check confirmed " + self.confirmed + ", so newTerry set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        """True when a call is the checkout: the buy tool with every fixed buy argument the brief gives."""
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """True when a call ends without a purchase: the refuse tool with every fixed refuse argument the brief gives."""
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        """The tools the main model may call this turn: never the ask tool, and fewer as the end nears.
        Within tail steps only the ending tools; within tail plus 3 also cart and cart view; within items plus 5 before any checkout also the offer tool."""
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
        """Replace the free text of a no purchase ending with a short fixed apology, within maxLength.
        A text is replaced when it names a seen listing, when the request did not ask for alternatives, or when a required text is empty."""
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
        """Map a title passed in place of an id to the real listing id for the cart, open, offer and compare tools.
        Applies only when the tool has one free text or array argument, which for tools other than cart has carried ids before.
        A text of 8 or more characters that cannot be an id is matched as the start of a row's text fields:
        one hit is used, several hits or none give a message instead of spending a step."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
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
        """True when the cart tool sets a line's count rather than adding to it: no mode enum and an integer argument with minimum 0."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update newTerry's record of the cart after a cart call, and mark the cart changed at this step.
        A mode argument with a non first enum value means removal; an array of line objects (even an empty one) replaces the whole record.
        Otherwise each line's count is added, set (cart_sets) or taken off; lines at 0 or below leave the record."""
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
        """A result as chat text: compact JSON with strings clipped to SHOW_CHARS, cut at OBS_LIMIT characters."""
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        """The compare tool's array argument when it is the only required one, else None."""
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many ids one compare call may carry: the schema's maxItems if set.
        After a failure, one under the smallest size that failed, capped by the largest that worked (never under 2); otherwise the largest that worked, at least 4."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge two or more single opens in one turn into one compare call, when compare shows attributes.
        Only known ids not gone, at most compare_room; the first call becomes the compare call and the others are marked joined."""
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
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
        """Run the tool calls of one model turn; returns (call id, result text) pairs.
        Refused without a step: an unknown or withheld tool, an ask call (the interviewer owns messages), arguments that are not one JSON object,
        schema faults, ids it cannot resolve, opening a gone listing, removing a line not in the cart, an ending sent with other calls,
        a non ending call in the last tail steps, the same call twice in a row, and reads whose cached answer is still current.
        A compare call may be cut to compare_room ids and leaves out listings already fully shown; a cheapest pick's first add may wait for first_add.
        A checkout first passes deep_read, wider_hold, offer re-reads, a cart view, sold out swaps, pointer trails and check_cart.
        A no purchase ending first passes the audit, two lead blocks (the second may confirm a lead for complete_cart),
        a late hold while a built cart exists, one push to search more, and gets its text quieted.
        Endings with 2 or more steps left go through the gate before the call is made.
        After a call: compare sizes are learned, id arguments remembered, the result absorbed, views noted, searches paged and the attribute tool learned.
        Cart calls update the cart record and the built cart; a refused checkout records the step.
        The text adds paged rows, cheapest rows, evidence, a count 0 hint, read notes, sold out advice and queued notes; reads are cached by version."""
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: newTerry opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): newTerry's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[newTerry] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[newTerry] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            first = self.first_add(name, args) if not ending and self.left() > self.tail() + 2 else ""
            if first:
                out.append((cid, "Not run yet. " + first + "\nCart now: " + self.cart_text()))
                continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    change = self.deep_read()
                    if change:
                        self.queued = []
                        out.append((cid, "Not run yet. Reading more result pages showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                        continue
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
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    parts = self.follow_parts()
                    if parts:
                        out.append((cid, "Not run yet. " + parts + "\nCart now: " + self.cart_text()))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("newTerry read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). newTerry re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
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
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
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
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[newTerry] " + said + trimmed
                more = self.spare_ask() if self.is_buy(name, args) else ""
                if more:
                    out.append((cid, "Not run yet. " + more + "\nCart now: " + self.cart_text()))
                    continue
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[newTerry] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
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
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
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
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[newTerry] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[newTerry re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on newTerry's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode when the model loop stops; returns text only when the store refused the ending.
        Checks out when a cart is held, or was touched and not viewed since with 2 or more steps left; the order gate runs first.
        With again=True a refused checkout is counted and its reply returned so the loop can react;
        otherwise it goes on to end without a purchase with quieted text, after the end gate."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
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
        """Keep the chat under HISTORY_LIMIT characters by shortening the older half of the tool results.
        Each one over 600 characters keeps its first 400 and a list of up to 20 listing ids it named."""
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
        """Play the episode: read the request, absorb the store's opening view and start the chat with GUIDE, the request and the status block.
        Runs the model loop under LOOP_CALLS and SPEND_HARD; when it stops unfinished, finish(again=True) tries to end.
        A refused ending gets 10 more loop calls with a little more spend; a last finish ends anything still open.
        Returns the trace, closed by an end entry with steps, cost and model calls."""
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
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[newTerry] newTerry tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """The main model loop: each turn the model (LIGHT, then SPARE) picks calls among offered tools and run_calls answers them.
        Runs while unfinished, with steps left, under loop_cap, spend_cap and CLOCK_WRAP; the interview runs when ready_to_ask says so.
        A cart, buy or refuse call before the interview is held (more than 8 steps left) so the shopper is asked first.
        Arguments are encoded back as JSON; a reply without calls gets a reminder, and three such replies in a row end the loop.
        More than 2 failed model turns, no credit or ALL_CALLS also end it; the history is trimmed before each turn.
        The status block is appended to the last tool result of each turn."""
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and self.clock() < CLOCK_WRAP:
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
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
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


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point: build a NewTerry episode from the problem data, run it and return its trace.
    A crash while building returns a one entry crash trace; a crash while running is logged and finish is tried so the episode still ends."""
    try:
        episode = NewTerry(problem_data)
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
