"""Tradewind: built on the public KYO agent. Buys for one shopper inside an ORO store world.

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
WORDS_ARG = r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword|terms|words"
FILTER_ARG = r"shelf|label|group|categor|section|department|dept|aisle|catalog|maker|brand|vendor|seller|shop|store|kind|class|family|collection|tag"
PAGE_ARG = r"page|token|cursor|next|offset|start|from"
SPARE_SKIP = r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|part numbers?|warrant\w*|videos?|url|links?|images?|photos?)\b"
NOT_ASKED = r"\b(?:closest|nearest|substitut\w*|alternatives?|replacements?|instead|listings?)\b"
SPARE_LAST = r"^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
ROOM_SECONDS = 250          # after this the loop stops and the episode is finished with what the cart holds
LAST_SECONDS = 280          # after this a finish makes no further model calls
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
MENU_KEYS = 22              # keys per category the planner sees
LIST_KEYS = 16              # property names one message lists at most
SWEEP_PER_ITEM = 8          # unread listings the sweep reads for one product at most
AUTO_PAGES = 3              # further search pages Tradewind reads itself for a cheapest, most or least pick
TOLD_READS = 12             # rows of the searches on what the shopper said that are read first, over all products
CHEAPER_OPENS = 4           # cheaper rows of a cart line's kind whose attributes Tradewind reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Tradewind, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Tradewind knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Tradewind's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Tradewind carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Tradewind's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Tradewind adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts; a listing that states it nowhere does not meet it, and of several keys naming the property the one named exactly as the property decides. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- The ledger is the whole list of what matters. Never rule a listing out for something no requirement names (an age or size it suits, who it seems meant for, its brand, its quality), however it reads against the story.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Tradewind reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use built into its name) into musts: those belong to "name". "name" keeps the request's own words for the kind of product, whole when they name a kind or a department ("garden & patio lamps", "shoes or sandals" stay as they are); when the request says outright which category or department it wants, "name" is that, word for word, not a product the story mentions; and "name" leaves out what is said about it in a phrase of its own: that is a must, however it is worded. "I want a lamp, and the style of lamp is a torchiere" gives name "lamp" and must "style: torchiere"; "a rug that would work for a porch" and "rugs, and they're for a porch" give must "occasion: porch"; "a rug in the shaggy style" gives must "style: shaggy". A remark that the shopper will verify something marks a firm requirement. Words saying only that a listing must suit the request name no property. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps; nor is a mere invitation to ask the shopper things.
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
Name a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item's three most carried keys.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them. "N values among M fitting" counts the listings read so far that meet the request: a key with several values there still separates them, so prefer it over one with a single value.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, unwritten preferences nearly always sit on one item alone: when an item is marked HUB, put every pick on it; otherwise put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


PLAN_TURNS = "With several items, unwritten preferences nearly always sit on one item alone: when an item is marked HUB, put every pick on it; otherwise put every item's hinted properties first, then take turns between the items."
PLAN_TURNS_SEVERAL = "With several items, each item can hold an unwritten preference of its own, so give every item a pick: an item marked HUB goes first, every item's hinted properties come before listed keys, and the items take turns."


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, in the answer's own words and never the question's: the answer "my trim should be matte" to a question on the finish gives "trim: matte"; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. Never fail a listing for something no requirement names (an age or size it suits, who it seems meant for, its brand, its quality). The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit; nothing else rules a listing out, so never fail one for something no requirement names (an age or size it suits, who it seems meant for, its brand, its quality)."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

POINTED = """A shopper's requirement takes its value from another listing and describes that listing. Say how to find it. Return ONE JSON object and nothing else: {"property": the property whose value is wanted, "relation": "partner" when the wanted value is another value that belongs with what that listing gives rather than that value itself, "unlike" when it must differ from it, else "same", "shop": the name of the shop or seller it names, or "", "kind": the kind of product that listing is, in a word or two, "words": every value it gives to pick that listing out, each as short as the description has it, at most 6}.
REQUIREMENT: """

HOP = """A store page can send its reader to another listing for the value of one property. Read the page below for the property named. Return ONE JSON object and nothing else: {"points": true only when the page sends the reader to one other listing for this property's value (a remark that some other version of the item is around, without telling the reader to use it for this property, does not count, nor a pointer for a different property, nor a statement of which values belong together), "shop": that other listing's shop or seller, or "", "kind": its kind of product in the page's words, "words": every value the page gives to pick that listing out, each as short as the page has it, at most 6, "value": this page's own value for the property, from its attributes, title or remarks, or ""}.
"""

PARTNER = """A store page can state which values of a property belong together. Read the page below. Return ONE JSON object and nothing else: {"partner": the value the page says belongs with the value named, exactly as the page has it, or "" when the page does not say}.
"""

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


_BLANK = {"", "null", "none", "n/a", "na", "not specified", "unknown", "-"}
_MOST = re.compile(r"^(?:at most|no more than|not more than|up to|under|below|less than|maximum(?: of)?|max)\b\s*|\s*\b(?:or less|or fewer|at most|max|maximum)$")
_LEAST = re.compile(r"^(?:at least|no less than|not less than|over|above|more than|minimum(?: of)?|min)\b\s*|\s*\b(?:or more|at least|min|minimum)$")
_NOT = re.compile(r"^(?:not|no|never|anything but|anything other than|other than|except|without)\b\s*")


def _plain(text: Any) -> str:
    """Lower case, separators as spaces, a number parted from the unit glued to it."""
    text = re.sub(r"(\d)([a-z])", r"\1 \2", str(text).lower())
    return re.sub(r"[^\w.%]+", " ", text).strip()


def _read_need(need: str) -> dict[str, Any]:
    """A requirement "property: constraint" as its parts: the property's words, how it compares, the values."""
    prop, mark, rest = need.partition(":")
    if not mark:
        prop, rest = "", need
    text = rest.strip().lower().rstrip(".")
    relation = "is"
    for name, form in (("most", _MOST), ("least", _LEAST), ("not", _NOT)):
        if form.search(text):
            relation, text = name, form.sub("", text).strip()
            break
    values = [v.strip() for v in re.split(r"\s*(?:,|;|/|\bor\b)\s*", text) if v.strip()]
    return {"prop": _stem(prop), "relation": relation, "values": values or [text], "raw": need, "whole": _plain(rest.strip().rstrip("."))}


def _same_value(value: str, fact: Any) -> bool:
    wanted, stated = _stem(_plain(value)), _stem(_plain(fact))
    return bool(wanted) and bool(stated) and (wanted <= stated or stated <= wanted)


def _weigh_value(need: dict[str, Any], fact: Any) -> str:
    """yes, no or open for one requirement against one property's value. Measures in different units stay open. A value that
    reads word for word as the record does is met, whatever it starts with ("not available", "no warranty")."""
    relation = need["relation"]
    if need["whole"] and need["whole"] == _plain(fact):
        return "yes"
    if relation in ("most", "least"):
        limit, stated = _plain(need["values"][0]), _plain(fact)
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", limit)]
        found = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", stated)]
        if not numbers or not found:
            return "open"
        unit_a = {w for w in _stem(limit) if not re.fullmatch(r"[\d.]+", w)}
        unit_b = {w for w in _stem(stated) if not re.fullmatch(r"[\d.]+", w)}
        if unit_a and unit_b and not unit_a & unit_b:
            return "open"
        inside = [g <= numbers[0] if relation == "most" else g >= numbers[0] for g in found]
        return "yes" if all(inside) else "no" if not any(inside) else "open"
    matched = any(_same_value(v, fact) for v in need["values"])
    if relation == "not":
        return "no" if matched else "yes"
    return "yes" if matched else "no"


def _rival(need: dict[str, Any], prop: frozenset[str], facts: dict[str, Any]) -> bool:
    """A key on the same subject as the property (a shared long word) that does not name it and states another value."""
    subject = {w for w in prop if len(w) >= 4}
    for key, value in facts.items():
        words = _stem(key)
        if str(value).strip().lower() not in _BLANK and subject & words and not _names(prop, words) and not any(_same_value(v, value) for v in need["values"]):
            return True
    return False


def _series(need: dict[str, Any], text: str) -> str:
    """yes or mixed when a listing's text names a model series (letters, then digits, then x's) against a model code in the requirement,
    else an empty string. Only a ranking: a different series is no proof against a listing."""
    low = str(text).lower()
    found = re.findall(r"([a-z]{2,})[ -]?(\d+)x+\b", low)
    for value in need["values"]:
        for brand, code in re.findall(r"([a-z]{2,})[ -]?(\d{2,})\b", value.lower()):
            named = [digits for word, digits in found if word == brand]
            if named:
                return "yes" if any(code.startswith(digits) for digits in named) else "mixed"
    return ""


def _weigh_need(need: dict[str, Any], facts: dict[str, Any], text: str, own: frozenset[str]) -> str:
    """yes, mixed, said, no or open for one requirement against a listing. The product's own name in a key or a requirement does
    not count ("<product> style" is "style"). A key named exactly as the property decides; else the keys that contain its name,
    and when those disagree the listing is mixed. With no such key the value may stand under another key or in the listing's
    text. A listing that was read and states a need of the shopper's nowhere does not meet it."""
    prop = (need["prop"] - own) or need["prop"]
    # a blank-looking value still counts when it is the very value wanted ("na")
    live = {key: value for key, value in facts.items() if str(value).strip().lower() not in _BLANK or _plain(value) == need["whole"]}
    exact = [_weigh_value(need, value) for key, value in live.items() if prop and (_stem(key) == prop or (_stem(key) - own) == prop)]
    named = [_weigh_value(need, value) for key, value in live.items() if _names(prop, _stem(key))]
    if need["relation"] == "not" and "no" in named:
        return "no"
    # stated, while another key on the same subject says otherwise: such a listing ranks behind the clean ones without being ruled out
    rival = need["relation"] == "is" and _rival(need, prop, live)
    if exact:
        if "yes" in exact:
            # two spellings of one key can disagree on the same record; that is not a plain statement
            return "mixed" if rival or "no" in exact else "yes"
        return "open" if "open" in exact else "no"
    if named:
        if "yes" in named:
            return "mixed" if "no" in named or rival else "yes"
        return "open" if "open" in named else "no"
    if need["relation"] != "is":
        return "open"
    series = _series(need, text)
    if series:
        return series
    if any(_same_value(v, fact) for v in need["values"] for fact in live.values()):
        return "said"
    if any(_stem(_plain(v)) and _stem(_plain(v)) <= _stem(_plain(text)) for v in need["values"]):
        return "said"
    # silence rules a listing out only on a need the shopper stated: a requirement read from the request can carry the reader's own
    # wording or be met in ways the keys do not show ("Fits: every <series>" for one model), and that once cost right listings
    return "no" if live and need.get("heard") else "open"


_FILLER = {"a", "an", "and", "for", "in", "of", "or", "the", "to", "with"}


def _content(text: Any) -> set[str]:
    return {w for w in _stem(_plain(text)) if w not in _FILLER}


_SEE_PART = re.compile(r"\b[Ss]ee part\s+([A-Z]{1,6}-?\d{3,8})\b")
_MODEL = re.compile(r"\bModel(?:\s+(?:code|no\.?|number))?\s*:\s*([A-Z]{2,6}[- ]?\d[\dA-Za-z-]*)")
_MORE_PART = re.compile(r"\b(?:more on|details? (?:on|in)|refer to) part\s+([A-Z]{1,6}-?\d{3,8})\b", re.I)
_ONE_AT_A_TIME = re.compile(r"\bone\b.{0,24}\b(?:at a time|thing|matter|question|subject)\b|\btoo many\b|\bsplit\b|\bseparate(?:ly)?\b|\bsingle question\b", re.I)
_MODEL_REF = re.compile(r"\bModel\s+ref(?:erence)?\.?\s*:?\s*([A-Z]{2,6}[- ]?\d[\dA-Za-z-]*)")
_FITS = re.compile(r"\b(?:(?:Compatible with|Fits|Works with|For use with)\s*:|Goes with)\s*([^.]*)", re.I)
_CODE = re.compile(r"\b[A-Z]{2,6}[- ]?\d[\dxX]*\b")
_GOES = re.compile(r"([\w][\w ,'-]*?)\s+(?:goes|go)\s+(?:together\s+)?with\s+([\w][\w -]*?)(?=\s*(?:[,.;]|\band\b|$))", re.I)
_PAIRS = re.compile(r"\bpairs?\s+(.+?)(?:\.|$)", re.I)
_ELSEWHERE = re.compile(r"\belsewhere\b|\b(?:go by|refer to|look to|look at|taken? (?:it |its \w+ )?from|use the|check the|comes? from|instead)\b", re.I)


def _squash(code: Any) -> str:
    return re.sub(r"[\s-]", "", str(code)).upper()


def _compat_of(text: Any) -> dict[str, Any]:
    """What a listing's own words say: its model, the models and model families it fits, the ones it leaves out, the parts it refers to."""
    text = str(text or "")
    out: dict[str, Any] = {"model": "", "named": False, "fits": [], "except": [], "parts": _SEE_PART.findall(text)}
    named = _MODEL_REF.search(text)
    found = named or _MODEL.search(text)
    if found:
        out["model"] = _squash(found.group(1))
        out["named"] = bool(named)
    for clause in _FITS.findall(text):
        head, _, rest = clause.partition("except")
        out["fits"] += [_squash(c) for c in _CODE.findall(head)]
        out["except"] += [_squash(c) for c in _CODE.findall(rest)]
    return out


def _fits_model(compat: dict[str, Any], model: str) -> bool:
    """A stated code or code family covers this model, and no exception names it."""
    if model in compat["except"]:
        return False
    return any(fit == model or (fit.endswith("X") and model.startswith(fit.rstrip("X"))) for fit in compat["fits"])


def _pairings(text: Any) -> list[tuple[str, str]]:
    """The pairings a remark states, each as (one value, what it goes with)."""
    # three forms seen: "a goes with b, and c goes with d"; "our rule pairs a with b, c with d"; "its a goes together with b"
    text = str(text or "")
    pairs: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.;])\s+", text):
        listed = _PAIRS.search(sentence)
        if listed:
            for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", listed.group(1)):
                left, mark, right = part.partition(" with ")
                if mark and left.strip() and right.strip():
                    pairs.append((left.strip().lower(), right.strip().rstrip(".").lower()))
            continue
        for part in re.split(r",\s*(?:and\s+)?", sentence):
            hits = _GOES.findall(part)
            for left, right in hits:
                pairs.append((left.strip().lower(), right.strip().lower()))
            left, mark, right = part.partition(" with ")
            # the same list carried on without its verb: "banana goes with peach, lemon with salted caramel"
            if not hits and mark and pairs and left.strip() and right.strip():
                pairs.append((left.strip().lower(), right.strip().rstrip(".").lower()))
    return pairs


def _singular(word: str) -> str:
    """A plural noun's singular by the regular English endings."""
    if re.search(r"[^a]lves$", word):
        return word[:-3] + "f"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:ch|sh|x|ss)es$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _plain_names(label: Any, text: str = "") -> list[str]:
    """The separate product names a label joins ("x and y", "x, y & z"), singular unless the store's own text uses the plural more. A label naming one product gives none."""
    parts = [p.strip() for p in re.split(r"\s+(?:and|or)\s+|\s*[,&/]\s*", str(label or "").lower()) if p.strip()]
    if len(parts) < 2:
        return []
    names = []
    for part in parts:
        words = part.split()
        single = " ".join(words[:-1] + [_singular(words[-1])])
        plural_seen = len(re.findall(r"\b" + re.escape(part) + r"\b", text))
        name = part if plural_seen > len(re.findall(r"\b" + re.escape(single) + r"\b", text)) else single
        if name not in names:
            names.append(name)
    return names


class Board:
    """One requested product's standing in the store: which listings seen belong to it, and what each one's own record
    shows against every requirement in force. Built fresh from the agent's records whenever it is asked."""

    def __init__(self, agent: Any, number: int):
        self.agent, self.number = agent, number
        self.item = agent.items[number - 1]
        self.own = _stem(self.item.name) | _stem(self.item.shelf or "")
        self.needs = []
        for need in self.item.musts + self.item.told:
            if need in self.item.dropped:
                continue
            heard = need in self.item.told
            if any(n == number and text == need for n, text in agent.refs):
                value = agent.ref_values.get(need)
                if not value:
                    continue            # its value is still on another listing's page
                need = f"{need.split(':')[0]}: {value}"
            self.needs.append({**_read_need(need), "heard": heard})

    def labels(self) -> set[str]:
        key = self.agent.category_key()
        if not key:
            return set()
        seen = {_words(str(r.get(key) or "")) for r in self.agent.rows.values()} - {""}
        exact = {l for l in seen if l in (_words(self.item.shelf or ""), _words(self.item.name))}
        if exact:
            return exact
        long_words = {w for w in self.own if len(w) >= 4}
        return {l for l in seen if {w for w in _stem(l) if len(w) >= 4} & long_words}

    def lane(self) -> list[str]:
        """Listings of this product: those on its shelf; with one product and no shelf known, all of them."""
        key, labels = self.agent.category_key(), self.labels()
        rows = self.agent.rows
        if key and labels:
            return [i for i in self.agent.seen_order if _words(str((rows.get(i) or {}).get(key) or "")) in labels]
        return list(self.agent.seen_order) if len(self.agent.items) == 1 else []

    def weigh(self, ident: str) -> dict[str, list[str]]:
        record = self.agent.full.get(ident) or self.agent.rows.get(ident) or {}
        facts = _fact_block(record) or {}
        text = " ".join(str(v) for v in record.values() if isinstance(v, str))
        out: dict[str, list[str]] = {"yes": [], "mixed": [], "said": [], "no": [], "open": []}
        for need in self.needs:
            out[_weigh_need(need, facts, text, self.own)].append(need["raw"])
        return out

    def affordable(self, ident: str) -> bool:
        budget, price = self.agent.budget, self.agent.price_of(ident)
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or price is None:
            return True
        # nothing seen is within the limit: the limit and the prices are in different units, so they cannot be compared
        prices = [self.agent.price_of(i) for i in self.agent.seen_order]
        if not any(p is not None and p <= budget for p in prices):
            return True
        return price * max(1, self.item.qty) <= budget

    def fits(self) -> list[str]:
        """Read, buyable, affordable listings of this product with no requirement against them, best first."""
        found = []
        for ident in self.lane():
            if ident not in self.agent.full or not self.agent.buyable(ident) or not self.affordable(ident):
                continue
            standing = self.weigh(ident)
            if not standing["no"]:
                price = self.agent.price_of(ident)
                found.append((len(standing["open"]), len(standing["said"]) + len(standing["mixed"]), price if price is not None else float("inf"), ident))
        if self.agent.brief.get("pick") == "cheapest":
            found.sort(key=lambda f: (f[2], f[0], f[1]))
        else:
            found.sort(key=lambda f: (f[0], f[1], f[2]))
        return [f[3] for f in found]

    def plain_fit(self, ident: str) -> bool:
        """Every requirement in force is stated outright by the listing's own keys."""
        standing = self.weigh(ident)
        return bool(self.needs) and len(standing["yes"]) == len(self.needs)

    def wording(self, ident: str) -> int:
        """How many requirements' values a listing's row names."""
        text = _stem(_plain(_dump(self.agent.rows.get(ident))[:900]))
        return sum(1 for need in self.needs if any(_stem(_plain(v)) and _stem(_plain(v)) <= text for v in need["values"]))

    def unread(self) -> list[str]:
        """Listings of this product never read, buyable and affordable, those naming most of the requirements' words first
        (the cheapest first under a cheapest pick)."""
        wanted: frozenset[str] = frozenset()
        for need in self.needs:
            for value in need["values"]:
                wanted |= _stem(_plain(value))
        ranked = []
        for ident in self.lane():
            if ident in self.agent.full or ident in self.agent.opened or not self.agent.buyable(ident) or not self.affordable(ident):
                continue
            price = self.agent.price_of(ident)
            ranked.append((-len(wanted & _stem(_plain(_dump(self.agent.rows.get(ident))[:900]))), price if price is not None else float("inf"), ident))
        cheapest = self.agent.brief.get("pick") == "cheapest"
        ranked.sort(key=(lambda r: (r[1], r[0])) if cheapest else (lambda r: (r[0], r[1])))
        return [r[2] for r in ranked]


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Tradewind:
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
        self.started = time.monotonic()
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.kept_deal = False
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.swept = False
        self.searches: list[tuple[str, dict[str, Any], Any]] = []
        self.paged_marks: set[str] = set()
        self.cap_marks: set[str] = set()
        self.fit_rows: set[str] = set()
        self.auto_shelves: set[int] = set()
        self.shelf_held = False
        self.cheaper_held: set[str] = set()
        self.broken_held: set[str] = set()
        self.firmed: set[str] = set()
        self.multi_held = 0
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
        self.walked: set[str] = set()
        self.parts_done: set[str] = set()
        self.walk_memo: dict[str, str] = {}
        self.compat_held = 0
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
        self.plain_done = False
        self.paged_obs: list[Any] = []

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order take stay back."""
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
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
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
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit goes to the heavy one
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
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 2000) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m and not re.match(r"\s*(?:in[ _]stock|stock|availab\w*|on[ _]hand|price|budget|cost)\b", str(m), re.I)]
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
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 2000) or {}
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
            when = {"choose": "before choosing", "order": "at the order (Tradewind does it when you check out)", "end": "at an ending without a purchase (Tradewind does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        self.settle_shelves()
        left = self.left()
        text = f"[Tradewind] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 7 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        text += self.board_line() + self.lacking_line() + self.compat_line()
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
        wanted = [(_words(item.shelf or ""), _words(item.name), set() if item.shelf else {w for w in _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and not any(" " in v for v in values):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:MENU_KEYS]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), set() if item.shelf else {w for w in _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
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

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Tradewind reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") and not self.value_pending() else None
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
        pages = min(-(-(count - shown) // shown), AUTO_PAGES - self.pages_read, self.left() - self.tail() - 14)
        if pages < 1:
            return (f"\n[Tradewind] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {self.brief_rows(seen)}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Tradewind read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Tradewind read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        self.settle_shelves()
        cat_key, shelves = self.category_key(), {_words(item.shelf) for item in self.items if item.shelf}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None
                and (not shelves or not cat_key or _words(str(self.rows[i].get(cat_key) or "")) in shelves)]
        if len(rows) < 2:
            return ""
        return "\n[Tradewind] buyable rows here on the request's shelf, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:8])

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

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Tradewind's own call and read any note in it; returns what the note changed."""
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
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:5], 22)

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
        self.settle_shelves()
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        table = self.key_table()
        self.fit_rows = {i for number in range(1, len(self.items) + 1) for i in Board(self, number).fits()}
        hub = self.hub_item()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name}" + (" (HUB: the other items must be compatible with this one)" if number == hub else "") + " - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}{self.split_note(label, name)}): " + " | ".join(values))
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        plan_text = PLAN if len(self.items) < 2 else PLAN.replace(PLAN_TURNS, PLAN_TURNS_SEVERAL)
        prompt = plan_text.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip() and not re.search(NOT_ASKED, str(x), re.I)]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5 or re.search(NOT_ASKED, label, re.I):
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        # the planner's picks can run out before the messages do, and a message never sent is a need never heard
        ordered += self.spare_asks(table, ordered)
        if hub:
            ordered.sort(key=lambda pick: pick[0] != hub)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # one question at a time: a product with no stated need and the fewest questions so far goes next, since the next need is seldom on the
        # product that just stated one and a first question to a product draws more than a repeat; what the request itself asks to be asked keeps
        # its place. An answer that wants a single question gets that question once more, named with example values
        queue = list(ordered)
        stated: dict[int, int] = {}
        put_to: dict[int, int] = {}
        pinned = len(first)
        retried = False
        # each product's names in the plan's order; what the request asks to be asked is put as a plain question, so it is left out of them
        pools: dict[int, list[str]] = {}
        for number, label in ordered[len(first):]:
            pools.setdefault(number, []).append(label)
        lists_on = True
        listed: dict[int, int] = {}
        plan_rank = {pick: rank for rank, pick in reversed(list(enumerate(ordered)))}
        while queue and self.msgs_free() > 0 and not self.finished and self.left() > 6:
            # a property an answer has already stated is not asked again: the queue was fixed before the answers came
            queue = [pick for pick in queue if not self.said_already(*pick)]
            if not queue:
                break
            plain_one = pinned > 0
            if plain_one:
                pinned -= 1
            else:
                queue.sort(key=lambda pick: (stated.get(pick[0], 0) > 0, put_to.get(pick[0], 0), plan_rank.get(pick, 0)))
            number, label = queue.pop(0)
            put_to[number] = put_to.get(number, 0) + 1
            names = self.list_names(number, pools.get(number, []), listed.get(number, 0)) if lists_on and not plain_one else []
            as_list = len(names) >= 2
            if as_list:
                listed[number] = listed.get(number, 0) + 1
            batch = self.send([(number, label, self.list_question(number, names) if as_list else self.question(number, label))])
            if not batch:
                break
            sent += batch
            before = len(told)
            heard = self.hear(batch, (LIGHT, HEAVY))
            crowded = bool(_items(heard.get("crowded")))
            if as_list:
                # a not-picky answer to a list does not say which property it means; a muddled reply is not a request for one thing at a time unless it says so
                heard["free"] = []
                crowded = crowded and bool(_ONE_AT_A_TIME.search(batch[-1][2]))
                if crowded:
                    lists_on = False
                    self.trace.append({"kind": "lists_off", "reply": batch[-1][2][:120]})
            self.take_heard(heard, told)
            if len(told) > before:
                stated[number] = stated.get(number, 0) + 1
            if crowded and not retried and self.msgs_free() > 0 and self.left() > 6:
                retried = True
                samples = self.samples(number, label)
                if len(samples) >= 2:
                    again = self.send([(number, label, self.question(number, label, samples))])
                    if again:
                        sent += again
                        self.take_heard(self.hear(again, (LIGHT, HEAVY)), told)
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Tradewind's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = self.find_pointed()
        if pointed:
            summary += "\n" + pointed
        since = len(self.searches)
        found = self.search_told() if told else ""
        if found:
            summary += "\n[Tradewind searched on what the shopper just said, one step a page]\n" + found
        plain = self.plain_search()
        if plain:
            summary += "\n[Tradewind searched each product name its store label joins, one step a name]\n" + plain
        read = self.told_rows(since) if told else ""
        if read:
            summary += "\n[Tradewind read the rows of these searches first, one step each]\n" + read
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def spare_asks(self, table: list[tuple[str, str, int, list[str]]], ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Listed keys to ask after the planner's picks, each product's most carried first, the products in turn: none the request
        settles, none already picked or answered, no identifiers or logistics, keys naming several features at once last."""
        lanes: dict[int, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            item = self.items[number - 1]
            labels = Board(self, number).labels()
            if not labels and len(self.items) > 1:
                lanes[number] = []
                continue
            settled = [_stem(_plain(m.split(":")[0])) for m in item.musts]
            picked = {_words(label) for n, label in ordered if n == number}
            lane: list[str] = []
            for label, name, _, _ in table:
                if labels and _words(label) not in labels:
                    continue
                if not name or name in picked or name in lane or re.search(SPARE_SKIP, name):
                    continue
                own = _stem(name)
                if any(prop and (prop <= own or own <= prop) for prop in settled) or self.said_already(number, name):
                    continue
                lane.append(name)
            lanes[number] = [k for k in lane if not re.search(SPARE_LAST, k)] + [k for k in lane if re.search(SPARE_LAST, k)]
        spare: list[tuple[int, str]] = []
        while any(lanes.values()):
            for number, lane in lanes.items():
                if lane:
                    spare.append((number, lane.pop(0)))
        return spare

    def list_names(self, number: int, pool: list[str], turn: int) -> list[str]:
        """The property names for this product's next list. An unwanted name earlier in a list can take the reply, so each list should lead somewhere new:
        best guess first, then backwards, then from the middle with the names not listed yet in front."""
        live = [name for name in pool if not self.said_already(number, name)]
        head, rest = live[:LIST_KEYS], live[LIST_KEYS:]
        if turn == 0:
            return head
        if turn == 1:
            return head[::-1]
        middle = len(head) // 2
        shift = (middle + turn - 2) % max(1, len(head))
        return (rest[(turn - 2) * 2:] + head[shift:] + head[:shift])[:LIST_KEYS]

    def list_question(self, number: int, names: list[str]) -> str:
        item = self.items[number - 1].name
        schema = self.schemas.get(self.roles.get("ask") or "") or {}
        limits = [v["maxLength"] for v in (schema.get("properties") or {}).values() if isinstance(v, dict) and isinstance(v.get("maxLength"), int)]
        room = min(limits) if limits else 0

        def text(kept: list[str]) -> str:
            return f"What {', '.join(kept[:-1])} or {kept[-1]} would you like for the {item}?"
        kept = list(names)
        while room and len(kept) > 2 and len(text(kept)) > room:
            kept.pop()
        return text(kept)

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
        """Two values the listings show for this property: they say which property a question means."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

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

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (LIGHT, HEAVY)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        heard = self.ask_json(prompt, list(models), 1500) or {}
        if len(sent) == 1:
            for entry in _items(heard.get("needs")):
                if isinstance(entry, dict) and entry.get("points") is not True and str(entry.get("need") or "").strip():
                    entry["need"] = self.name_by_answer(str(entry["need"]).strip(), sent[0][2])
        return heard

    def gate(self, stage: str) -> tuple[bool, str]:
        if stage == "order":
            self.open_parts()
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
                return False, ("Before going on, Tradewind did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Tradewind: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, Tradewind did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
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
        return "lines by Tradewind's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        # the light model still reads the request: a requirement the first reading lost is invisible to the code's own check
        plain = not lead and self.all_plain(ids)
        if plain:
            self.trace.append({"kind": "check_light", "cart": ids})
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [LIGHT] if plain else [HEAVY, LIGHT], 1500)
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
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Tradewind opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
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
            text += "[Tradewind opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

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
        if not stale or self.left() < len(stale) + 5:
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
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        plain = self.cheapest_after(lost) or self.board_picks(lost)
        # the code has read what fits: an agent left to choose wandered, emptied its cart and ended with a plain fit known
        sure = next((i for i in plain if self.plain_anywhere(i)), None)
        if sure and len(lost) == 1 and self.left() >= 3 and self.swap_line(lost[0], sure, self.cart_lines.get(lost[0], 1)):
            return f"[Tradewind] Sold out now: {lost[0]}. It will not come back, so Tradewind took it out of the cart and put {sure} in its place: its own keys plainly state every requirement. Check out."
        picks = plain or self.replacements(lost)
        text = f"[Tradewind] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        if self.cost > SPEND_HARD:
            return []
        records = []
        elsewhere = self.off_lane(lost)
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident) or ident in elsewhere:
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or self.cost > SPEND_HARD:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            rows += [i for i in self.unread_rows() if i not in rows]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def settle_shelves(self) -> None:
        """An item with no shelf from the tools takes the category label seen on rows that covers all of its name's words and either
        adds none or matches at least two, when one label clearly does; settled again as new labels appear. One word shared with a
        longer label settles nothing: a wrong guess here makes the agent drop a right listing."""
        cat_key = self.category_key()
        if not cat_key:
            return
        seen = {str(r.get(cat_key)) for r in self.rows.values() if isinstance(r.get(cat_key), str) and r.get(cat_key).strip()}
        for number, item in enumerate(self.items, 1):
            if item.shelf and number not in self.auto_shelves:
                continue
            want = _content(item.name)
            scored = sorted(((-len(want & _content(l)), len(_content(l) - want), l) for l in seen), key=lambda s: s[:2])
            sure = bool(scored) and bool(want) and -scored[0][0] == len(want) and (not scored[0][1] or len(want) >= 2)
            if not sure or (len(scored) > 1 and scored[0][:2] == scored[1][:2]):
                if number in self.auto_shelves:
                    item.shelf = ""
                    self.auto_shelves.discard(number)
                continue
            item.shelf = scored[0][2]
            self.auto_shelves.add(number)

    def off_shelf(self) -> str:
        """Once, before a checkout with room to change it: a cart line on a neighbouring label, one that is no requested item's shelf
        but shares a word with a requested item's name. Another department altogether is the model's reading of the request to make."""
        cat_key = self.category_key()
        if self.shelf_held or not cat_key or self.left() < 4 or not all(item.shelf for item in self.items):
            return ""
        shelves = {_words(item.shelf) for item in self.items}
        names = set().union(*(_content(item.name) for item in self.items))
        stray = [i for i in self.cart_ids() if str((self.rows.get(i) or {}).get(cat_key) or "").strip() and _words(str(self.rows[i][cat_key])) not in shelves and _content(self.rows[i][cat_key]) & names]
        if not stray:
            return ""
        self.shelf_held = True
        return (f"{', '.join(stray)} is on the store's label {self.rows[stray[0]][cat_key]!r}, which is not this request's: only "
                + " and ".join(repr(item.shelf) for item in self.items) + " count. Replace it with a fitting listing from there.")

    def lacking_line(self) -> str:
        """With several products and most of the steps going: those with no cart line yet, and any with nothing seen inside the budget."""
        if len(self.items) < 2 or self.buys or self.left() > self.max_steps * 0.6:
            return ""
        cart = set(self.cart_ids())
        parts = []
        for number, item in enumerate(self.items, 1):
            board = Board(self, number)
            lane = board.lane()
            if cart & set(lane):
                continue
            priced = [i for i in lane if self.price_of(i) is not None]
            cheap = [i for i in priced if board.affordable(i)]
            parts.append(f"item {number} ({item.name})" + (", every row seen is over the budget and rows come in no price order, so search it under other words" if priced and not cheap else ""))
        return "\n[no cart line yet, turn to these now and leave the others as they are] " + "; ".join(parts) if parts else ""

    def board_line(self) -> str:
        """Per product, the listings read so far that nothing speaks against, best evidence first (cheapest first under a cheapest pick)."""
        parts = []
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            named = []
            for ident in board.fits()[:4] if board.needs else []:
                stand = board.weigh(ident)
                weak = len(board.needs) - len(stand["yes"])
                named.append(f"{ident} ({weak} of {len(board.needs)} not plainly stated)" if weak else f"{ident} (all stated)")
            if named:
                parts.append(f"item {number}: " + ", ".join(named))
            elif board.needs:
                # right rows were seen and left unread while other listings were opened until the steps ran out
                rows = [i for i in sorted(board.unread(), key=lambda r: -board.wording(r)) if board.wording(i)][:3]
                if rows:
                    parts.append(f"item {number}: nothing read fits yet, read these unread rows that name its requirements first: " + ", ".join(rows))
        return "\n[read listings with nothing against them, by the code's reading] " + "; ".join(parts) if parts else ""

    def name_by_answer(self, need: str, answer: str) -> str:
        """A need named by the property the answer names, when listings show that key: a reader can keep the question's word."""
        prop, mark, value = need.partition(":")
        said = _stem(_plain(answer))
        known = {_words(key) for record in self.full.values() for key in (_fact_block(record) or {})}
        hits = [key for key in known if key and _stem(_plain(key)) <= said]
        if not mark or not hits:
            return need
        best = max(hits, key=lambda key: (len(_stem(_plain(key))), len(key)))
        named = _stem(_plain(prop))
        if named <= said and not named < _stem(_plain(best)):
            return need
        return f"{best}:{value}"

    def brief_rows(self, observation: Any) -> str:
        """A result page's rows with their long texts cut: enough to choose by price and shelf, at a fraction of the tokens."""
        rows = [_clip(record, 70) for _, record in self.records_in(observation)]
        return _dump(rows) if rows else self.show(observation)

    def broken_line(self) -> str:
        """Once per cart: a cart line whose own keys speak against a need the shopper stated. Only those: on requirements read from the
        request the same hold made the agent drop right listings, because that reading can carry its own wording."""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.broken_held or self.left() < 4:
            return ""
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            for line in [i for i in self.cart_ids() if i in board.lane() and i in self.full]:
                stated = {need["raw"] for need in board.needs if need["heard"]}
                against = [need for need in board.weigh(line)["no"] if need in stated]
                if against:
                    self.broken_held.add(mark)
                    others = [i for i in board.fits() if i != line][:3]
                    return (f"The record of the cart's {line} does not state: {'; '.join(against)}. Replace it"
                            + (f" (listings read that fit: {', '.join(others)})" if others else "") + " unless a key of its record does say that in other words.")
        return ""

    def word_rows(self) -> set[str]:
        """Listings that a search on words has shown."""
        found: set[str] = set()
        for name, args, seen in self.searches:
            props = (self.schemas.get(name) or {}).get("properties") or {}
            if isinstance(seen, (dict, list)) and any(isinstance(v, str) and v.strip() and "enum" not in (props.get(k) or {}) and re.search(WORDS_ARG, k, re.I) for k, v in args.items()):
                found |= {i for i, _ in self.records_in(seen)}
        return found

    def cheaper_fit(self) -> str:
        """Once per cart under a cheapest pick: rows of the line's shelf priced below it are read, cheapest first, until one states the
        requirements at least as plainly as the line does. Reads that came after this choice decided nothing. Not while a requirement
        waits on another listing: its value is not known yet."""
        mark = self.cart_mark() + f"#{self.version}"
        live = {need for _, need in self.needs_now()}
        if self.brief.get("pick") != "cheapest" or mark in self.cheaper_held or self.left() < 4 or any(need in live and need not in self.ref_values for _, need in self.refs):
            return ""
        self.cheaper_held.add(mark)
        # before a first checkout: a refused line takes three steps and its replacement a read or two
        keep = self.tail() + (2 if self.buys else 6)
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            lines = [i for i in self.cart_ids() if i in board.lane() and self.price_of(i) is not None]
            for line in lines:
                price = self.price_of(line)
                weak = len(board.needs) - len(board.weigh(line)["yes"])
                reads = 0
                while True:
                    best = next((i for i in board.fits() if i not in lines and self.price_of(i) is not None and self.price_of(i) < price and len(board.needs) - len(board.weigh(i)["yes"]) <= weak), None)
                    if best or reads >= CHEAPER_OPENS or self.left() <= keep:
                        break
                    rows = [i for i in board.unread() if self.price_of(i) is not None and self.price_of(i) < price]
                    worded = self.word_rows()
                    rows.sort(key=lambda i: (i not in worded, self.price_of(i)))
                    if not rows or not self.fill_facts(rows[:1], keep):
                        break
                    reads += 1
                if best:
                    if self.swap_line(line, best, board.item.qty):
                        return f"Tradewind swapped the cart's {line} at {price:g} for {best} at {self.price_of(best):g}: it states the requirements at least as plainly and the pick is the lowest price. Check out."
                    return (f"{best} at {self.price_of(best):g} states the requirements at least as plainly as the cart's {line} at {price:g} and costs less, and the pick is the "
                            "lowest price. Swap to it unless its record clearly breaks a requirement.")
        return ""

    def firm_up(self) -> str:
        """Once per cart, before a checkout with steps to spare on an ordinary pick: a cart line whose own keys do not plainly state
        every requirement, while its shelf may hold one that does. Unread rows are read one at a time, a new search when none are left."""
        mark = self.cart_mark() + f"#{self.version}"
        # a first checkout keeps room for a refused line; after a refusal only the checkout itself is still to come
        if mark in self.firmed or self.brief.get("pick") in ("cheapest", "most", "least") or self.left() < self.tail() + (3 if self.buys else 11):
            return ""
        self.firmed.add(mark)
        # before a first checkout: the re-read and the cart view still come, then a refused line takes three steps and its replacement a read
        keep = self.tail() + (1 if self.buys else 8)
        most = 5 if self.buys else 3
        said = []
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            lines = [i for i in self.cart_ids() if i in board.lane()]
            if not board.needs or not lines or board.plain_fit(lines[0]):
                continue
            reads = 0
            while True:
                better = next((i for i in board.fits() if i not in lines and board.plain_fit(i)), None)
                if better or reads >= most or self.left() <= keep:
                    break
                rows = board.unread()
                if not rows and self.search_again(board, keep):
                    rows = board.unread()
                if not rows or not self.fill_facts(rows[:1], keep):
                    break
                reads += 1
            if better:
                # the code acts itself here: a model was seen waving this away
                if self.swap_line(lines[0], better, board.item.qty):
                    said.append(f"Tradewind swapped the cart's {lines[0]} for {better}, whose own keys plainly state every requirement of item {number}.")
                else:
                    said.append(f"{better}'s own keys plainly state every requirement of item {number}; the cart's {lines[0]} leaves some to other keys or to its text. Swap to it unless its record clearly breaks a requirement.")
        self.trace.append({"kind": "firm_up", "said": said})
        return " ".join(said)

    def swap_line(self, old: str, new: str, quantity: int) -> bool:
        """Put one listing in place of another in the cart, in code, when room is left for the checkout after it."""
        tool = self.roles.get("cart")
        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i != old}
        lines[new] = quantity
        calls = self.cart_calls(lines, [old]) if tool else None
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(tool), "args", faults)
            if faults:
                return False
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return False
            self.note_cart(tool, args)
            self.observe(tool, reply.get("observation"))
        self.trace.append({"kind": "swapped", "old": old, "new": new})
        return True

    def said_already(self, number: int, label: str) -> bool:
        """An answer has already stated a need on this property of this product."""
        asked = _stem(_plain(label))
        for need in self.items[number - 1].told:
            prop = _stem(_plain(need.split(":")[0]))
            short, long = (prop, asked) if len(prop) <= len(asked) else (asked, prop)
            # "fa pattern" is "pattern", but "mounting type" is not "type": a short general word does not make two properties one
            if prop == asked or (short and short < long and len(long - short) == 1 and all(len(word) >= 5 for word in short)):
                return True
        return False

    def value_pending(self) -> bool:
        """A requirement in force still waits on another listing's page: the board cannot weigh it, so its fits prove nothing. A forced
        order on such a board once overrode a right decision to end without a purchase."""
        live = {need for _, need in self.needs_now()}
        return any(need in live and need not in self.ref_values for _, need in self.refs)

    def cart_stands(self) -> bool:
        """Every product has a cart line, none sold out, and none whose own keys speak against a need the shopper stated."""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not cart or len(cart) < len(self.cart_ids()) or self.value_pending():
            return False
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            lines = [i for i in cart if i in board.lane()]
            if not lines:
                return False
            stated = {need["raw"] for need in board.needs if need["heard"]}
            if any(need in stated for i in lines if i in self.full for need in board.weigh(i)["no"]):
                return False
        return True

    def every_line(self) -> bool:
        """Every requested product has a cart line that has not sold out."""
        cart = {i for i in self.cart_ids() if not self.sold_out(i)}
        return bool(cart) and all(cart & set(Board(self, number).lane()) for number in range(1, len(self.items) + 1))

    def no_fit_shown(self) -> bool:
        """The sweep before this ending has run and every listing read for some product (three at least) fails by its own record, so a cart can only miss."""
        if not self.swept or len(self.items) < 2:
            return False
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            if sum(i in self.full for i in board.lane()) >= 3 and not board.fits():
                return True
        return False

    def several_note(self) -> str:
        """What each product of a several-product request has so far, for an agent about to end without a purchase."""
        parts = []
        for number, item in enumerate(self.items, 1):
            board = Board(self, number)
            fits, rows = board.fits()[:3], board.unread()[:3]
            parts.append(f"item {number} ({item.name}): " + (", ".join(fits) if fits else f"nothing read fits yet, unread rows: {', '.join(rows)}" if rows else "nothing seen yet, search for it with fewer words"))
        return "Ending without a purchase is almost never right on a request for several products, and a cart that misses costs no more than that ending. So far: " + "; ".join(parts) + ". Cart the best listing for each item and check out."

    def all_plain(self, ids: list[str]) -> bool:
        """Every product has cart lines whose own keys plainly state each of its requirements, and no requirement waits on another listing."""
        live = {need for _, need in self.needs_now()}
        if not ids or any(need in live and need not in self.ref_values for _, need in self.refs):
            return False
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            lines = [i for i in ids if i in board.lane()]
            if not lines or not all(board.plain_fit(i) for i in lines):
                return False
        return True

    def off_lane(self, lost: list[str]) -> set[str]:
        """Listings seen that are on no shelf of the sold-out lines' products: they cannot replace them."""
        lanes: set[str] = set()
        for number in range(1, len(self.items) + 1):
            lane = set(Board(self, number).lane())
            if set(lost) & lane:
                lanes |= lane
        return set(self.seen_order) - lanes if lanes else set()

    def plain_anywhere(self, ident: str) -> bool:
        """A listing whose own keys plainly state every requirement of the product whose shelf it is on."""
        return any(ident in board.lane() and board.plain_fit(ident) for board in (Board(self, n) for n in range(1, len(self.items) + 1)))

    def cheapest_after(self, lost: list[str]) -> list[str]:
        """A sold-out line under a cheapest pick: the cheapest unread rows of its shelf that name the requirements are read at once, below
        the best fit known so far. Left to wander, an agent ran out of steps with the lowest fitting price seen and unread."""
        if self.brief.get("pick") != "cheapest":
            return []
        keep = self.tail() + 3
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            if not set(lost) & set(board.lane()):
                continue

            def known() -> list[str]:
                return [i for i in board.fits() if i not in lost and not board.weigh(i)["open"]]

            for _ in range(3):
                best = known()
                limit = self.price_of(best[0]) if best and self.price_of(best[0]) is not None else float("inf")
                rows = [i for i in board.unread() if board.wording(i) and self.price_of(i) is not None and self.price_of(i) < limit]
                if not rows or self.left() <= keep or not self.fill_facts(rows[:1], keep):
                    break
            return known()[:4]
        return []

    def board_picks(self, lost: list[str]) -> list[str]:
        """Listings read so far whose own keys plainly state every requirement of a sold-out line's product: these need no model call."""
        picks: list[str] = []
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            if set(lost) & set(board.lane()):
                picks += [i for i in board.fits() if i not in lost and board.plain_fit(i)]
        return picks[:4]

    def split_note(self, label: str, key: str) -> str:
        """How many values a key has among the read listings of a shelf that still fit, for the planner."""
        cat_key = self.category_key()
        values: list[str] = []
        fitting = 0
        for ident in self.fit_rows:
            record = self.full.get(ident) or {}
            if cat_key and str(record.get(cat_key) or "listings") != label:
                continue
            fitting += 1
            for name, value in (_fact_block(record) or {}).items():
                if _words(name) == key:
                    values.append(str(value).strip().lower())
                    break
        return f"; {len(set(values))} values among {fitting} fitting" if fitting >= 2 and values else ""

    def page_of(self, ident: str, keep: int) -> dict[str, Any] | None:
        """A listing's page and attributes, read once; the record held for it, or None when no step is left for it."""
        tool, target = self.offer_tool(), self.single_target(self.offer_tool())
        if ident not in self.opened:
            if not tool or not target or self.left() <= keep or self.finished:
                return None
            faults: list[str] = []
            call = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            reply = self.act(tool, call) if not faults else None
            if reply is None or self.finished:
                return None
            self.observe(tool, reply.get("observation"))
            self.opened.add(ident)
        self.fill_facts([ident], keep)
        return self.full.get(ident) or self.rows.get(ident)

    def pick_page(self, shop: str, kind: str, words: list[str], taken: set[str], keep: int) -> tuple[str, dict[str, Any]] | None:
        """The listing a pointer describes. A shop sells lookalikes, so the shop's first row proves nothing: the listing meant is the one whose
        remark restates the pointer's words. Pages already open are looked at first, then each search's top rows of that shop."""
        wanted = [c for c in (_content(w) for w in words) if c]
        shop_low = shop.lower()

        def sold_by(record: dict[str, Any]) -> bool:
            return not shop_low or any(isinstance(v, str) and v.strip().lower() == shop_low for v in record.values())

        def standing(record: dict[str, Any]) -> tuple[int, int]:
            said = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"remark|note|comment", k, re.I))
            # a sentence about another version or another listing says nothing of this one: "A shirt collar version exists elsewhere"
            own = " ".join(x for x in re.split(r"(?<=[.;])\s+", said) if not _ELSEWHERE.search(x))
            remark = _content(own)
            whole = _content(_dump({k: v for k, v in record.items() if not re.search(r"remark|note|comment", str(k), re.I)})) | remark
            # "condition new" against a remark saying only "is new": half of a value's words are enough, but never another number.
            # The record's keys and values together must show all of it.
            told = sum(len(w & remark) * 2 >= len(w) and all(t in remark for t in w if t[:1].isdigit()) for w in wanted)
            return told, sum(w <= whole for w in wanted)

        best: tuple[tuple[int, int], str, dict[str, Any]] | None = None
        for ident in list(self.opened):
            record = self.full.get(ident) or self.rows.get(ident)
            if ident in taken or not isinstance(record, dict) or not sold_by(record):
                continue
            score = standing(record)
            if wanted and score[0] == len(wanted):
                return ident, record
            if best is None or score > best[0]:
                best = (score, ident, record)
        read = 0
        tried: set[str] = set()
        clean = [" ".join(t for t in re.findall(r"[\w.'-]+", w) if t.lower() not in _FILLER | {"made", "from", "by"}) for w in words]
        short = [" ".join(w.split()[-2:]) for w in clean]
        # the store ranks the listing meant first for its shop and its values, so its order is kept; a product word in the search can push it down
        for parts in ([shop] + clean, [shop] + short, [shop, kind] + short, [shop, kind], [kind] + short):
            phrase = " ".join(list(dict.fromkeys(" ".join(x for x in parts if x).split()))[:9])
            if not phrase or phrase in tried or read >= 6:
                continue
            tried.add(phrase)
            rows = self.search_phrase(phrase, keep)
            if not shop_low:
                # the shop's name was read as part of the description: a row's seller named among the pointer's words
                spoken = set(re.findall(r"[a-z0-9]+", " ".join([kind] + words).lower()))
                shop_low = next((v.strip().lower() for _, r in rows for k, v in r.items() if isinstance(v, str) and re.search(r"seller|shop|vendor|store", k, re.I) and v.strip().lower() in spoken), "")
                if shop_low:
                    wanted = [w for w in wanted if shop_low not in w] or wanted
            same = [r for r in rows if r[0] not in taken and r[0] not in self.opened and sold_by(r[1])]
            for ident, _ in same[:2]:
                record = self.page_of(ident, keep)
                if record is None:
                    break
                read += 1
                score = standing(record)
                if wanted and score[0] == len(wanted):
                    return ident, record
                if best is None or score > best[0]:
                    best = (score, ident, record)
        # no remark restates the whole description: a page showing all of it, or a remark saying all but one of four or more. A wrong hop reads a value with
        # confidence, a lost chain stays open
        if best and sold_by(best[2]) and (not wanted or best[0][1] == len(wanted) or (len(wanted) >= 4 and best[0][0] >= len(wanted) - 1)):
            return best[1], best[2]
        return None

    def partner_of(self, value: str, prop: str, last: dict[str, Any]) -> str:
        """The value the store sets beside this one. The chain's last page first, then every page read; a sentence about "its" property speaks of that page's own value."""
        want, name = _content(value), _content(prop)
        texts = [_dump({k: v for k, v in last.items() if isinstance(v, str) and re.search(r"remark|note|comment|about|description", k, re.I)})]
        texts += [_dump({k: v for k, v in r.items() if isinstance(v, str) and re.search(r"remark|note|comment", k, re.I)}) for r in self.full.values() if isinstance(r, dict) and r is not last]
        for index, text in enumerate(texts):
            for left, right in _pairings(text):
                words = _content(left)
                if want and want <= words:
                    return right
                if index == 0 and (words <= name | {"item", "this", "it", "its", "s"} or name <= words):
                    return right
        return ""

    def walk_need(self, need: str, keep: int) -> str:
        """Walk one requirement's chain from its own description to the page that gives the value, and read the value the way the requirement
        asks for it. Returns what was found, or an empty string when the chain could not be finished."""
        got = self.ask_json(POINTED + need, [LIGHT, HEAVY], 1500) or {}
        prop = str(got.get("property") or need.split(":")[0]).strip()
        relation = str(got.get("relation") or "same").strip().lower()
        shop, kind = str(got.get("shop") or "").strip(), str(got.get("kind") or "").strip()
        words = [str(w).strip() for w in _items(got.get("words")) if str(w).strip()]
        if not shop:
            # a name inside the description, written as a name
            named = re.findall(r"\b[A-Z][a-z]{3,}\b", need.partition(":")[2] or need)
            shop = named[0] if named else ""
        words = [w for w in words if not _content(w) <= _content(kind + " " + shop)]
        mark = _dump([prop.lower(), relation, shop.lower(), sorted(_plain(w) for w in words)])
        if mark in self.walk_memo:
            self.ref_values[need] = self.walk_memo[mark]
            self.version += 1
            return f"{need} -> the value to use is {self.walk_memo[mark]}"
        taken: set[str] = set()
        path: list[str] = []
        value, last = "", None
        for _ in range(8):
            if self.left() <= keep or self.elapsed() >= ROOM_SECONDS or not (shop or kind or words):
                break
            found = self.pick_page(shop, kind, words, taken, keep)
            if not found:
                break
            ident, last = found
            taken.add(ident)
            path.append(ident)
            said = " ".join(str(v) for k, v in last.items() if isinstance(v, str) and re.search(r"remark|note|comment", k, re.I))
            page = "REMARKS: " + said + "\nRECORD: " + _dump(_clip(last, 300))[:2500]
            hop = self.ask_json(HOP + f"PROPERTY: {prop}\nPAGE: {page}", [LIGHT, HEAVY], 1500) or {}
            if hop.get("points") is True and (hop.get("shop") or (hop.get("words") and not _pairings(said))):
                shop, kind = str(hop.get("shop") or "").strip(), str(hop.get("kind") or "").strip()
                words = [str(w).strip() for w in _items(hop.get("words")) if str(w).strip()]
                words = [w for w in words if not _content(w) <= _content(kind + " " + shop)]
                continue
            value = str(hop.get("value") or "").strip()
            if not value:
                # the page's own attribute on that property, when the reading came back empty
                wanted = _content(prop)
                table = _fact_block(last) or {}
                named = [v for k, v in table.items() if _content(k) == wanted] or [v for k, v in table.items() if wanted and wanted <= _content(k)]
                value = str(named[0]).strip() if named else ""
            break
        self.trace.append({"kind": "walk", "need": need[:90], "relation": relation, "path": path, "value": value})
        if not value or last is None:
            return ""
        if relation == "partner":
            partner = self.partner_of(value, prop, last)
            if not partner and said:
                read = self.ask_json(PARTNER + f"PROPERTY: {prop}\nVALUE: {value}\nPAGE: {said[:1500]}", [LIGHT, HEAVY], 1500) or {}
                partner = str(read.get("partner") or "").strip()
            self.trace.append({"kind": "walk_partner", "value": value, "partner": partner})
            value = partner or value
        elif relation == "unlike":
            value = f"not {value}"
        self.ref_values[need] = value
        self.walk_memo[mark] = value
        self.version += 1
        return f"{need} -> pages opened: {', '.join(path)}; the value to use is {value}"

    def compat_line(self) -> str:
        """On a several-product request with a hub: the whole orders the listings seen allow, lowest total first. The best single row of each
        product says nothing about the best order: the lowest total can sit on a dearer hub whose parts are cheap."""
        hub = self.hub_item()
        if not hub:
            return ""
        boards = {n: Board(self, n) for n in range(1, len(self.items) + 1)}

        def usable(board: Board, ident: str) -> bool:
            return not self.sold_out(ident) and self.price_of(ident) is not None and not board.weigh(ident)["no"]

        heads: dict[str, str] = {}
        for ident in boards[hub].lane():
            model = _compat_of(self.text_of(ident))["model"]
            if model and usable(boards[hub], ident) and (model not in heads or self.price_of(ident) < self.price_of(heads[model])):
                heads[model] = ident
        whole, short = [], []
        for model, head in heads.items():
            lines = [head]
            total = self.price_of(head) * self.items[hub - 1].qty
            gap = ""
            for number, board in boards.items():
                if number == hub:
                    continue
                good = [i for i in board.lane() if usable(board, i) and _fits_model(_compat_of(self.text_of(i)), model)]
                if not good:
                    gap = self.items[number - 1].name
                    break
                best = min(good, key=lambda i: self.price_of(i))
                lines.append(best)
                total += self.price_of(best) * self.items[number - 1].qty
            if gap:
                short.append(f"{model} (no {gap} row seen yet)")
            else:
                whole.append((total, f"{model}: " + " + ".join(lines) + f" = {total:g}"))
        if not whole and not short:
            return ""
        whole.sort()
        text = "\n[orders the listings' model wording allows, lowest total first] " + " | ".join(w for _, w in whole[:3])
        return text + (" [not complete yet] " + ", ".join(short[:3]) if short else "")

    def find_pointed(self) -> str:
        """Each requirement whose value is on another listing: its chain is walked at once, before steps go to anything its value decides."""
        live = {need for _, need in self.needs_now()}
        # fitting another requested product is settled by the products chosen, not by a page to read first
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.walked
                   and not re.search(r"compatib|\bfits?\b", need.partition(":")[0], re.I)]
        if not pending or not self.offer_tool() or not self.single_target(self.offer_tool()):
            return ""
        keep = self.tail() + 6
        said = []
        for need in pending[:3]:
            if self.left() <= keep or self.elapsed() >= ROOM_SECONDS:
                break
            self.walked.add(need)
            done = self.walk_need(need, keep)
            if done:
                said.append(done)
        chain, _ = self.trail()
        if not said and not chain:
            return ""
        return "[Tradewind followed the listings these requirements point at]\n" + "\n".join(said) + ("\n" + chain if chain else "")

    def part_row(self, code: str, keep: int) -> str:
        """The listing a part code names: its title carries the code; other listings only mention it."""
        for ident in list(self.opened):
            record = self.full.get(ident) or self.rows.get(ident) or {}
            if code in " ".join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"title|name|head", k, re.I)):
                return ident
        rows = self.search_phrase(code, keep)
        titled = [i for i, r in rows if code in " ".join(str(v) for k, v in r.items() if isinstance(v, str) and re.search(r"title|name|head", k, re.I))]
        named = [i for i, r in rows if code in _dump(r)]
        return (titled or named or [""])[0]

    def look_up_parts(self, keep: int) -> None:
        """On a several-product cart, a line that points to a part: that part's row is looked up once before the order, by its code. Its row is what counts, so it is not opened."""
        if len(self.items) < 2:
            return
        titles = " ".join(str(v) for r in self.rows.values() for k, v in r.items() if isinstance(v, str) and re.search(r"title|name|head", k, re.I))
        for ident in self.cart_ids():
            record = self.full.get(ident) or self.rows.get(ident) or {}
            for code in _MORE_PART.findall(" ".join(str(v) for v in record.values() if isinstance(v, str))):
                if code in self.parts_done or code in titles or self.left() <= keep or self.finished:
                    continue
                self.parts_done.add(code)
                if self.search_phrase(code, keep):
                    self.trace.append({"kind": "part_row", "code": code})

    def open_parts(self) -> str:
        """A cart line's text can send the reader to a part "for the rest": those pages are opened before the order counts, and any part they name."""
        keep = self.tail() + 1
        said: list[str] = []
        queue: list[str] = []
        for ident in self.cart_ids():
            record = self.full.get(ident) or self.rows.get(ident) or {}
            queue += _compat_of(" ".join(str(v) for v in record.values() if isinstance(v, str)))["parts"]
        self.look_up_parts(keep)
        while queue and len(said) < 6 and self.left() > keep and not self.finished:
            code = queue.pop(0)
            if code in self.parts_done:
                continue
            self.parts_done.add(code)
            ident = self.part_row(code, keep)
            record = self.page_of(ident, keep) if ident else None
            if not record:
                continue
            said.append(f"{code} = {ident}")
            queue += _compat_of(" ".join(str(v) for v in record.values() if isinstance(v, str)))["parts"]
        if said:
            self.trace.append({"kind": "parts", "opened": said})
        return "[Tradewind opened the parts the cart's lines refer to] " + "; ".join(said) if said else ""

    def text_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return " ".join(str(v) for v in record.values() if isinstance(v, str))

    def hub_item(self) -> int:
        """The product the others must fit: its listings state a model, the other products' listings state what they are compatible with."""
        if len(self.items) < 2:
            return 0
        models, fits, named = {}, {}, {}
        for number in range(1, len(self.items) + 1):
            compat = [_compat_of(self.text_of(i)) for i in Board(self, number).lane()]
            models[number] = sum(bool(c["model"]) for c in compat)
            fits[number] = sum(bool(c["fits"]) for c in compat)
            named[number] = sum(bool(c["named"]) for c in compat)
        hubs = [n for n in models if models[n] >= 2 and models[n] > fits[n]]
        # a spec sheet's "Model: X" on another shelf does not make a second hub when one shelf names its models as references
        if any(named[n] >= 2 for n in hubs):
            hubs = [n for n in hubs if named[n] >= 2]
        return hubs[0] if len(hubs) == 1 and any(fits[n] >= 2 for n in fits if n != hubs[0]) else 0

    def compat_fault(self) -> str:
        """A cart line whose own text rules out the model of the hub product chosen, on a request that asks for things to fit together."""
        hub = self.hub_item()
        if not hub or self.compat_held >= 2 or not re.search(r"compatib|\bfits?\b|work(?:s|ing)? with|go(?:es)? with|mates? |pairs? ", self.request, re.I):
            return ""
        lane = set(Board(self, hub).lane())
        cart = self.cart_ids()
        heads = [i for i in cart if i in lane]
        model = _compat_of(self.text_of(heads[0]))["model"] if heads else ""
        if not model:
            return ""
        bad = []
        for ident in cart:
            compat = _compat_of(self.text_of(ident))
            if ident not in lane and compat["fits"] and not compat["parts"] and not _fits_model(compat, model):
                bad.append(ident)
        if not bad:
            return ""
        self.compat_held += 1
        notes = []
        for ident in bad:
            shelf = next((set(Board(self, n).lane()) for n in range(1, len(self.items) + 1) if ident in Board(self, n).lane()), set())
            good = [i for i in self.seen_order if i in shelf and i != ident and not self.sold_out(i) and _fits_model(_compat_of(self.text_of(i)), model)]
            good.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            notes.append(f"{ident} does not fit it" + (f"; rows that state they do: {', '.join(good[:6])}" if good else "; search that product again for rows stating this model or its family"))
        return f"The chosen {self.items[hub - 1].name} is model {model}. " + " | ".join(notes) + ". Replace those lines, or choose another model that every line fits, then check out."

    def text_search(self) -> tuple[str, dict[str, Any], str] | None:
        """The newest search made with free words and the argument that took them; else the search tool and its words argument.
        A shelf, section or shop filter is no place for words: a shop's name searched as a section finds nothing."""
        for name, args, _ in reversed(self.searches):
            props = (self.schemas.get(name) or {}).get("properties") or {}
            texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and "enum" not in (props.get(k) or {})
                     and not re.search(PAGE_ARG, k, re.I)]
            named = [k for k in texts if re.search(WORDS_ARG, k, re.I)]
            loose = [k for k in texts if not re.search(FILTER_ARG, k, re.I)]
            pick = (named or loose or [None])[0]
            if pick:
                return name, args, pick
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        named = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(WORDS_ARG, k, re.I)]
        return (tool, {}, named[0]) if tool and named else None

    def search_phrase(self, phrase: str, keep: int) -> list[tuple[str, dict[str, Any]]]:
        """One search with these words, in the shape of the newest search that took words; the rows it returned."""
        if not phrase or self.left() <= keep:
            return []
        found = self.text_search()
        if not found:
            return []
        name, args, text_arg = found
        props = (self.schemas.get(name) or {}).get("properties") or {}
        faults: list[str] = []
        call = _shape({**self.word_base(name, args), text_arg: phrase}, self.schemas.get(name), "args", faults)
        if faults:
            return []
        reply = self.act(name, call)
        if reply is None or self.finished:
            return []
        seen = reply.get("observation")
        self.observe(name, seen)
        self.searches.append((name, dict(call), seen))
        return self.records_in(seen)

    def word_base(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The fixed-choice arguments a search on words keeps, with a mode argument set to search by words whatever the last search of that tool did."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        base = {k: v for k, v in args.items() if (props.get(k) or {}).get("enum") is not None}
        for key, spec in props.items():
            values = spec.get("enum") if isinstance(spec, dict) else None
            chosen = [v for v in values or [] if isinstance(v, str) and re.search(r"word|phrase|text|keyword|query", v, re.I)]
            if chosen:
                base[key] = chosen[0]
        return base

    def plain_search(self) -> str:
        """Once, after the answers, on a cheapest pick: a store label that joins several product names ("x and y") hides listings a browse of the
        whole shelf reaches late or never, and a search on one name alone ranks them first. One step a name, inside the steps kept for the rest."""
        if self.plain_done or self.brief.get("pick") != "cheapest":
            return ""
        self.plain_done = True
        found = self.text_search()
        if not found:
            return ""
        name, args, text_arg = found
        base = self.word_base(name, args)
        texts: list[str] = []
        seen_rows: list[Any] = []
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            # the label most of this product's rows carry, as the store spells it; the request's own name may use only part of it
            key = self.category_key()
            shown = [str((self.rows.get(i) or {}).get(key) or "") for i in board.lane()] if key else []
            label = max(sorted(set(shown)), key=shown.count) if shown else ""
            # every row repeats its shelf's label: only the rows' own words say how this store spells a product
            own_text = " ".join(self.text_of(i) for i in board.lane()).lower().replace(label.lower() or "\0", " ")
            for phrase in _plain_names(label or board.item.shelf or board.item.name, own_text):
                if self.left() - self.tail() - 14 < 1 or self.elapsed() >= ROOM_SECONDS or self.finished:
                    break
                faults: list[str] = []
                call = _shape({**base, text_arg: phrase}, self.schemas.get(name), "args", faults)
                if faults:
                    break
                reply = self.act(name, call)
                if reply is None or self.finished:
                    break
                seen = reply.get("observation")
                self.observe(name, seen)
                self.searches.append((name, dict(call), seen))
                seen_rows.append(seen)
                texts.append(f"{_dump(call)}: " + self.brief_rows(seen))
        self.trace.append({"kind": "plain_search", "read": len(texts)})
        return ("\n".join(texts) + self.cheapest_here(seen_rows)) if texts else ""

    def told_rows(self, since: int) -> str:
        """After the answers, for each product that stated a need: one search on the requirement values alone, then the rows of the searches since the answers are read before anything
        else, in the store's order (the cheapest first under a cheapest pick), until one's own keys state every requirement. A store's search can rank a listing on its attributes while
        its title names something else, so a row's wording says little here."""
        if self.value_pending():
            return ""
        boards = [b for b in (Board(self, n) for n in range(1, len(self.items) + 1)) if any(need["heard"] for need in b.needs)]
        if not boards:
            return ""
        keep = self.tail() + 8
        share = max(3, TOLD_READS // len(boards))
        pick = self.brief.get("pick")
        texts: list[str] = []
        for board in boards:
            heard_first = sorted(board.needs, key=lambda need: not need["heard"])
            values = list(dict.fromkeys(need["values"][0] for need in heard_first if need["values"]))
            phrase = " ".join(" ".join(values).split()[:6])
            asked = {str(v) for _, args, _ in self.searches for v in args.values()}
            if phrase and phrase not in asked:
                self.search_phrase(phrase, keep)
            lane = set(board.lane())
            rows: list[str] = []
            for _, _, seen in reversed(self.searches[since:]):
                for ident, _ in self.records_in(seen) if isinstance(seen, (dict, list)) else []:
                    if ident in lane and ident not in rows:
                        rows.append(ident)
            if pick == "cheapest":
                rows.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            reads = 0
            for ident in rows:
                if ident not in self.full:
                    if reads >= share or self.left() <= keep or self.elapsed() >= ROOM_SECONDS or self.finished:
                        break
                    if not self.buyable(ident) or not board.affordable(ident):
                        continue
                    got = self.fill_facts([ident], keep)
                    if not got:
                        continue
                    reads += 1
                    texts.append(got[0][:500])
                if pick not in ("most", "least") and board.plain_fit(ident) and not board.weigh(ident)["no"]:
                    break
        self.trace.append({"kind": "told_rows", "read": len(texts)})
        return "\n".join(texts)

    def search_told(self) -> str:
        """After the answers, one search per product that has stated needs, on their values: what the shopper said, then what the
        request said, then the product. Rows and any further pages come back for the agent."""
        texts = []
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            if not any(need["heard"] for need in board.needs) or not self.search_again(board, self.tail() + 8, True):
                continue
            name, args, seen = self.searches[-1]
            paged = self.more_pages(name, args, seen)
            texts.append(f"{_dump(args)}: " + self.brief_rows(seen) + paged + self.cheapest_here([seen] + self.paged_obs))
        return "\n".join(texts)

    def search_again(self, board: Board, keep: int, fresh: bool = False) -> bool:
        """One more look for a product: the next page of a search that showed its listings, else a fresh search in the same
        call shape with the product's name and the requirements' values. With `fresh`, only the new search."""
        if self.left() <= keep or self.elapsed() >= ROOM_SECONDS or not self.searches:
            return False
        lane = set(board.lane())
        for name, args, seen in [] if fresh else reversed(self.searches):
            shown = {i for i, _ in self.records_in(seen)} if isinstance(seen, (dict, list)) else set()
            if lane and not shown & lane:
                continue
            plan = self.paging(name, seen)
            if not plan or plan[3] is None:
                continue
            mark = name + _dump(args) + _dump(plan[3])
            if mark in self.paged_marks:
                continue
            self.paged_marks.add(mark)
            faults: list[str] = []
            call = _shape({**args, plan[2]: plan[3]}, self.schemas.get(name), "args", faults)
            if faults:
                continue
            reply = self.act(name, call)
            if reply is None or self.finished:
                return False
            self.observe(name, reply.get("observation"))
            self.searches.append((name, dict(call), reply.get("observation")))
            self.trace.append({"kind": "sweep_page", "item": board.number, "args": _clip(call, 80)})
            return True
        found = self.text_search()
        if not found:
            return False
        name, args, text_arg = found
        said = list(dict.fromkeys([need["values"][0] for need in reversed(board.needs) if need["heard"]] + [need["values"][0] for need in board.needs if not need["heard"]]))
        words = " ".join((" ".join(said[:2]) + " " + board.item.name).split()[:6]).strip()
        mark = name + words
        if not words or mark in self.paged_marks:
            return False
        self.paged_marks.add(mark)
        faults = []
        call = _shape({**self.word_base(name, args), text_arg: words}, self.schemas.get(name), "args", faults)
        if faults:
            return False
        reply = self.act(name, call)
        if reply is None or self.finished:
            return False
        self.observe(name, reply.get("observation"))
        self.searches.append((name, dict(call), reply.get("observation")))
        self.trace.append({"kind": "sweep_search", "item": board.number, "words": words})
        return True

    def sweep(self) -> str:
        """Before an ending without a purchase: spare steps go to the products still lacking a fit (their unread rows, further
        pages, a fresh search). Returns what to tell the agent when every product now has a listing nothing speaks against."""
        if self.swept or not self.items:
            return ""
        self.swept = True
        keep = self.tail() + len(self.items) + 3
        boards = [Board(self, n) for n in range(1, len(self.items) + 1)]
        lacking = [b for b in boards if not b.fits()]
        spare = self.left() - keep
        texts: list[str] = []
        for board in lacking:
            share = min(SWEEP_PER_ITEM, max(2, spare // max(1, len(lacking))))
            reads = 0
            # one listing at a time, until this product has a fit or its share of the spare steps is spent
            while reads < share and self.left() > keep and self.elapsed() < ROOM_SECONDS and not self.finished and not board.fits():
                rows = board.unread()
                if not rows:
                    if not self.search_again(board, keep):
                        break
                    rows = board.unread()
                    if not rows:
                        break
                got = self.fill_facts(rows[:1], keep)
                if not got:
                    break
                texts.append(got[0][:500])
                reads += 1
        named = {str(b.number): b.fits()[:4] for b in boards}
        self.trace.append({"kind": "sweep", "lacking": [b.number for b in lacking], "read": len(texts), "named": named})
        # a fit with nothing against it and no requirement left unaddressed holds the ending back too: a store with no fit shows none
        settled = all(any(not b.weigh(i)["open"] for i in b.fits()) for b in boards) and not self.value_pending()
        if not all(named.values()) or not (lacking or settled):
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        head = ("Tradewind read more listings first:\n" + "\n".join(texts) + "\n") if texts else ""
        return head + "These listings' own records state nothing against the requirements: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + ". Open them if needed, put the best one for each item in the cart (respecting the pick rule) and check out. End without a purchase only if each one's record clearly fails a requirement."

    def price_field(self, name: str) -> str | None:
        """The one numeric argument of a search tool that limits the price, by its name."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        found = [k for k, v in props.items() if isinstance(v, dict) and set(_kinds(v)) & {"number", "integer"}
                 and re.search(r"price|cost|cap|limit|budget|ceiling|max", k, re.I) and not re.search(r"page|start|offset|skip|from|begin|min", k, re.I)]
        return found[0] if len(found) == 1 else None

    def cap_sweep(self) -> None:
        """A cheapest pick and a search tool that takes a price limit: the shelf of each cart line is listed below that line's
        price, so that cheaper rows are seen wherever they sit in the store's order. A step a line, once per price."""
        if self.brief.get("pick") != "cheapest" or not self.searches:
            return
        for number in range(1, len(self.items) + 1):
            lane = set(Board(self, number).lane())
            for line in self.cart_ids():
                price = self.price_of(line)
                mark = f"{number}|{line}|{price}"
                if line not in lane or price is None or price <= 0 or mark in self.cap_marks or self.left() <= self.tail() + 6:
                    continue
                self.cap_marks.add(mark)
                for name, args, seen in reversed(self.searches):
                    field = self.price_field(name)
                    shown = {i for i, _ in self.records_in(seen)} if isinstance(seen, (dict, list)) else set()
                    if not field or not shown & lane:
                        continue
                    limit = price - 1 if float(price).is_integer() else price * 0.99
                    base = {k: v for k, v in args.items() if not re.search(r"page|token|cursor|next|offset|skip|start|begin|from", k, re.I)}
                    faults: list[str] = []
                    call = _shape({**base, field: int(limit) if float(limit).is_integer() else limit}, self.schemas.get(name), "args", faults)
                    if faults:
                        break
                    reply = self.act(name, call)
                    if reply is not None and not self.finished:
                        self.observe(name, reply.get("observation"))
                        self.searches.append((name, dict(call), reply.get("observation")))
                        self.trace.append({"kind": "cap_sweep", "line": line, "limit": limit})
                    break

    def finish_cart(self, settled: bool = False, unread: bool = False) -> None:
        """A forced finish: products the cart lacks get the best listing their board shows, when every product then has a
        line. Nothing changes when one of them has none. With `settled`, only listings that leave no requirement unaddressed;
        with `unread`, a product with no fit takes its best unread row."""
        cart_tool = self.roles.get("cart")
        if self.finished or not cart_tool or not self.items or self.left() < 2:
            return
        now = {i for i in self.cart_ids() if not self.sold_out(i)}
        lines: dict[str, int] = {}
        for number in range(1, len(self.items) + 1):
            board = Board(self, number)
            if now & set(board.lane()):
                continue
            fits = [i for i in board.fits() if i not in lines and not (settled and board.weigh(i)["open"])]
            if not fits and unread:
                fits = [i for i in sorted(board.unread(), key=lambda r: -board.wording(r)) if i not in lines]
            if not fits:
                return
            lines[fits[0]] = board.item.qty
        if not lines:
            return
        calls = self.cart_calls({**{i: self.cart_lines.get(i, 1) for i in now}, **lines}, [i for i in self.cart_ids() if self.sold_out(i)])
        if not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart_tool), "args", faults)
            if faults:
                return
            reply = self.act(cart_tool, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart_tool, args)
            self.observe(cart_tool, reply.get("observation"))
        self.trace.append({"kind": "finish_cart", "lines": lines})

    def unread_rows(self) -> list[str]:
        """Rows never read, on a shelf where a listing was read, in stock and within the budget, those naming the most words
        of the requirements in force first."""
        cat_key = self.category_key()
        shelves = {_words(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in self.opened} if cat_key else set()
        shelves.discard("")
        wanted: frozenset[str] = frozenset()
        for item in self.items:
            for need in item.musts + item.told:
                if need not in item.dropped:
                    wanted |= _stem(need.split(":", 1)[-1])
        ranked = []
        for ident in self.seen_order:
            if ident in self.opened or ident in self.full or self.sold_out(ident):
                continue
            row = self.rows.get(ident) or {}
            if shelves and _words(str(row.get(cat_key) or "")) not in shelves:
                continue
            price = self.price_of(ident)
            if isinstance(self.budget, (int, float)) and price is not None and price > self.budget:
                continue
            ranked.append((-len(wanted & _stem(_dump(row)[:900])), ident))
        ranked.sort(key=lambda pair: pair[0])
        return [ident for _, ident in ranked]

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        shelves = {_words(item.shelf) for item in self.items if item.shelf}
        cat_key = self.category_key()
        for ident in reversed(self.seen_order):
            if self.sold_out(ident) or (shelves and cat_key and _words(str((self.rows.get(ident) or {}).get(cat_key) or "")) not in shelves):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 20:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[: 4 if self.left() >= 10 else 2]]
        else:
            return False
        if self.left() < len(plan) + 5:
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
        """The agent ends without buying although the independent check confirmed a reviewed listing: Tradewind carts the reviewed
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
            self.note_cart(cart, args)
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
        return ("Not run (Tradewind acted instead): the independent check confirmed " + self.confirmed + ", so Tradewind set the cart (" + "; ".join(done)
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
        elif left <= len(self.items) + 7 and self.buys == 0:
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
        untried = {n for n in self.schemas if n not in self.facts_from}
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Tradewind opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Tradewind's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Tradewind] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in untried and self.facts_from.get(name) is False and self.facts_tool() not in (None, name):
                out.append((cid, f"Not run (no step used): {name} shows no attributes; {self.facts_tool()} does. Call that one for the listings you want to judge."))
                continue
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
                    trimmed += f"\n[Tradewind] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    held_back = self.off_shelf() or self.broken_line() or self.cheaper_fit() or self.firm_up()
                    if held_back:
                        out.append((cid, "Not run yet. " + held_back + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 5:
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
                    misfit = self.compat_fault()
                    if misfit:
                        out.append((cid, "Not run yet (no step used). " + misfit + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    self.cap_sweep()
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3)
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Tradewind read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Tradewind re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5:
                        swept = self.sweep()
                        if swept:
                            out.append((cid, "Not run yet (no step used). " + swept))
                            continue
                        review = self.audit() if len(self.items) == 1 else ""
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
            if self.is_refuse(name, args) and len(self.items) > 1 and self.multi_held < 2 and self.left() > 3 and not self.no_fit_shown():
                self.multi_held += 1
                self.trace.append({"kind": "several_held", "left": self.left()})
                out.append((cid, "Not run yet (no step used). " + self.several_note()))
                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
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
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            several = self.is_refuse(name, args) and len(self.items) > 1 and self.left() >= 2 and not self.no_fit_shown()
            if several:
                # a several-product request is a purchase: such orders pass now and then, and the ending has never been right
                self.finish_cart(False, True)
            if several and self.every_line() and self.roles.get("buy"):
                faults = []
                call = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(self.roles["buy"]), "args", faults)
                if not faults:
                    self.trace.append({"kind": "several_checkout", "cart": self.cart_ids()})
                    name, args = self.roles["buy"], call
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and 2 <= self.left() <= 5 and not self.value_pending():
                self.finish_cart(True)
            if self.is_refuse(name, args) and 1 <= self.left() <= 5 and self.cart_stands():
                faults = []
                call = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(self.roles.get("buy") or ""), "args", faults)
                if self.roles.get("buy") and not faults:
                    self.trace.append({"kind": "late_checkout", "cart": self.cart_ids()})
                    name, args = self.roles["buy"], call
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and self.refused_at >= 0 and not self.kept_deal and self.roles.get("buy") and self.cart_stands():
                # the shopper answered the checkout by changing the deal: the same cart still stands, so the order stays unless the new budget rules it out
                spent = 0
                for number, item in enumerate(self.items, 1):
                    lane = Board(self, number).lane()
                    spent += sum((self.price_of(i) or 0) * item.qty for i in self.cart_ids() if i in lane)
                if not isinstance(self.budget, (int, float)) or spent <= self.budget:
                    faults = []
                    call = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(self.roles["buy"]), "args", faults)
                    if not faults:
                        self.kept_deal = True
                        self.trace.append({"kind": "deal_changed_kept", "cart": self.cart_ids(), "spent": spent, "budget": self.budget})
                        name, args = self.roles["buy"], call
                        mark = name + _dump(args)
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
                    trimmed = "\n[Tradewind] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Tradewind] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
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
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            if name not in plain and not ending:
                self.searches.append((name, dict(args), observation))
                self.read_shape(name, observation)
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
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Tradewind] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Tradewind re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Tradewind's own calls] " + " | ".join(self.queued)
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
        self.finish_cart(False, len(self.items) > 1)
        if self.finished or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2 and self.elapsed() < LAST_SECONDS:
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
                if self.left() >= 2 and self.elapsed() < LAST_SECONDS:
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
        walked = self.find_pointed()
        if walked:
            intro += "\n\n" + walked
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Tradewind] Tradewind tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and self.elapsed() < ROOM_SECONDS:
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
    try:
        episode = Tradewind(problem_data)
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
