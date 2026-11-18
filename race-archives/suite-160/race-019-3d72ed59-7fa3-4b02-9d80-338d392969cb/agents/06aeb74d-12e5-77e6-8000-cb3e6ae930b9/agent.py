"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

Added in this version: a reference follower (code that finds the listing a
requirement points at, opens every page of the chain with the page tool and
writes the value it reads into the ledger), a gate that holds back an ending
without a purchase until the open steps are taken, a fit rule for items tied
to a model code, fewer repeated reads of the cart's offers, and a clock for
the 300 s episode.
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

LOOP_CALLS = 42
LOOP_CALLS_MANY = 48        # actor turns when the request has three items or more
ALL_CALLS = 80
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_EXTRA = 0.008         # most the reference follower's own model calls may add to both spend limits (a chain costs under 0.002)
FOLLOWER_SPEND = 0.02       # most the reference follower may spend on its own model calls in an episode
CLOCK_FAST = 120            # seconds after which the light model works without thinking, so a slow episode still finishes
CLOCK_SOFT = 220            # seconds into the 300 s episode after which the actor loop stops and the best cart is checked out
CLOCK_HARD = 262            # seconds after which no new model call starts
RESOLVE_STEPS = 36          # store steps the reference follower may use in an episode (a third of the steps or a little more)
RESOLVE_HOPS = 6            # pages one chain may pass
RESOLVE_OPENS = 5           # candidate pages opened for one hop at the first level of effort
RESOLVE_CALLS = 24          # model calls the reference follower may make in an episode
DECLINE_HOLDS = 4           # times an ending without a purchase is held back in an episode
REREAD_AGE = 25             # steps after which a cart line's offer is read again before a checkout (baskets of four lines or more)
HOP_STEPS = 16              # store steps the reference follower may spend on one hop of a chain
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever the title or other properties say. A listing can carry one property under two keys (two spellings of the same name); when one of them states the required value, the requirement is met even if the other says something else. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When the request asks for several items that go together (a device and accessories for it), judge every item by the requirements the request states for it. A printed compatibility line on the listings ("fits part series ...", "works with every ... model", a series code in a title) is not one of the request's requirements and is not checked: never search for series codes, never skip a listing that meets its requirements because that line does not match, never pay more for a matching line, and never end without a purchase because of it. Only a model code the shopper names (see below) is checked against such a line.
- A request can tie an item to a model code the shopper names. Such an item fits only when the listing's own compatibility sentence (the first sentence or list of its description) names that exact code, or a family pattern the sentence itself writes (such as "every XYZ21xx model") unless it excepts the code; a later line saying that another part also fits does not count. A shopper's vehicle code usually matches nothing in a search (the compatibility sentence is not indexed): when a search with the code returns no rows, search by the product's words and read each description; series codes written in titles can still be searched. An item the request frees from the code is never filtered by it. Kyo marks rows with [fit] lines.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; "a pair of X" and "one X sold as a pair" are 1, but "two X sold as a pair" is 2: the pair is how one listing is sold, which is a must on its pack form), "fit_code": the model code of the shopper's vehicle or device (letters and digits, nothing else) that the request says THIS item must fit; null when the request names none for it or frees it from the code ("does not matter", "not needed", "does not apply"), even if it names another machine for it (the code is never a must), "price_each": the most ONE unit of this item may cost when the request caps it ("at most 900 cents each", "no more than 12 dollars apiece" -> 900, 12), as a number in the request's own unit, else null, "pairs_with": the number (counting from 1 in this list) of ANOTHER requested item this item must work with, fit or be compatible with ("two cables that work with that phone" -> the phone's number), else null, "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A make, brand, maker, vehicle or machine the request binds one item to ("a caliper for a Honda", "a Bosch filter", "mats for a Nissan") is a must on that item ("make: honda", "brand: bosch") even though a model code never is; a machine named as the setting of the whole errand is not. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; a vehicle or machine named only to say what a part is for is context as well, and when the request frees an item from the model code, never write "fits <machine>" or "compatible with <machine>" among that item's musts. Words written into the product's name that state one of its properties are musts as well (keep them in "name" too): what it is made of, its colour, size or style, its pack form (a multipack, a pair, a set, a number of pieces; "sold as 1 piece" gives "pack size: 1 piece"), its position (upper, lower, inner, outer), its condition, a feature or service it comes with, and the occasion, season or audience it is meant for ("a multipack of green tea for the office" gives "pack type: multipack"; "linen shirts for weddings" gives "material: linen" and "occasion: wedding"; "queen beds in a modern style" gives "size: queen" and "style: modern"). Only the words that name the kind of product itself (its store department) are not musts. A make or vehicle named after the last of several items ("A and B for a Toyota") belongs to that last item only, unless the request says all of them are for it; words that spread a model code over several items spread the code and never a make. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase), "status" (reports how many messages to the shopper are allowed, left or used; null when no tool does), "browse" (lists the listings of one shelf, aisle or section named in its arguments, often under a price cap; null when no tool does).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"search_arg": the name of the argument of the "search" tool that takes the words to search for, else null.
"search_args": the fixed arguments that make the "search" tool search by the words of a query (for a tool with a mode argument, that mode), else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish, a hint about the place it goes (a room, a corner, a space it must fit) at its size or one of its dimensions such as height or width (whichever the request leaves open).
2. Vehicle parts and accessories. An item marked "must fit a model code" is one, and so is any other part or accessory of a car or a motorbike (a rotor, a caliper, a sway bar, a ball joint, a bearing, a seal, a floor mat, a windshield). What the shopper holds firm on is nearly always a property that tells such listings apart: its type, its position on the vehicle (upper, lower, inner, outer, front, rear), its pack size (1 piece, a pair, a set), the make or model it suits, its colour, its material or its fit type. Take them from the keys that item's listings carry with different values, and skip one the request already settles. Floor mats, windshields, shades and other accessories can hide a colour, a material or a fit type just as a part can hide a position. Give every such item one question before any item gets a second one. Put these after the hinted properties and before the ranked keys.
3. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
With three or more items, take turns between them so that each item gets one question before any item gets a second.
Skip a property the request already settles for that item, a second name for a property already chosen (a key that repeats another key's name with one extra word, such as a search filter's copy of it), a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, lists of compatible models, warranty and other logistics. A measure the story mentions (a height, width, length or weight) is a property too. Prefer properties on which one item's listings differ among themselves and that take a few plain values. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike, compatible with, at least or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true, whatever else the answer says. When the answer names a shop, a kind of product and properties of a listing instead of a value, it is such a pointer. Its value is the answer's own words copied exactly, from the comparison words to the end of that description and any words after it that finish the comparison ("or above", "or more", "or less"), including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails (when the listing has two keys for that property, one that states the value is enough); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement to fit a model code is met only when the listing's own compatibility sentence names that exact code (or a family pattern the sentence itself writes, unless it excepts the code), and a later line saying another part also fits does not count. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, and of two keys for one property the one that states the value counts; one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, but a value that leaves one of the required words out does not, and neither does a key's different value, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement to fit a model code is met only when the listing's own compatibility sentence names that exact code (or a family pattern the sentence itself writes, unless it excepts the code); out-of-stock listings and listings that break the budget do not fit."""

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

DESCRIBE = """You read a pointer in a shopping request. A requirement of the shopper takes its value from ANOTHER listing (the target) instead of stating it. Using the REQUEST, the REQUIREMENT and the shopper's own words when they are given, return ONE JSON object and nothing else:
{"pointer": true when the requirement takes its value from another listing, else false,
 "property": "the property of the BOUGHT product whose value is wanted, in plain words",
 "relation": one of "same" (the value that listing itself has), "partner" (a value that goes with, suits, works alongside or pairs with the one on that listing: the store's own pairing rule names the matching value), "differ" (anything other than, or not, the value on that listing), "at_least" (that value or above; words such as "at" before the description and "or above" after it count), "at_most" (that value or below), "near" (close to that value),
 "tolerance": "for near: how close, as written, else an empty string",
 "seller": "the shop's name alone, as the shop is called (no 's, no word shop, no word listing)",
 "kind": "the kind of product that listing is, as written",
 "words": "the shopper's own words that list that listing's properties, copied exactly from the text (for "the listing from Vexa's shop, with a, b and c" they are "a, b and c"); an empty string when none are listed",
 "attributes": ["each property and value that picks that listing out among the shop's listings, as written", ...],
 "values": ["only the value of each of those properties, without the property's name, in the same order (for 'finish matte' and 'width 30 cm': 'matte', '30 cm')", ...]}
The shop, the kind and the properties named after it describe the OTHER listing; they are never properties of the bought product."""

PAGE_READ = """You read one listing page for a shopper's pointer. The page is the listing the pointer leads to; it has been matched already, so do not judge whether it is the right listing. The shopper wants the value of a PROPERTY, to be read from this page, and a page can instead send the reader on to a further listing for that property. The PAGE RECORD shows the listing's title, description and the seller's note (its extra text; the record may call that field a note, fine print or similar) and, once it has been read, its spec sheet. Return ONE JSON object and nothing else, with its keys in this order:
{"value": "the value of the PROPERTY that this page states for itself, from its note or its spec sheet: the value alone (no article or preposition), all of it when the page lists several; or an empty string",
 "value_words": "the page's own words that state that value, or an empty string",
 "next": null, or {"seller": "the shop's name alone", "kind": "the kind of product that other listing is", "words": "the note's own words that describe that other listing, copied exactly"} when the note sends the reader on to another listing for the PROPERTY (refers to it, points at it, says to use it or take the answer from it),
 "rule": "the store's pairing rule for the PROPERTY quoted from the page, or an empty string",
 "partner": "the value that rule pairs with this page's value, or an empty string"}
A page that sends the reader on has no value of its own for the PROPERTY. A list of options or a pairing rule is not this page's own value: give a value only when the page's own words state it, otherwise leave it empty."""

ADJUDICATE = """A shopper's request points at a listing in a store by its shop, its kind and some of its properties. The candidates below are that shop's listings of that kind whose pages agree with the description, except that a figure may be written in another unit (convert before comparing: a length of 3.048 m is 10 ft). Say which candidate the description picks out. Return ONE JSON object and nothing else: {"pick": "the listing id of the one candidate that agrees with every property of the description, or an empty string when none does or more than one does", "reason": "one short sentence"}"""

PARTNER = """A store states, on a listing's page, a pairing rule between the values of a property. Given the value that page shows and the rule as quoted, say which value the rule pairs with that value. Read the rule exactly as worded: a rule that says one value goes with another pairs the first with the second, and it runs the other way round only where the rule says so too; a value that the rule pairs with itself is its own partner. Return ONE JSON object and nothing else: {"partner": "the paired value as the rule writes it", "reason": "one short sentence"}"""

PARTS = """One listing's record from an online store is shown below. Name the exact key of the record that holds each part, or null when no key holds it. Return ONE JSON object and nothing else: {"title": the key of the listing's name or title line, "overview": the key of its description or blurb (free text about the product, not a table), "seller": the key naming the shop or seller offering it (not the maker or brand), "note": the key of a remark, note, comment or fine print the seller added (free text, often empty), "kind": the key of the store category, aisle or shelf label it sits in}. Never name the same key twice; never name a key whose value is a table, a list or a number."""


STATUS = """A store tool reported something about the messages an agent may send to its shopper. Return ONE JSON object and nothing else: {"left": the number of messages the agent may still send (the allowance not yet used), or null when the result does not say}"""


LINEREVIEW = """A shopping agent has chosen one listing for each requested item. For each line below decide whether the listing's records CONTRADICT one of the item's requirements: \
the records state a value that is not acceptable for it (units converted, limits include their edge, "about" allows 12%). \
The store's category decides the kind of product: every listing below is filed under its item's category, so it IS the requested kind even when its title names a related part, form or variant; never call a line wrong for its kind. \
The agent's code has already checked any model code, so never judge the fit either. A requirement the records do not mention at all is NOT a contradiction. \
Return ONE JSON object and nothing else: {"bad": [{"id": listing id, "requirement": the requirement it contradicts, copied from its line, "why": "short reason"}]}. Use only ids from the list; [] when no line is contradicted."""


CHOOSE = """A shopping agent must replace one cart line. The item and its requirements are shown with candidate listings (cheapest first). Choose the cheapest candidate whose records \
meet EVERY requirement (units converted, limits include their edge, "about" allows 12%; a requirement that takes its value from another listing is met when the records show that value), or null when none does. \
Every candidate is filed under the item's category, which settles the kind of product, and the agent's code has checked any model code: judge the requirements only. \
Return ONE JSON object and nothing else: {"pick": a listing id from the list, or null}."""


TEXTSTATE = """A shopper's requirement is shown with some listings. Decide for each listing whether its TITLE, DESCRIPTION or ATTRIBUTES state a value that meets the requirement. Convert units where needed and let "about", "around" or "roughly" in the requirement allow a difference of about 12% ("about 4 inches" is met by "10 cm" or "9 cm"); limits include their edge; a figure that does not say what it measures counts for nothing; a listing that states a different value does not count. Return ONE JSON object and nothing else: {"states": [the ids of the listings that state it]}. Use only ids from the list."""


SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null, "price_unit": "cents" when the price values are whole minor units (cents, pence), "units" when they are whole or decimal amounts of the currency itself, null when unsure}"""


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
    # punctuation never belongs to a word ("Bearings, Bushings & Seals" -> bearing, bushing, seal)
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", _words(text)))


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


# one question naming one property for several items collided in the Race 12 probe whenever one of the other items has a subject on that property
MERGE_PROPERTIES = False
FANOUT = False              # OFF: paid Fitment, v16 with it on: hidden needs asked 4 of 12 (v14: 8 of 12), 4/10 passes (v14: 6/10). True = a model-code request
                            # asks each property of every product that carries it, in one message
FANOUT_NAMES = 2            # most products one fan-out question names (free tests: 4 or 3 names lost 12 of 211 points to decoys on another named product, 2 names lost 3)
PROCEDURES = True           # fitment and basket requests: Kyo carries the request through itself right after the interview (run_procedure)
PROC_ROUNDS = 3             # pick - cart - review rounds of that procedure
BUNDLE_EARLY = 0.4          # share of the steps that may be left when Kyo's own pick starts for a request of three items or more
# keys about the warranty, the options, the year and the like are never what a shopper holds firm on (the PLAN skips them as logistics)
PART_KEYS: dict[str, str] = {}        # the record keys the light model named for this store's title / overview / seller / note / kind (one episode)
_LOGISTICS = re.compile(r"warrant|option|year|origin|condition|shipping|certif|seller|brand|sku|part number|model number", re.I)


# the record key that holds the aisle label, found from the rows by Kyo.category_key() (stores name it differently)
_KIND_KEY: list[str] = []


def _guarded(default: Any):
    """Run an optional part of the agent so that whatever goes wrong in it costs only its help, never the episode."""
    def wrap(fn):
        def inner(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append({"kind": "guarded", "fn": fn.__name__, "error": f"{type(exc).__name__}: {exc}"[:200]})
                return default(self, *args, **kwargs) if callable(default) else default
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner
    return wrap


def _flat(text: Any) -> str:
    """Text lower-cased with every run of other characters as one space, for comparing what a page says with what a reader reports."""
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip() + " "


def _text(value: Any) -> str:
    """A scalar read as trimmed text; anything else is empty."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    return str(value).strip()


def _shop(text: str) -> str:
    """A shop's name read from a model reply, without a possessive ending or the words around it ("the X shop's listing" -> "X")."""
    name = re.sub(r"[\u2019']s\b", "", str(text or ""))
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\b(?:the|shop|store|seller|listing|ltd|inc|llc|co)\b\.?", " ", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip(" '\"`.,;:()")


def _target(value: Any) -> dict[str, Any] | None:
    """A described listing (its shop, its kind, the properties that pick it out and their bare values) read from a model reply;
    None when it names neither the shop nor the kind."""
    if not isinstance(value, dict):
        return None
    seller, kind = _shop(_text(value.get("seller"))), _text(value.get("kind"))
    attrs = [t for t in (_text(a) for a in _items(value.get("attributes"))) if t]
    values = [t for t in (_text(a) for a in _items(value.get("values"))) if t]
    return {"seller": seller, "kind": kind, "attributes": attrs, "values": values, "clause": _text(value.get("words"))} if seller or kind else None


def _reply_node(answer: Any, key: str) -> dict[str, Any] | None:
    """The dict of a message answer that carries the reply key: the answer itself or one dict value of it (a notice wrapper)."""
    if not isinstance(answer, dict):
        return None
    if key in answer:
        return answer
    return next((v for v in answer.values() if isinstance(v, dict) and key in v), None)


def _cart_modes(options: list) -> tuple[Any, Any]:
    """(the mode that puts units in, the mode that takes them off) of a cart tool's mode list, by the words of the options; the first and the
    second option when the words say nothing."""
    adds = [o for o in options if re.search(r"add|put|includ|insert|more|plus|increas|append", str(o), re.I)]
    removes = [o for o in options if re.search(r"remov|withdraw|drop|less|delet|take|out|minus|decreas|subtract|clear", str(o), re.I) and o not in adds]
    add = adds[0] if adds else next((o for o in options if o not in removes), options[0])
    remove = removes[0] if removes else next((o for o in options if o != add), options[-1])
    return add, remove


def _rows_level(observation: Any) -> dict[str, Any] | None:
    """The dict holding a list of two or more records: the observation itself, or the one dict value of it that holds such a list (a notice
    wrapper), else None."""
    if not isinstance(observation, dict):
        return None
    has_rows = lambda d: any(isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v) for v in d.values())
    if has_rows(observation):
        return observation
    inner = [v for v in observation.values() if isinstance(v, dict) and has_rows(v)]
    return inner[0] if len(inner) == 1 else None


def _fault(observation: Any) -> bool:
    """A reply that only reports a fault: a dict of at most three entries holding short text and nothing a listing or a reply carries (no list,
    no nested record, no figure), under a key that speaks of a fault, or whose text itself says the call failed."""
    if not isinstance(observation, dict) or not observation or len(observation) > 3:
        return False
    if any(isinstance(v, (list, dict)) for v in observation.values()):
        return False
    strings = [v for v in observation.values() if isinstance(v, str)]
    if any(re.search(r"fault|error|issue|oops|fail|problem|refus|reject|denied|invalid", str(k), re.I) for k in observation):
        return bool(strings) or all(isinstance(v, (str, bool, int, float, type(None))) for v in observation.values())
    return (len(observation) <= 2 and bool(strings) and all(len(s.split()) <= 25 for s in strings)
            and any(re.search(r"\b(?:not|cannot|can't|unknown|no such|invalid|unrecogni[sz]ed|refused|rejected|exceeded|too many|no more)\b", s, re.I) for s in strings))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _first_sentence(text: str) -> str:
    """The text up to the end of its first sentence."""
    text = text.strip()
    found = re.search(r"(?<=[.!?])\s|\n", text)
    return text[: found.start()] if found else text


_CODE_SPLIT = re.compile(r"^([A-Za-z]+)[\s\-]*(\d+[A-Za-z0-9]*)$")
_FAMILY = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,})[\s\-]?(\d+)(x+)(?![A-Za-z0-9])", re.I)
_EXCEPT = re.compile(r"\bexcept(?:ing)?\b|\bbut not\b|\bapart from\b", re.I)


def _code_re(code: str) -> str:
    """A pattern for a model code as the texts write it: "BEX3520", "BEX 3520" and "BEX-3520" are one code."""
    m = _CODE_SPLIT.match(str(code).strip())
    if not m:
        return re.escape(str(code))
    return re.escape(m.group(1)) + r"[\s\-]?" + re.escape(m.group(2))


def _names_code(text: str, code: str) -> bool:
    """Whether the text names the model code: written out (the exceptions of a sentence do not count) or covered by a family pattern the
    text writes itself ("every BEX 35xx model" covers BEX 3520 and not BEX 7820; an exception that names the code takes it out)."""
    if not code:
        return False
    parts = _EXCEPT.split(str(text), maxsplit=1)
    head, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    flat = r"(?<![A-Za-z0-9])" + _code_re(code) + r"(?![A-Za-z0-9])"
    if re.search(flat, head, re.I):
        return True
    m = _CODE_SPLIT.match(str(code).strip())
    if not m or re.search(flat, rest, re.I):
        return False
    letters, digits = m.group(1).lower(), m.group(2)
    for fam in _FAMILY.finditer(head):
        if fam.group(1).lower() == letters and len(digits) == len(fam.group(2)) + len(fam.group(3)) and digits.startswith(fam.group(2)):
            return True
    return False


def _fit_state(record: Any, code: str) -> tuple[str, str]:
    """How a listing record speaks of a model code: ("names", the sentence) when the first sentence of one of its texts names it,
    ("later", "") when it comes up only after that, ("no", "") when its texts never mention it, ("unknown", "") when it has no text."""
    if not isinstance(record, dict) or not code:
        return "unknown", ""
    texts = [v.strip() for v in record.values() if isinstance(v, str) and len(v.strip()) >= 12]
    if not texts:
        return "unknown", ""
    later = False
    for text in texts:
        first = _first_sentence(text)
        if _names_code(first, code):
            return "names", first[:90]
        if _names_code(text, code):
            later = True
    return ("later", "") if later else ("no", "")


# ---- reference matcher: begin ----
STOP = frozenset("""a an the of and or with in on for to from is are was were it its this that these those not at as by be has have had
comes come made sold sells offers offer here there which who whose than rather instead item items product products listing listings version
another other different only just must exactly""".split())
def squash(text: str) -> str:
    """Lower case, '&' as 'and', separators to spaces, and a figure glued to the short word after it ('16 gb' -> '16gb') so that both
    spellings of a quantity read the same."""
    t = str(text or "").lower().replace("&", " and ")
    t = re.sub(r"\b(two|three|four|five|six|seven|eight|nine|ten)\b", lambda m: str(["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].index(m.group(1)) + 2), t)
    t = re.sub(r"[_/]", " ", t)
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    # a figure and the short word after it read as one ("16 gb" -> "16gb"), unless that word is a plain function word ("1 and")
    return re.sub(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*-?\s*([a-z]{1,6})(?![a-z])", lambda m: m.group(0) if m.group(2) in STOP else m.group(1) + m.group(2), t)


_FIG = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$")


def quantities(text: str) -> list[tuple[float, str]]:
    """(figure, unit word) for each figure written with a unit in the text; units are compared as written, never converted."""
    out = []
    for w in re.findall(r"\d+(?:\.\d+)?[a-z]+", squash(text)):
        m = _FIG.match(w)
        if m:
            out.append((float(m.group(1)), stem(m.group(2))))
    return out


def same_qty(a: tuple[float, str], b: tuple[float, str]) -> bool:
    return a[1] == b[1] and abs(a[0] - b[0]) <= 0.012 * max(abs(a[0]), abs(b[0]), 1e-9)


def other_unit(a: tuple[float, str], b: tuple[float, str]) -> bool:
    """Two quantities written in different units: code cannot say whether they are the same quantity."""
    return a[1] != b[1]


def _spell(w: str) -> str:
    """British -> American by rule: colour -> color, litre -> liter, centre -> center."""
    if len(w) > 4 and w.endswith("our"):
        return w[:-2] + "r"
    if len(w) > 4 and w.endswith("tre"):
        return w[:-2] + "er"
    return w


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def words(text: str, drop_stop: bool = True) -> list[str]:
    t = squash(text)
    out = []
    for w in re.findall(r"\d+(?:\.\d+)?[a-z]*|[a-z][a-z0-9]*", t):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]+)", w)
        if drop_stop and w in STOP:
            continue
        out.append(m.group(1) + stem(m.group(2)) if m else (_spell(stem(w)) if not w[0].isdigit() else w))
    return [w for w in out if (w not in STOP or not drop_stop)]


# ---------------------------------------------------------------- notes: sentences, negated spans, pointer sentence
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(])")
_NEG = re.compile(r"\b(?:is|are|was|were|does|do)?\s*(?:not|n't|isn't|aren't|never)\b|\bunlike\b|\bother than\b|\bapart from\b|\bexcept\b", re.I)
# a negation that only says "take the value from another listing" is the pointer, not a negated fact
_NEG_POINTER = re.compile(r"^(?:not|n't)?\s*(?:from|by|to|at|in)?\s*(?:this|these|that|those|the same)\b", re.I)
_NEG_END = re.compile(r"(?:,\s*(?:and|but|so)\b|;|\.\s|\.$|\bbut\b|\bwhile\b)", re.I)
POINT_CUE = re.compile(r"\b(?:look(?:s|ing)?\s+(?:to|at)|go(?:es)?\s+by|refer(?:s)?\s+to|taken?\s+(?:it\s+|that\s+|the\s+\w+\s+|this\s+|them\s+)?from|(?:read|taken|comes?|come)\s+from|use\b|see\b|check\b|take\b|get\b|answer\s+from|from\s+the)\b", re.I)


def sentences(note: str) -> list[str]:
    return [s.strip() for s in _SENT.split(str(note or "").strip()) if s.strip()]


_ELSEWHERE = re.compile(r"\belsewhere\b|\b(?:version|variant|edition|model|one)s?\b[^.]{0,40}\b(?:exists?|is available|are available|can be found|is sold|are sold)\b", re.I)


def negated_text(note: str) -> tuple[str, str]:
    """(positive text, negated text) of a note: a clause after 'not' / 'is not the ... version' is negated, except the
    pointer tail 'not from this item'. A pointer sentence's own tail is dropped from both."""
    pos: list[str] = []
    neg: list[str] = []
    for s in sentences(note):
        start = 0
        kept: list[str] = []
        if _ELSEWHERE.search(s) and not POINT_CUE.search(s):
            neg.append(s)       # "A dimmable version exists elsewhere": what it describes is some other listing
            continue
        for m in _NEG.finditer(s):
            if m.start() < start:
                continue
            rest = s[m.end():].lstrip(" ,")
            if _NEG_POINTER.match(rest) or re.match(r"(?:the answer|from|by|to)\b", rest, re.I) and re.search(r"\bthis\b|\bthese\b|\bthem\b", rest[:40], re.I):
                kept.append(s[start:m.start()])
                start = len(s)
                break
            end = _NEG_END.search(rest)
            span = rest[:end.start()] if end else rest
            kept.append(s[start:m.start()])
            neg.append(span)
            start = m.end() + (end.start() if end else len(rest)) + (len(s[m.end():]) - len(rest))
        kept.append(s[start:] if start < len(s) else "")
        pos.append(" ".join(kept))
    return " ".join(pos), " ".join(neg)


def pointer_sentence(note: str, prop: str) -> str:
    """The sentence of a note that sends the reader to another listing for the property: it names the property (most of its
    content words) and carries a pointing cue; '' when the note has none."""
    pw = [w for w in words(prop) if w not in ("type",)] or words(prop)
    best = ""
    for s in sentences(note):
        sw = set(words(s))
        if pw and sum(1 for w in pw if w in sw) < max(1, int(0.6 * len(pw) + 0.99)):
            continue
        if POINT_CUE.search(s) or re.search(r"\bnot\b[^.]*\b(?:this|these|that)\b", s, re.I) or re.search(r"\b(?:instead|rather than)\b", s, re.I):
            return s
    return best


# ---------------------------------------------------------------- descriptor fragments and page matching
_LEAD = re.compile(r"^(?:and|or|but|with|which|that|who|is|are|has|have|having|comes?|made|sold|listed|in|of|for|by|a|an|the|from|another|other|different|product|products|listing|listings|item|items)\b\s*", re.I)


def clean_clause(clause: str) -> str:
    c = str(clause or "").strip(" .;:,-")
    c = re.sub(r"\((?:in|measured in)\s+[a-z]+\)", " ", c, flags=re.I)        # "(in gb)" is a unit hint, not a property
    c = re.sub(r"\s+", " ", c).strip()
    return c


def fragments(clause: str, kind_words: frozenset[str] = frozenset()) -> list[dict]:
    """Split a described listing's properties ('light effects flashing, length 2 m and light bulb colour clear') into fragments:
    content stems (the kind's own words left out) and quantities."""
    c = clean_clause(clause)
    if not c:
        return []
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in c:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and ch in ",;":
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out: list[str] = []
    for p in parts:
        out += re.split(r"\s+(?:and|plus|&)\s+(?=[\w(])", p)
    frags: list[dict] = []
    for p in out:
        p = p.strip()
        while True:
            q = _LEAD.sub("", p)
            if q == p:
                break
            p = q
        ws = [w for w in words(p) if w not in kind_words]
        qs = quantities(p)
        if ws or qs:
            last = max([i for i, w in enumerate(ws) if w in GENERIC] or [-1])
            val = ws[last + 1:] if 0 <= last < len(ws) - 1 else ([w for w in ws if w not in GENERIC] if last >= 0 else ws)
            if last < 0:
                # no generic property word: an attribute name the store's own listings carry ("occasion", "closure type") ends the property part
                for ph in KEY_PHRASES:
                    if len(ws) > len(ph) and tuple(ws[:len(ph)]) == ph:
                        val = ws[len(ph):]
                        break
            frags.append({"text": p, "stems": ws, "qty": qs, "val": val})
    return frags


def bag(text: str) -> tuple[set[str], list[tuple[float, str]]]:
    """The words of a text, a figure written with its unit also as the bare figure, and its quantities."""
    ws = words(text, drop_stop=False)
    out = set(ws)
    for w in ws:
        m = _UNITISH.match(w)
        if m:
            out.add(m.group(1))
    return out, quantities(text)


KEY_PHRASES: list[tuple[str, ...]] = []      # attribute names seen on listings (longest first), set by the agent
GENERIC = frozenset("""size type pack material colour color capacity weight length height width depth condition position make model style form finish
pattern shape count number amount quantity kind connectivity source""".split())
_UNITISH = re.compile(r"^(\d+(?:\.\d+)?)[a-z]+$")


_SUFFIXES = frozenset(["s", "t", "ts", "ed", "er", "ing", "ly", "al", "ion", "ness", "ment", "ful", "es"])


def _has(w: str, stems: set[str]) -> bool:
    """A word is on the page, or a longer / shorter form of it is ('thick' for 'thickness')."""
    if w in stems:
        return True
    if len(w) >= 5 and not w[0].isdigit():
        for t in stems:
            if t == w or t[:1].isdigit() or len(t) < 4 or abs(len(t) - len(w)) > 5:
                continue
            short, long_ = (t, w) if len(t) < len(w) else (w, t)
            if len(short) >= 4 and long_.startswith(short) and long_[len(short):] in _SUFFIXES:
                return True
    return False


def _fig_on_page(w: str, stems: set[str]) -> str:
    """For a word that is a figure with a unit ('3.048m'): 'same' when the page writes that quantity in that unit (rounding allowed),
    'other' when the page has figures in other units only, '' when it has no figure at all."""
    m = _UNITISH.match(w)
    if not m:
        return ""
    mine = quantities(w)
    if not mine:
        return ""
    theirs = [q for t in stems if _UNITISH.match(t) for q in quantities(t)]
    if any(same_qty(mine[0], q) for q in theirs):
        return "same"
    if any(not other_unit(mine[0], q) for q in theirs):
        return ""                 # the page gives that unit with another figure: a different quantity
    return "other" if theirs else ""


def frag_level(frag: dict, b: tuple[set[str], list[tuple[float, str]]]) -> int:
    """2 = every word of the fragment is on the page (a quantity within rounding); 1 = the page leaves the property's own name out
    but states the value (the fragment's value words); 0 = not shown; -1 = shown, apart from a quantity the page writes in another
    unit (code cannot compare those: a reader may)."""
    stems, qs = b
    need = list(frag["stems"])
    if not need:
        return 0
    present = [(w, _has(w, stems) or _fig_on_page(w, stems) == "same") for w in need]
    if all(ok for _, ok in present):
        return 2
    missing = [w for w, ok in present if not ok]
    # a bare figure under the property's name ("display size: 2" for "2 inches"): the figure is there, the unit is left out
    bare = [w for w in missing if _UNITISH.match(w) and _UNITISH.match(w).group(1) in stems]
    if bare and len(bare) == len(missing) and any(_has(w, stems) for w in need if not _UNITISH.match(w)):
        return 1
    others = [w for w in missing if _fig_on_page(w, stems) == "other"]
    if others and len(others) == len(missing):
        return -1
    val = frag.get("val") or need
    if val and len(val) < len(need) and all(_has(w, stems) or _fig_on_page(w, stems) == "same" for w in val):
        return 1
    if val and len(val) < len(need) and all(_has(w, stems) or _fig_on_page(w, stems) in ("same", "other") for w in val) and any(_fig_on_page(w, stems) == "other" for w in val):
        return -1
    return 0


def frag_in(frag: dict, b: tuple[set[str], list[tuple[float, str]]]) -> bool:
    return frag_level(frag, b) == 2


def fact_pairs(facts: dict) -> list[tuple[list[str], str]]:
    out = []
    for k, v in (facts or {}).items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            out.append((words(str(k).replace("_", " ")), str(v)))
        elif isinstance(v, list):
            out.append((words(str(k).replace("_", " ")), ", ".join(str(x) for x in v)))
    return out


def frag_conflict(frag: dict, pairs: list[tuple[list[str], str]]) -> bool:
    """The spec sheet states another value under the key this fragment names: the key's words open the fragment, and every such key's
    value shares nothing with what else the fragment says (a figure in the same unit with another figure: a different quantity; a
    figure in another unit is no conflict - code cannot compare those). One key that supports the fragment clears it (sheets repeat a
    key in two spellings)."""
    fs = list(frag["stems"])
    against = supports = 0
    for key, value in pairs:
        if not key or key == ["type"] or fs[:len(key)] != key:
            continue
        rest = set(fs[len(key):]) - GENERIC
        if not rest:
            continue
        vb = bag(value)
        figs = [w for w in rest if _UNITISH.match(w)]
        if figs:
            if any(_fig_on_page(w, vb[0]) == "same" for w in figs):
                supports += 1
                continue
            mine = [quantities(w)[0] for w in figs if quantities(w)]
            theirs = quantities(value)
            if mine and theirs and any(not other_unit(m1, t1) for m1 in mine for t1 in theirs):
                against += 1
            continue
        if vb[0] and not (rest & vb[0]):
            against += 1
        elif vb[0]:
            supports += 1
    return against > 0 and supports == 0


def answers(prop: str, note: str, facts: dict | None) -> bool:
    """The page speaks to the wanted property: its note names it or its spec sheet has a key for it."""
    pw = [w for w in words(prop) if w != "type"] or words(prop)
    if not pw:
        return False
    nw = set(words(note))
    if note and sum(1 for w in pw if w in nw) >= max(1, int(0.6 * len(pw) + 0.99)):
        return True
    for key, _ in fact_pairs(facts or {}):
        kk = [w for w in key if w != "type"] or key
        if kk and (set(pw) <= set(kk) or set(kk) <= set(pw)) and len(set(pw) & set(kk)) >= max(1, min(len(pw), len(kk)) - 0):
            return True
    return False


_NEGATOR = re.compile(r"^\s*(?:not|no|non|without)\b", re.I)
_NEG_WORD = re.compile(r"\b(?:no|not|non|without)\b", re.I)


def judge(frags: list[dict], note: str = "", facts: dict | None = None, title: str = "", overview: str = "", prop: str = "", kind: str = "") -> dict:
    """Score one listing against the described properties. Each fragment is: stated by the note (positive part), by the spec sheet,
    stated with the property's name left out (soft), by the title only, or not shown; or it is ruled out: the note negates it or
    the spec sheet states another value for it. The sentence of the note that sends the reader to another listing describes that
    listing and is left out; a spec value that begins with a negation ("not waterproof") never confirms the plain property."""
    own = pointer_sentence(note, prop) if prop and note else ""
    pos, neg = negated_text(note.replace(own, " ") if own else note)
    items = list((facts or {}).items())
    plain = [(k, v) for k, v in items if not _NEGATOR.match(str(v))]
    negated = [(k, v) for k, v in items if _NEGATOR.match(str(v))]
    join = lambda pairs: " ; ".join(f"{k} {v}" for k, v in pairs)
    nb, tb = bag(pos), bag(title + " " + (overview or "")[:300])
    fb_all, fb_plain, fb_neg = bag(join(items)), bag(join(plain)), bag(" ; ".join(str(v) for _, v in negated))
    negb = bag(neg)
    ub_all, ub_plain = bag(pos + " ; " + join(items) + " ; " + kind), bag(pos + " ; " + join(plain) + " ; " + kind)
    pairs = fact_pairs(dict(items))
    got = {"note": 0, "facts": 0, "soft": 0, "title": 0, "no": 0, "maybe": 0}
    detail = []
    for f in frags:
        negative = bool(_NEG_WORD.search(f["text"]))
        fb, ub = (fb_all, ub_all) if negative else (fb_plain, ub_plain)
        if frag_level(f, nb) == 2:
            got["note"] += 1; detail.append("note"); continue
        if neg and (any((w in negb[0] and w not in nb[0]) for w in f["stems"] if len(w) > 1 and w not in GENERIC)
                    or (f["qty"] and any(any(same_qty(q, p) for p in negb[1]) for q in f["qty"]))):
            got["no"] += 1; detail.append("NEG"); continue
        if facts and frag_conflict(f, pairs):
            got["no"] += 1; detail.append("CONFLICT"); continue
        val = [w for w in (f.get("val") or f["stems"]) if w not in GENERIC]
        if facts and not negative and negated and val and all(w in fb_neg[0] for w in val) and not all(w in fb_plain[0] for w in val):
            got["no"] += 1; detail.append("CONFLICT"); continue
        if facts and frag_level(f, fb) == 2:
            got["facts"] += 1; detail.append("facts"); continue
        if frag_level(f, ub) == 2:
            got["facts"] += 1; detail.append("union"); continue
        if max(frag_level(f, nb), frag_level(f, fb) if facts else 0) == 1:
            got["soft"] += 1; detail.append("soft"); continue
        if frag_level(f, tb) >= 1:
            got["title"] += 1; detail.append("title")
        elif max(frag_level(f, nb), frag_level(f, fb) if facts else -9, frag_level(f, ub)) == -1:
            got["maybe"] += 1; detail.append("maybe")
        else:
            detail.append("-")
    n = len(frags)
    sure = got["note"] + got["facts"] + got["soft"]
    ans = answers(prop, note, facts) if prop else False
    return {"n": n, **got, "unseen": n - sure - got["title"] - got["no"] - got["maybe"], "has_note": bool(note), "answers": ans, "detail": detail,
            "full": n > 0 and sure == n, "maybe_full": n > 0 and got["maybe"] > 0 and sure + got["maybe"] == n,
            "key": (got["no"] == 0, n > 0 and sure == n, n > 0 and sure + got["maybe"] == n, ans, got["note"] + got["facts"], sure, got["title"], bool(note))}


# ---------------------------------------------------------------- the pointer: who is the next listing, what describes it
_TAIL = re.compile(r"(?:[,;]\s*)?(?:and\s+|so\s+|as\s+)?(?:not\b|rather\s+than\b|instead\b|take\s+that\b|as\s+nothing\b|nothing\s+here\b)", re.I)
_CAP = re.compile(r"\b([A-Z][a-z]{3,}(?:[a-z]{0,})?)(?:['\u2019]s)?\b")
FUNC_CAPS = frozenset("""The This That These Those They Its And But With From Here There Also Other Another Different For Take Instead Rather Not Look Check Refer
Use See Get Read Comes Come Sold Listed Made Product Listing Item Items One Pair""".split())
_KIND_END = re.compile(r"\s*(?:[,;:(]|\b(?:with|which|that|who|in|for|made|listed|sold|from|at|is|are|has|have|having|and|of|on|comes?|come|as|where)\b)", re.I)
_PRE_JUNK = re.compile(r"^(?:the|a|an|another|other|different|product|products|listing|listings|item|items|from|by|of|to|at|,|:)\b[\s,:]*", re.I)
_POST_TAIL = re.compile(r"\b(?:product|products|listing|listings|item|items)\b\s*$", re.I)


def pointer_target(sentence: str, seller_hint: str = "", own_seller: str = "", skip: frozenset = frozenset(), sellers: frozenset = frozenset()) -> dict:
    """From a note's pointer sentence: the shop of the next listing, its kind (the words naming what it is) and the words that
    describe it (its properties). `skip` = lower-case words that are no shop (the aisle names, the property); `sellers` = shops seen."""
    s = str(sentence or "").strip()
    cue = POINT_CUE.search(s)
    after = s[cue.end():] if cue else s
    after = re.sub(r"^[\s,:-]+", "", after)
    seller = ""
    at = -1
    hint = (seller_hint or "").strip()
    m = re.search(r"\b" + re.escape(hint) + r"\b", after, re.I) if hint else None
    if m:
        seller, at = hint, m.start()
    else:
        picks = []
        low_sellers = {t.lower() for t in sellers}
        for m in _CAP.finditer(after):
            w = m.group(1)
            if w in FUNC_CAPS or w.lower() in skip or stem(w.lower()) in skip or (own_seller and w.lower() == own_seller.lower()):
                continue
            # a shop's name: a known shop; after "from" / "by"; before 's; not the first of a run of capitalised words (a kind: "Fairy Lights")
            score = (3 if w.lower() in low_sellers else 0) + (2 if re.search(r"\b(?:from|by)\s+$", after[:m.start()], re.I) else 0) \
                + (2 if after[m.end():m.end() + 2] in ("'s", "\u2019s") or after[m.end() - 2:m.end()] in ("'s", "\u2019s") else 0) \
                - (2 if re.match(r"\s+[A-Z][a-z]", after[m.end():]) else 0)
            picks.append((w, m.start(), score))
        if picks:
            best = max(picks, key=lambda x: (x[2], -x[1]))
            seller, at = best[0], best[1]
        elif own_seller and re.search(r"\b(?:other|another|same shop)\b", after, re.I):
            seller = own_seller
            m = re.search(r"\b" + re.escape(own_seller) + r"\b", after, re.I)
            at = m.start() if m else -1
    if not seller:
        return {"seller": "", "kind": "", "clause": clean_clause(after), "sentence": s}
    # the tail ("not from this item") is cut where it stands after the shop's name
    tail = next((t for t in _TAIL.finditer(after) if t.start() >= max(at, 0) + len(seller)), None)
    body = after[:tail.start()] if tail else after
    if at >= 0:
        pre = body[:at]
        post = re.sub(r"^(?:['\u2019]s)?", "", body[at + len(seller):])
    else:
        pre, post = "", body
    while True:
        q = _PRE_JUNK.sub("", pre.strip())
        if q == pre.strip():
            break
        pre = q
    pre = re.sub(r"\s*\b(?:from|by|of)\b\s*$", "", pre.strip(), flags=re.I).strip(" ,:")
    kind = pre
    if not kind:
        km = _KIND_END.search(post.lstrip())
        lead = post.lstrip()[:km.start()] if km else post.lstrip()
        kind = lead.strip()
        post = post.lstrip()[len(lead):] if km else ""
    kind = _POST_TAIL.sub("", kind).strip()
    return {"seller": seller, "kind": kind, "clause": clean_clause(post), "sentence": s}


def unparen(clause: str) -> str:
    """Parentheses that list properties become separators; short unit hints like '(3-5cm)' stay part of their words."""
    def fix(m: re.Match) -> str:
        inner = m.group(1)
        if "," in inner or len(re.findall(r"[A-Za-z]{3,}", inner)) >= 3:
            return ", " + inner + ", "
        return " " + inner + " "
    return re.sub(r"\(([^()]*)\)", fix, str(clause or ""))


def _word_pat(w: str) -> str:
    """A word of an aisle label, whatever its number: 'Mattresses' matches 'mattress' and 'mattresses'."""
    base = w[:-2] if len(w) > 4 and w.lower().endswith("es") and w[-3].lower() in "sxz" else (w[:-1] if len(w) > 3 and w.lower().endswith("s") else w)
    return re.escape(base) + r"(?:e?s)?"


def strip_kind(clause: str, labels: list[str]) -> tuple[str, str]:
    """Take the kind's own name (an aisle label the store uses) out of a descriptor clause; returns (clause without it, the label)."""
    found = ""
    c = " " + str(clause or "") + " "
    for lab in sorted({l for l in labels if l}, key=lambda x: -len(x)):
        parts = [p.strip() for p in re.split(r"\s*&\s*|\s+and\s+", lab) if p.strip()]
        sep = r"\s*(?:&|and)\s*"
        pat = re.compile(r"(?<![A-Za-z])" + sep.join(r"\s+".join(_word_pat(w) for w in p.split()) for p in parts) + r"(?![A-Za-z])", re.I)
        if pat.search(c):
            c = pat.sub(" ", c, count=1)
            found = found or lab
    return re.sub(r"\s+", " ", c).strip(), found


def descriptor(clause: str, labels: list[str], extra_kind: str = "") -> tuple[list[dict], str]:
    c, kind = strip_kind(unparen(clause), list(labels) + ([extra_kind] if extra_kind else []))
    kw = frozenset(words(kind or extra_kind))
    frags = fragments(c, kw)
    # a fragment that is only a stray listing word of the pointer sentence says nothing about the page
    return [f for f in frags if not (set(f["stems"]) <= {"product", "listing", "item", "from", "by"})], kind


def fact_value(facts: dict | None, prop: str) -> str:
    """The value a spec sheet states for a property: the fact whose key names it (the key's words are the property's words, any order,
    'type' aside); the nearest key wins, '' when none or when two keys disagree."""
    pw = {w for w in words(prop.replace("_", " ")) if w != "type"} or set(words(prop))
    if not pw:
        return ""
    best: list[tuple[int, str]] = []
    for key, value in fact_pairs(facts or {}):
        kw = {w for w in key if w != "type"} or set(key)
        if not kw or not str(value).strip():
            continue
        if kw == pw:
            best.append((0, str(value).strip()))
        elif pw <= kw and len(kw) - len(pw) <= 1 and (len(pw) >= 2 or (kw - pw) <= {"family", "group", "name", "value"}):
            best.append((1, str(value).strip()))
    if not best:
        return ""
    top = min(b[0] for b in best)
    vals = []
    for r, v in best:
        if r == top and v.lower() not in [x.lower() for x in vals]:
            vals.append(v)
    return vals[0] if len(vals) == 1 else (vals[0] if all(words(v) == words(vals[0]) for v in vals) else "")


_RULE = re.compile(r"\b(?:goes?|go|pairs?|paired|pairing|pairings|matches|match|matched|suits?|works?)\s+(?:together\s+)?(?:with|alongside)\b|\bpairings?\b|\brule\b", re.I)


def store_rule(note: str, prop: str = "") -> str:
    """The store's pairing rule on a page, quoted: the sentence(s) of the note that tell which value goes with which."""
    out = []
    for s in sentences(note):
        if _RULE.search(s) and not POINT_CUE.search(s.split(":")[0] if ":" not in s[:3] else s):
            out.append(s)
    return " ".join(out)


# ---------------------------------------------------------------- a requirement checked against a listing's own attributes
KNOWN_REFS: set[str] = set()          # requirement texts the brief or the shopper marked as pointers at another listing (one episode)


def _is_ref(text: str) -> bool:
    """Whether a requirement text points at another listing: marked as one by the brief's lookups or the shopper's answer, or worded like one."""
    return text in KNOWN_REFS or _REF_WORDS.search(text) is not None


_REF_WORDS = re.compile(r"\b(?:same as|the very same|something compatible|goes with|pairing rule|that listing|page of|wherever|the value the store|listing it points)\b", re.I)
_REL_LEAD = re.compile(r"^(at least|at most|no more than|no less than|up to|anything but|none of|any but|not|other than|more than|less than)\b[:\s]*", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def split_must(must: str) -> tuple[str, str]:
    head, sep, rest = str(must).partition(":")
    return (head.strip(), rest.strip()) if sep else ("", head.strip())


_COUNTING = r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"


def _counted_unit(request: str, must: str) -> bool:
    """A requirement whose value is the unit the request counts the product in ("three pairs of sneakers" -> "pack type: pair")."""
    _, cons = split_must(must)
    value = _words(cons)
    if not value or len(value.split()) > 2:
        return False
    unit = re.escape(value.split()[-1].rstrip("s"))
    return re.search(r"\b" + _COUNTING + r"\s+(?:\w+\s+)?" + unit + r"s?\s+of\b", request, re.I) is not None


def _contains(cons_words: list[str], value: str) -> bool:
    vb = set(words(value, drop_stop=False))
    return bool(cons_words) and all(_has(w, vb) for w in cons_words)


def _figures(cons: str, value: str) -> str:
    """'pass' / 'fail' / '' for a requirement that carries a figure: the same figure in the same unit (as written), or bare figures."""
    cq, vq = quantities(cons), quantities(value)
    if cq and vq:
        pairs = [(c, v) for c in cq for v in vq if c[1] == v[1]]
        if pairs:
            return "pass" if any(same_qty(c, v) for c, v in pairs) else "fail"
        return ""                         # other units: not compared by code
    nc, nv = _NUM.findall(cons), _NUM.findall(value)
    if nc and nv:
        return "pass" if set(nc) <= set(nv) else "fail"
    return ""


def _one_value(cons: str, value: str, rel: str, skip: frozenset = frozenset()) -> str:
    """'pass' / 'fail' / '' for one attribute value against one requirement. `skip`: the property's own words (a requirement "pack of 3" on
    "pack size" is met by "3")."""
    if rel in ("at least", "no less than", "more than", "at most", "no more than", "up to", "less than"):
        cq, vq = quantities(cons), quantities(value)
        cn, vn = (cq[0][0] if cq else None), (vq[0][0] if vq else None)
        if cq and vq and cq[0][1] != vq[0][1]:
            return ""
        if cq and not vq and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(value)):
            vn = float(str(value).strip())
        if cn is None:
            nc, nv = _NUM.findall(cons), _NUM.findall(value)
            if not nc or not nv:
                return ""
            cn, vn = float(nc[0]), float(nv[0])
        if vn is None:
            return ""
        ok = vn >= cn - 1e-9 if rel in ("at least", "no less than", "more than") else vn <= cn + 1e-9
        return "pass" if ok else "fail"
    elems = [e for e in re.split(r",|&|/|;|\+|\band\b", str(value)) if e.strip()] or [str(value)]
    # a requirement written as a closed list ("wifi, wi-fi", "a or b") is met by any one of its entries
    alts = [a for a in re.split(r",|;|/|\bor\b", cons) if a.strip()] if not _NUM.search(cons) else [cons]
    cw = words(cons)
    if rel in ("anything but", "none of", "any but", "not", "other than"):
        hit = any(_contains(words(a), e) for a in alts for e in elems) or any(_contains(words(a), value) for a in alts)
        return "fail" if hit else "pass"
    if _NUM.search(cons):
        states = [_figures(cons, e) for e in elems]
        bare = not quantities(cons)               # figures written without a unit: a version, a count, a bucket label
        extra = [w for w in cw if not _NUM.fullmatch(w) and w not in skip]
        if "pass" in states:
            # "5000 and up" is a bucket of its own, not the bare "5000"
            if bare and extra and not _contains(extra, value):
                return "fail"
            return "pass"
        if "fail" in states and "" not in states:
            return "fail"
        if "pass" not in states and bare and not _NUM.search(value) and any(_has(w, set(words(value, drop_stop=False))) for w in extra):
            return "fail"                         # "windows" for "windows 10": the value leaves out the figure
        return "pass" if _contains(cw, value) else ""
    return "pass" if any(_contains(words(a), e) or _contains(words(a), value) for a in alts for e in elems) else "fail"


def verify_must(must: str, facts: dict | None, title: str = "", overview: str = "", kind: str = "") -> tuple[str, str]:
    """A requirement ("property: value") against a listing's own attributes, by the words: ('pass' | 'fail' | 'unknown', why).
    A key that names the property (its words contain the property's words) decides; of two such keys one that states the value is enough;
    with no such key a key named by the property's last word may confirm but never rule out; with no key at all the title and description
    may confirm. Figures are compared in the unit as written; a reference that is not yet resolved is left alone."""
    prop, cons = split_must(must)
    if not prop or not cons or _is_ref(must) or _is_ref(cons) or _is_ref(prop):
        return "unknown", ""
    rel = ""
    m = _REL_LEAD.match(cons)
    if m:
        rel = m.group(1).lower()
        cons = cons[m.end():].strip()
        if not cons:
            return "unknown", ""
    cons = re.sub(r"\(.*?\)", " ", cons).strip() or cons
    # the property's own words (only plain function words dropped: "meat products" is not "meat")
    pw = [w for w in words(prop, drop_stop=False) if w not in ("a", "an", "the", "of", "and", "or", "for", "with", "in", "on", "to", "by")]
    if not pw:
        return "unknown", ""
    pairs = fact_pairs(facts or {})
    named = [(k, v) for k, v in pairs if set(pw) <= set(k)]
    head = [(k, v) for k, v in pairs if len(k) == 1 and k[0] == pw[-1]] if not named else []
    for group, final in ((named, True), (head, False)):
        if not group:
            continue
        states = [(k, v, _one_value(cons, v, rel, frozenset(pw))) for k, v in group]
        if any(s == "pass" for _, _, s in states) and rel not in ("anything but", "none of", "any but", "not", "other than"):
            return "pass", ""
        if rel in ("anything but", "none of", "any but", "not", "other than"):
            bad = [(k, v) for k, v, s in states if s == "fail"]
            if bad and final:
                return "fail", f"{' '.join(bad[0][0])} says '{bad[0][1][:40]}', which is what it must not be"
            return "pass", ""
        if final and all(s == "fail" for _, _, s in states):
            k, v, _ = states[0]
            return "fail", f"the attribute '{' '.join(k)}' says '{str(v)[:40]}', not '{cons[:40]}'"
        return "unknown", ""
    text = " ; ".join([title or "", (overview or "")[:300], kind or ""])
    if rel in ("", "at least", "at most") and _contains(words(cons), text) and not _NUM.search(cons):
        return "pass", ""
    return "unknown", ""
# ---- reference matcher: end ----


_KIND_PROPS = frozenset(["type", "kind", "category", "product"])


def _echoes_name(must: str, name: str, shelf: str) -> bool:
    """A requirement that restates only part of the product's own name or aisle label as its "type" ("type: natural" for natural sweeteners):
    a reading mistake, since the item is already that kind. The whole name as the type ("type: tapered roller bearing") is a requirement."""
    prop, cons = split_must(must)
    own_words = set(words(name, drop_stop=False)) | set(words(shelf, drop_stop=False))
    if not prop or not set(words(prop, drop_stop=False)) <= (_KIND_PROPS | own_words):
        return False
    if not set(words(prop, drop_stop=False)) & _KIND_PROPS:
        return False
    value, whole = words(cons), words(name)
    if not value or not whole:
        return False
    return set(value) <= set(whole) | set(words(shelf)) and not set(whole) <= set(value)


_CUE_PARTNER = re.compile(r"\b(?:compatible|alongside|goes|go|works?|pairs?|suits?|complements?|fits?)\b", re.I)
_CUE_DIFFER = re.compile(r"\b(?:different|unlike|other|differs?|except|not)\b", re.I)
_CUE_ABOVE = re.compile(r"\b(?:or\s+(?:above|more|higher|greater|over)|at\s+least|no\s+less)\b", re.I)
_CUE_BELOW = re.compile(r"\b(?:or\s+(?:below|less|lower|under|fewer)|at\s+most|no\s+more|up\s+to)\b", re.I)


def _relation_cue(need: str) -> str:
    """The comparison a requirement's own wording makes with the other listing's value ("compatible with ...", "different from ...", "... or above"),
    read from the words before the description of that listing; "" when the wording shows none of them."""
    text = need.split(":", 1)[-1]
    head = re.split(r"\b(?:could\s+you|the\s+one\s+on|the\s+page|the\s+listing|listing)\b", text, maxsplit=1, flags=re.I)[0]
    if _CUE_ABOVE.search(text):
        return "at_least"
    if _CUE_BELOW.search(text):
        return "at_most"
    if _CUE_PARTNER.search(head):
        return "partner"
    if _CUE_DIFFER.search(head):
        return "differ"
    return ""


def _reference_echoes(request: str, musts: list[str]) -> list[str]:
    """Drop requirements the brief took from the description of a referenced listing ("... Savexshibel's listing, with condition new and megapixels 1
    (or wherever that page leads)"): those attributes pick out the other listing. A requirement whose property the request names only inside such
    a description is one of them; the requirement that carries the reference itself stays."""
    descs = re.findall(r"listing,?\s*with\s+(.+?)\s*\(or wherever", request, flags=re.I | re.S)
    if not descs:
        return musts
    rest = request
    for d in descs:
        rest = rest.replace(d, " ")
    rest_words = set(words(rest, drop_stop=False))
    desc_words = set(words(" ".join(descs), drop_stop=False))
    keep = []
    for must in musts:
        prop, cons = split_must(must)
        pw = set(words(prop, drop_stop=False))
        vw = set(words(cons, drop_stop=False))
        # the property AND the value belong to the description only ("make: toyota" stays when the request itself says "for a Toyota")
        if pw and vw and not _is_ref(must) and pw <= desc_words and not pw <= rest_words and vw <= desc_words and not vw <= rest_words:
            continue
        keep.append(must)
    return keep


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str], fit_code: str = ""):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.fit_code = fit_code
        self.price_cap: float | None = None     # the most one unit of this item may cost, in the request's money unit
        self.pairs_with: int | None = None      # another requested item this one must work with (its model code becomes this item's fit_code)
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Kyo:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        KNOWN_REFS.clear()
        PART_KEYS.clear()
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 100
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
        self.budget_fitted = False
        self.msg_limit = 0
        self.msg_assumed = False
        self.msg_used = 0
        self.reply_key: str | None = None
        self.cart_adds: bool | None = None       # whether the cart tool adds units to a line (True) or sets them (False): learned from the store's own view
        self.last_cart: tuple[str, int, int] | None = None
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
        self.vetoed: set[str] = set()          # cart listings a review named as wrong: no pick takes them again
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
        self.gap_holds = 0
        self.retakes = 0
        self.picked: dict[str, int] = {}
        self.explored: dict[int, int] = {}
        self.explore_budget: int | None = None               # steps the narrower searches of all items may use together
        self.explore_spent = 0
        self.explore_item: dict[int, int] = {}
        _KIND_KEY.clear()
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
        self.also: dict[tuple[int, str], list[int]] = {}      # merged questions: the other items a question names
        self.asked_for: dict[str, list[int]] = {}             # question text -> the items it asked about
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
        self.parts_tried = False
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        # the 300 s episode clock, and the replies that were all thinking
        self.t0 = time.monotonic()
        self.cuts = 0
        self.quiet_light = False
        self.mute_ok = True
        self.scope = ""                      # "resolve" while the reference follower's model calls are made
        self.extra = 0.0                     # what those calls cost
        self.loop_cap = LOOP_CALLS
        self.turn_cap = 0                    # calls the store allows in one turn (0: not stated)
        # following references
        self.search_key: str | None = None
        self.chains: dict[str, dict[str, Any]] = {}
        self.resolved: dict[str, str] = {}   # pointer text -> the plain requirement written in its place
        self.ref_pages: set[str] = set()     # pages opened while following references
        self.ref_rows: set[str] = set()      # listings seen only through the follower's searches
        self.resolve_steps = 0
        self.resolve_calls = 0
        self.resolve_holds = 0
        self.replies: list[str] = []         # what the shopper said, in the order it came
        # model codes
        self.code_free = False               # learned: a search for a model code finds nothing
        self.searches = 0
        # endings without a purchase, changes to the task, offers read
        self.decline_holds = 0
        self.note_at = -1                    # step of the last change to the task
        self.work_at = -1                    # step of the last search, read or cart call made for the task
        self.stock_at = -1                   # step of the last news that a listing sold out, went away or changed its price
        self.audits = 0
        self.audit_again = False             # a change that can widen the fit set came after the last review
        self.note_held = -1                  # the change note an ending was last held back for
        self.page_seen: set[str] = set()     # listings opened with the page tool (not by a side-by-side view, not by a faulted call)
        self.line_added: dict[str, int] = {}

    # ---------- plumbing ----------
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def soft(self) -> float:
        return SPEND_SOFT + min(SPEND_EXTRA, self.extra)

    def hard(self) -> float:
        return SPEND_HARD + min(SPEND_EXTRA, self.extra)

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
        page = self.page_tool() or self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        if tool:
            return tool if self.facts_from.get(tool) is not False and self.single_target(tool) else None
        # no attributes view was named: the store's other single-listing view (apart from the offer view) may be it
        other = self.other_view(self.roles.get("offer") or "")
        return other if other and self.facts_from.get(other) is not False else None

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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              quiet: bool = False) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or self.elapsed() > CLOCK_HARD:
                return None
            self.calls += 1
            if self.scope == "resolve":
                self.resolve_calls += 1
            tries = 0
            # a call that asks for thinking off may be answered with thinking anyway (a route that ignores the setting): it gets room for both
            room = max(max_tokens, 2000) if quiet else max_tokens
            # thinking can be turned off for the light model only (the heavy one refuses it): a light reply that was all thinking and
            # was cut at its limit returns nothing and costs the clock, so the same call is sent once more with thinking off;
            # after the first 120 s, and after two such cuts, the light model works without thinking from the start
            mute = (quiet or self.quiet_light or self.elapsed() > CLOCK_FAST) and model == LIGHT and self.mute_ok
            retried = False
            while True:
                message, status = self.call_model(model, messages, room, tools, mute)
                if message is not None:
                    return message
                if status == 400 and mute:
                    # the route does not take the thinking setting: the same call without it, and never again
                    self.mute_ok = False
                    mute = False
                    continue
                if status == "length":
                    if model == LIGHT:
                        self.cuts += 1
                        if self.cuts >= 2:
                            self.quiet_light = True
                        if not retried and self.mute_ok and not mute and self.elapsed() < CLOCK_HARD - 15:
                            retried = True
                            mute = True
                            continue
                    break
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or self.elapsed() > CLOCK_SOFT):
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   mute: bool = False) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content,
        "length" for one that was cut at its limit before it said anything). With mute, an Anthropic model answers without thinking first."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if mute and self.mute_ok and model.startswith("anthropic/"):
            body["reasoning"] = {"enabled": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            thought = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") if isinstance(usage.get("completion_tokens_details"), dict) else None
            if mute and self.mute_ok and isinstance(thought, (int, float)) and not isinstance(thought, bool) and thought > 60:
                # the route took the setting and thought anyway: it is not relied on for the rest of the episode
                self.mute_ok = False
                self.trace.append({"kind": "mute_ignored", "model": model, "reasoning_tokens": thought})
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
                if self.scope == "resolve":
                    self.extra += float(usage["cost"])
            choice = data["choices"][0] if isinstance(data["choices"][0], dict) else {}
            message = choice.get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            cut = str(choice.get("finish_reason") or "") == "length"
            self.trace.append({"kind": "llm_fail", "model": model, "status": "length" if cut else 0})
            return None, ("length" if cut else 0)
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, quiet: bool = False) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, None, quiet)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS or self.elapsed() > CLOCK_HARD:
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
                if name in self.page_tools() and not (isinstance(seen, dict) and (seen.get("error") or _fault(seen))):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        ids = [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
                        self.opened.update(ids)
                        if name == self.page_tool():
                            self.page_seen.update(ids)
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
        self.search_key = self.pick_search_key(brief.get("search_arg") if isinstance(brief.get("search_arg"), str) else "")
        turn_cap = _count(self.view.get("max_calls_per_turn"))
        self.turn_cap = turn_cap if turn_cap and turn_cap >= 1 else 0
        for key in ("buy_args", "refuse_args", "search_args"):
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
            musts = _reference_echoes(self.request, [str(m) for m in _items(entry.get("musts")) if m])
            # a model code is one token of letters and digits that the request itself carries, and no value the request gives as a requirement
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./-]*", _text(entry.get("fit_code")))
            code = next((t for t in sorted(tokens, key=len, reverse=True) if len(t) >= 4 and re.search(r"\d", t) and re.search(r"[A-Za-z]", t)), "")
            if not code:
                # a code written with a space ("BEX 3520")
                spaced = re.search(r"(?<![A-Za-z])([A-Z]{2,})[\s\-]*(\d{3,}[A-Za-z0-9]*)", _text(entry.get("fit_code")))
                code = (spaced.group(1) + spaced.group(2)) if spaced else ""
            if code:
                # the code as the request writes it
                lit = re.search(r"(?<![A-Za-z0-9])" + _code_re(code) + r"(?![A-Za-z0-9])", self.request, re.I)
                code = lit.group(0) if lit else code
            if code and (not _names_code(self.request, code) or any(_names_code(m, code) for m in musts)):
                code = ""
            name, shelf = str(entry.get("name") or "the product"), str(entry.get("shelf") or "")
            self.items.append(Item(name, shelf, count, [m for m in musts if not _echoes_name(m, name, shelf)], code))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        entries = [e for e in _items(brief.get("items")) if isinstance(e, dict)]
        if len(entries) == len(self.items):
            for number, (entry, item) in enumerate(zip(entries, self.items), 1):
                cap = entry.get("price_each")
                if isinstance(cap, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", cap):
                    cap = float(cap)
                if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
                    item.price_cap = float(cap)
                other = _count(entry.get("pairs_with"))
                if other and 1 <= other <= len(self.items) and other != number and not item.fit_code:
                    item.pairs_with = other
        self.loop_cap = LOOP_CALLS_MANY if len(self.items) >= 3 else LOOP_CALLS
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
                KNOWN_REFS.add(text)
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
            # no cap stated: ask up to five and learn the cap from the store's first refusal, unless a step at the order or the
            # ending needs a message of its own (then three, as before: those messages are kept back from the interview)
            talks = any(isinstance(e, dict) and str(e.get("when") or "").strip().lower() in ("order", "end") and (_count(e.get("messages")) or 0) > 0
                        for e in _items(brief.get("process")))
            self.msg_limit = 3 if talks else 5
            self.msg_assumed = not talks
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

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            KNOWN_REFS.add(joined)
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def shelf_of(self, item: "Item") -> str:
        """The aisle label of an item's kind: the brief's, else the one aisle label the rows show whose words are the item's name or that the
        item's name contains (a lower ball joint -> "Ball Joints"); '' when none or several."""
        if item.shelf:
            return item.shelf
        cache = self.__dict__.setdefault("_shelf_cache", {})
        sig = (item.name, len(self.rows) // 25)
        if sig not in cache:
            sure = False
            try:
                iw = set(words(item.name))
                near = [(lab, set(words(lab))) for lab in self.aisle_labels()]
                # a label is a candidate when its words are a part of the name, the name is a part of its words, or the two share two words
                near = [(lab, lw) for lab, lw in near if lw and iw and (lw <= iw or iw <= lw or len(lw & iw) >= 2)]
                # the label that is the item's name, else the one that shares the most of its words and has the fewest others ("Mattresses" and
                # "Mattress Pads, Protectors & Toppers" both hold "mattress": "mattress pad with fixed straps" is the second, "hard mattress" the first)
                found = [lab for lab, lw in near if lw == iw]
                sure = bool(found)
                if not found and near:
                    best = max((len(lw & iw), -len(lw - iw)) for _, lw in near)
                    found = [lab for lab, lw in near if (len(lw & iw), -len(lw - iw)) == best]
                    # sure only when the label holds the item's head noun ("pad" in "mattress pad with fixed straps"); "gaming laptop bag" shares
                    # two words with "Gaming Laptops" but its noun, "bag", is in no label of that aisle: a guess, not exclusive
                    head = words(re.split(r"\b(?:with|for|in|that|which|of|made|having|and)\b", item.name)[0])[-1:]
                    sure = len(found) == 1 and bool(head) and head[0] in set(words(found[0]))
            except Exception:
                found = []
            cache[sig] = found[0] if len(found) == 1 else ""
            if cache[sig] and sure:
                self.__dict__.setdefault("_shelf_sure", set()).add(item.name)
        return cache[sig]

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            shelf = self.shelf_of(item)
            if shelf:
                head += f" (store label: {shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            if item.fit_code:
                text += f"; must fit {item.fit_code}: the listing's own compatibility sentence names this exact code or a family pattern it writes itself (not excepted); a later 'also fits' line does not count"
                if item.pairs_with:
                    text += f" ({item.fit_code} is the model code of the listing chosen for item {item.pairs_with}; if that listing changes, so does the code)"
                chained = [i for (i, c), ok in self.__dict__.get("_pair_fit", {}).items() if ok and c == item.fit_code]
                if chained:
                    text += f"; Kyo confirmed through another part's page that {', '.join(chained[:4])} also fit {item.fit_code}"
            elif any(other.fit_code for other in self.items):
                text += "; the request ties no model code to this item: search and judge it without the code"
            parts.append(text)
        for chain in self.chains.values():
            if chain.get("state") == "done" and chain.get("summary"):
                parts.append("reference followed by Kyo: " + str(chain["summary"]))
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            unit = ("in the store's price unit" if self.budget_fitted and self.in_price_unit(1) != 1 else brief.get("budget_unit")) or ""
            parts.append(f"budget: at most {self.budget:g} {unit} {scope}".replace("  ", " "))
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
        if self.elapsed() > CLOCK_SOFT - 50:
            text += " The clock is nearly out: settle the cart and check out within your next few moves, or end without a purchase only if your search showed that an item has no fitting listing."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += ("\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] "
                     + " | ".join(need + (f" (Kyo got as far as: {self.chains[need]['progress']})" if self.chains.get(need, {}).get("progress") else "") for need in lookups[:3]))
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
                    flags = ([self.stock_key] if self.stock_key else [k for k in node if re.search(r"stock|orderable|avail|buy|purchas|sale", str(k), re.I)]) or list(node)
                    if any(old.get(k) is True and node.get(k) is False for k in flags):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and node.get(k) is True for k in flags):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if not getattr(self, "parts_tried", True) and len([v for v in self.rows[ident].values() if isinstance(v, str)]) >= 4:
                        self.learn_parts(self.rows[ident])
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
        # the field that holds the aisle label is found as soon as rows exist (page_parts reads it)
        try:
            self.__dict__.pop("_ck", None)      # a field may have been added to rows already seen
            self.category_key()
        except Exception:
            pass

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
        n = len(self.rows)
        cached = self.__dict__.get("_ck")
        if cached and cached[0] == n:
            return cached[1]
        key = self._category_key_scan()
        self._ck = (n, key)
        if key:
            _KIND_KEY[:] = [key]
        return key

    def _category_key_scan(self) -> str | None:
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
        other = re.compile(r"seller|shop|store|vendor|maker|brand|title|name|about|desc|note|remark|price|refer|_id|id_", re.I)
        pool = {k: v for k, v in counts.items() if not other.search(str(k)) or any(x.strip().lower() in known for x in v)}
        if any(present[k] >= 0.8 * total and len(v) <= 40 for k, v in pool.items()):
            counts = pool
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
            if not record or ident in self.ref_pages or ident in self.ref_rows:
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
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                # a long list of compatible models (six entries or more) is no choice between listings
                if sum(1 for v in values if v.count(",") >= 5) * 2 >= len(values):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            if ident in self.ref_rows or ident in self.ref_pages:
                continue
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
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

    def learn_price_key(self, name: str, observation: Any) -> None:
        """Every request needs prices (the budget): a search result's shape is read once per tool whatever the pick rule (it used to be read only for
        cheapest / most / least requests, through paging)."""
        if self.price_key or name != self.roles.get("search") and name not in self.roles.values():
            return
        try:
            self.read_shape(name, observation)
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "learn_price_key", "error": f"{type(exc).__name__}: {exc}"[:200]})

    def guessed_price_key(self) -> str | None:
        """While no price key is named: a row key whose name says price or cost and whose values are numbers on most rows that carry it."""
        counts: dict[str, list[int]] = {}
        for rec in list(self.rows.values())[:60]:
            for key, value in rec.items():
                if re.search(r"price|cost", str(key), re.I) and not re.search(r"old|was|list_?before|original", str(key), re.I):
                    c = counts.setdefault(str(key), [0, 0])
                    c[1] += 1
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        c[0] += 1
        good = [(c[0], k) for k, c in counts.items() if c[1] and c[0] * 2 >= c[1] and c[0] >= 2]
        return max(good)[1] if good else None

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        key = self.price_key or self.guessed_price_key()
        value = (self.rows.get(ident) or {}).get(key) if key else None
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
    def price_unit(self) -> str | None:
        return next((s.get("price_unit") for s in self.shapes.values() if s and s.get("price_unit")), None)

    def in_price_unit(self, value: float) -> float:
        """A figure given in the request's money unit, in the unit the store's prices use (the request may count dollars while the store counts
        cents, or the other way round); unchanged when either unit is unknown."""
        unit = str(self.brief.get("budget_unit") or "").lower()
        pu = self.price_unit()
        if not pu or not unit:
            return value
        minor = re.search(r"cent|penny|pence|paise|kopek|minor", unit) is not None
        major = not minor and re.search(r"dollar|euro|pound|usd|eur|gbp|aud|cad|nzd|yen|rupee|franc|\$|€|£", unit) is not None
        if minor and pu == "units":
            return value / 100
        if major and pu == "cents":
            return value * 100
        return value

    def fit_budget(self) -> None:
        if self.budget_fitted or not isinstance(self.budget, (int, float)) or not self.price_unit():
            return
        fitted = self.in_price_unit(self.budget)
        self.budget_fitted = True
        if fitted != self.budget:
            self.trace.append({"kind": "budget_unit", "given": self.budget, "in_price_unit": fitted})
            self.budget = fitted

    def learn_parts(self, rec: dict[str, Any]) -> None:
        """Which keys of a listing's record hold its title, description, seller, seller's note and category: named once per episode by the light
        model from one record (a store renames them every race); page_parts falls back to the key names for anything it leaves out."""
        self.parts_tried = True
        sample = {k: (v if not isinstance(v, str) or len(v) <= 160 else v[:160] + "...") for k, v in rec.items() if not isinstance(v, (dict, list))}
        prompt = PARTS + "\n\nRECORD:\n" + _dump(sample)[:3000]
        try:
            got = self.ask_json(prompt, [LIGHT, HEAVY], 400) or {}
        except Exception:
            return
        for part in ("title", "overview", "seller", "note", "kind"):
            key = got.get(part)
            if isinstance(key, str) and key in rec and isinstance(rec[key], str):
                PART_KEYS[part] = key
        if PART_KEYS:
            self.trace.append({"kind": "parts", "keys": dict(PART_KEYS)})

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if isinstance(observation, list) and len(observation) >= 2 and all(isinstance(r, dict) for r in observation):
            observation = {"rows": observation}        # a bare list of rows: shaped as if under one key (no count, no next page)
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            # the rows may sit one level down, under a wrapper that carries a notice beside the result
            inner = _rows_level(observation)
            return self.read_shape(name, inner) if inner is not None and inner is not observation else None
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
        shape["price_unit"] = got.get("price_unit") if got.get("price_unit") in ("cents", "units") else None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.fit_budget()
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        if shape["rows"] not in observation:
            observation = _rows_level(observation) or observation
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
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
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

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
            number = self.item_of_label(kind)
            code = self.items[number - 1].fit_code if number else ""
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price
                    # a cheaper row that does not name the item's model code cannot be the cheapest valid one
                    and (not code or _fit_state(self.rows.get(i), code)[0] in ("names", "unknown"))]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        self.learn_price_key(name, observation)
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
            lanes.append(pool[: max(0, min(2 if len(self.items) > 1 else 5, 8 - opened if len(self.items) > 1 else 11 - opened))])
        depth = 2 if len(self.items) > 1 else 5
        queue = [lane[k] for k in range(depth) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode (a request for one product reads up to eight more: the keys that
        # tell its listings apart are then read off ten, not three)
        self.fill_facts(queue[:5 if len(self.items) > 1 else 8], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        # pages opened while following a reference say nothing about the requested items' shelves, and neither do the steps spent on them
        opened = len([i for i in self.full if i not in self.ref_pages and i not in self.ref_rows])
        # a basket of three or more lines needs steps for the pick: two listings per line show the property names, the interviewer reads more
        if opened >= (2 * len(self.items) + 2 if len(self.items) >= 3 else 3 * len(self.items) + 1):
            return True
        return self.step - self.resolve_steps >= max(8, self.max_steps // 4)

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

    def read_status(self) -> None:
        """The request stated no message cap and the store has a tool that reports it: read it once and let the light model take the number of messages
        still allowed from the result."""
        tool = self.roles.get("status")
        if not tool or not self.msg_assumed or self.__dict__.get("_status_read") or self.left() < 12 or self.finished:
            return
        self._status_read = True
        faults: list[str] = []
        args = _shape({}, self.schemas.get(tool), "args", faults)
        if faults:
            return
        reply = self.act(tool, args)
        observation = (reply or {}).get("observation")
        if observation is None:
            return
        got = self.ask_json(STATUS + "\n\nRESULT:\n" + _dump(_clip(observation, 300))[:1500], [LIGHT, HEAVY], 300) or {}
        left = _count(got.get("left"))
        if left is not None and 0 <= left <= 20:
            self.msg_limit = self.msg_used + left
            self.msg_assumed = False
            self.trace.append({"kind": "msg_cap_read", "left": left, "cap": self.msg_limit})

    def interview(self) -> str:
        self.asked = True
        try:
            self.read_status()
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "read_status", "error": f"{type(exc).__name__}: {exc}"[:200]})
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name}" + (" (must fit a model code)" if item.fit_code else "") + " - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered: list[tuple[int, str]] = []
        planned = 0
        self.asked_for = {}
        hinted: set[tuple[int, str]] = set()
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
                if str(pick.get("hint") or "").strip():
                    hinted.add((number, _words(label)))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        try:
            if not first:
                # what the request itself asks to be checked comes first, in the planner's order
                ordered = self.spread(ordered, hinted)
            ordered = self.snap_labels(ordered, table, hinted)
            ordered = self.merge_plan(ordered)
            planned = len(ordered)
            ordered = self.fill_questions(ordered, table, count)
        except Exception as exc:
            self.trace.append({"kind": "fill_questions_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # all messages but the last first; the last re-asks a question whose answer wanted a single question
        # (named with example values this time), else it goes to the next pick
        spare = 2 if (self.msg_assumed and count >= 4) else 1
        head = ordered[: count - spare] if count >= 2 else ordered[:count]
        for number, label in ordered:
            self.asked_for[self.question(number, label)] = self.asked_items(number, label)
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        heard = self.hear(batch)
        self.take_heard(heard, told)
        ni = len(head)
        for turn in range(max(0, count - len(head))):
            if count < 2 or self.msgs_free() <= 0:
                break
            again = None
            if turn == 0:
                crowded = [batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
                for number, text, _ in crowded:
                    label = next((lab for n, lab in head if n == number and self.question(n, lab) == text), None)
                    samples = self.samples(number, label) if label else []
                    near = self.similar_keys(number, label) if label else []
                    head = self.head_label(number, label) if label else ""
                    options = []
                    if label and near:
                        options.append(self.question(number, label, None, [], near))
                    if head:
                        options.append(self.question(number, head))
                    if label and len(samples) >= 2:
                        options.append(self.question(number, label, samples, self.sharing_items(number, label)))
                    option = next((o for o in options if o != text), None)
                    if label and option:
                        again = (number, label, option)
                        break
            if again is None:
                # the first question was merged and revealed nothing: it may have met a look-alike of one of its items (the property
                # most likely to be hidden is asked first, and its look-alikes sit on the same shelf), so it is asked for one item now
                again = self.reask_merged(head, heard)
            if again is None and ni < len(ordered):
                number, label = ordered[ni]
                ni += 1
                again = (number, label, self.question(number, label))
            if not again:
                break
            batch = self.send([again])
            if not batch:
                break
            sent += batch
            # one answer: the light model reads it
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
        if told:
            self.version += 1
        # a requirement that points at another listing is followed now, so the actor works with its value
        followed = self.resolve_pending("asked") if self.pending_refs() else ""
        told = [self.after_follow(t) for t in told]
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if followed:
            summary += "\n" + followed
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def keys_of(self, number: int, table: list[tuple[str, str, int, list[str]]] | None = None) -> list[str]:
        """The key names the listings of the item's category carry, most carried first."""
        table = table if table is not None else self.key_table()
        return [name for cat, name, _, _ in table if self.label_owner(cat) == number]

    @_guarded(lambda self, ordered, table, hinted=frozenset(): ordered)
    def snap_labels(self, ordered: list[tuple[int, str]], table: list[tuple[str, str, int, list[str]]],
                    hinted: "set[tuple[int, str]] | frozenset" = frozenset()) -> list[tuple[int, str]]:
        """A label that is a part of exactly one key of the item's listings ("size" for "display size") is asked as that key reads: the shopper's classifier
        scores a key's own wording 0.97-0.99 and a head word alone 0.3-0.7 (probe on the local packs)."""
        out: list[tuple[int, str]] = []
        for number, label in ordered:
            base = _words(label)
            keys = self.keys_of(number, table)
            # only a label of two or more words that is no story hint and no key of any read listing (the table shows the first 14 only) is snapped: a single
            # word is the property the model means ("type" is not "warranty type", "height" is not "package height"), and a key about the warranty, the
            # options or the year is never the answer
            if len(base.split()) >= 2 and (number, base) not in hinted and base not in keys and base not in self.known_keys(number):
                inside = [k for k in keys if set(base.split()) < set(k.split()) and not _LOGISTICS.search(k)]
                if len(inside) == 1:
                    label = inside[0]
            if all(_words(label) != _words(other) for n, other in out if n == number):
                out.append((number, label))
        return out

    def known_keys(self, number: int) -> set[str]:
        """Every key name any read listing of the item's category carries (not just the table's)."""
        cat_key = self.category_key()
        found: set[str] = set()
        for record in self.full.values():
            if cat_key and self.label_owner(str(record.get(cat_key) or "")) == number:
                found |= {_words(k) for k in (_fact_block(record) or {})}
        return found

    @_guarded(lambda self, ordered, hinted: ordered)
    def spread(self, ordered: list[tuple[int, str]], hinted: set[tuple[int, str]]) -> list[tuple[int, str]]:
        """Three or more parts that must fit a model code: every part gets one question before any part gets a second (a planner that spent 47 of 160 questions on
        rotors, which hid nothing, left the floor mats, which hide a colour or a material in half the tasks, with 3). Picks the story hints at stay first."""
        fit_parts = len(self.items) >= 3 and any(it.fit_code for it in self.items)
        # few messages for several products (2 messages, 3 products): two questions on two products found 0.34 needs per episode, two on one product 0.19
        if not fit_parts:
            try:
                scarce = len(self.items) >= 2 and self.msgs_free() < 2 * len(self.items)
            except Exception:
                scarce = False
            if not scarce:
                return ordered
        seen: dict[int, int] = {}
        ranked = []
        # the item others must work with carries the unwritten preference more often than its accessories: it leads each round
        mains = {it.pairs_with for it in self.items if getattr(it, "pairs_with", None)}
        for index, (number, label) in enumerate(ordered):
            nth = seen.get(number, 0)
            seen[number] = nth + 1
            ranked.append((-1 if (number, _words(label)) in hinted else nth, 0 if number in mains else 1, index, number, label))
        ranked.sort()
        return [(number, label) for _, _, _, number, label in ranked]

    def merge_plan(self, ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """A property the planner lists for several items is asked once, naming them all (a request that ties items to a model code only:
        there the unwritten preferences sit on a few shared properties). `self.also` keeps the other items of a merged question."""
        self.also = {}
        if FANOUT and len(self.items) >= 2 and any(it.fit_code for it in self.items):
            try:
                return self.fan_out(ordered)
            except Exception as exc:
                self.trace.append({"kind": "guarded", "fn": "fan_out", "error": f"{type(exc).__name__}: {exc}"[:200]})
                self.also = {}
                return ordered
        if not MERGE_PROPERTIES or not any(it.fit_code for it in self.items):
            return ordered
        groups: list[list[Any]] = []
        for number, label in ordered:
            group = next((g for g in groups if _words(g[0]) == _words(label)
                          and bool(self.items[g[1][0] - 1].fit_code) and bool(self.items[number - 1].fit_code)), None)
            if group is None:
                groups.append([label, [number]])
            elif number not in group[1] and len(group[1]) < 3:
                group[1].append(number)
        out = []
        for label, numbers in groups:
            out.append((numbers[0], label))
            if len(numbers) > 1:
                self.also[(numbers[0], _words(label))] = numbers[1:]
        return out

    def fan_out(self, ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """The planner's picks, each extended to every other product whose listings carry the same key: the pick stays where the planner put it
        (it knows which property is likely to be hidden), and the message that asks it for one product asks it for all of them. The planner's
        later pick of the same key for a product already named is dropped (the message covers it)."""
        carry: dict[str, dict[int, int]] = {}
        for cat, name, carried, _ in self.key_table():
            owner = self.label_owner(cat)
            if owner is None or _LOGISTICS.search(name):
                continue
            carry.setdefault(name, {})[owner] = carried
        covered: set[tuple[int, str]] = set()
        out: list[tuple[int, str]] = []
        for number, label in ordered:
            key = _words(label)
            if (number, key) in covered:
                continue
            owners = carry.get(key, {})
            others = sorted((m for m in owners if m != number and (m, key) not in covered and not self.settled_prop(m, key)),
                            key=lambda m: (-owners[m], m))[:FANOUT_NAMES - 1]
            out.append((number, label))
            if others:
                self.also[(number, key)] = sorted(others)
            covered |= {(m, key) for m in [number] + others}
        if self.also:
            self.trace.append({"kind": "fan_out", "asks": [(key, [n] + o) for (n, key), o in self.also.items()]})
        return out

    def reask_merged(self, head: list[tuple[int, str]], heard: dict[str, Any]) -> tuple[int, str, str] | None:
        """The first merged question whose answers revealed nothing, asked again for one item (the one the request settles least)."""
        for number, label in head:
            others = getattr(self, "also", {}).get((number, _words(label)))
            if not others:
                continue
            group = [number] + list(others)
            got = False
            for entry in _items(heard.get("needs")):
                # the property decides, not the item the reader pinned the reveal to (a reveal names its product in its own words)
                if isinstance(entry, dict) and _stem(str(entry.get("need") or "").split(":")[0]) & _stem(label):
                    got = True
            if got:
                continue
            pick = min(group, key=lambda n: (len(self.items[n - 1].musts) + len(self.items[n - 1].told), n))
            self.also.pop((number, _words(label)), None)
            text = self.question(pick, label)
            self.asked_for[text] = [pick]
            return pick, label, text
        return None

    def asked_items(self, number: int, label: str) -> list[int]:
        return [number] + list(getattr(self, "also", {}).get((number, _words(label)), []))

    def lookalikes(self, number: int, label: str) -> list[str]:
        """Keys of the item's own category that repeat the property's name and add a word or two after it ("color family" beside "color"):
        the shopper's classifier scores such a key as a subject too, and two subjects make it reply that it wants a single question. Read off the
        listings' own attributes (the planner's table drops a key that only copies another). A key that puts a word before the name
        ("accessories material" beside "material") is not named: that lowers the real property's score."""
        try:
            base = _words(label)
            cat_key = self.category_key()
            count: dict[str, int] = {}
            for ident, record in self.full.items():
                if ident in self.ref_pages or ident in self.ref_rows:
                    continue
                if self.label_owner(str(record.get(cat_key) or "") if cat_key else "") != number:
                    continue
                for key in _fact_block(record) or {}:
                    name = _words(key)
                    if name.startswith(base + " ") and 1 <= len(name.split()) - len(base.split()) <= 2:
                        count[name] = count.get(name, 0) + 1
            return [n for n, c in sorted(count.items(), key=lambda kv: -kv[1]) if c >= 2][:2]
        except Exception:
            return []

    def sharing_items(self, number: int, label: str) -> list[str]:
        """Names of the other requested items whose listings carry an attribute with this very name ("capacity" on both the internal SSDs and the
        internal hard drives): the shopper's classifier takes a question about one of them as a question about both. Asked once per property, so a question
        reads the same each time it is built."""
        cache = self.__dict__.setdefault("_share_cache", {})
        base = _words(label)
        if (number, base) in cache:
            return cache[(number, base)]
        found: list[str] = []
        try:
            table = self.key_table()
            if any(name == base and self.label_owner(cat) == number for cat, name, _, _ in table):
                # the other items' keys with this very name first, then those that end in the same word (len 5+) as the property ("storage capacity" beside
                # "capacity"): live, an SSD question was answered "one question at a time" 77% of the time when a hard-drive line was also in the basket
                head = base.split()[-1] if base.split() else ""
                for exact in (True, False):
                    for cat, name, _, _ in table:
                        other = self.label_owner(cat)
                        if other is None or other == number or self.items[other - 1].name in found:
                            continue
                        if (name == base) if exact else (len(head) >= 5 and name.split()[-1:] == [head]):
                            found.append(self.items[other - 1].name)
        except Exception:
            found = []
        cache[(number, base)] = found[:2]
        return cache[(number, base)]

    def similar_keys(self, number: int, label: str) -> list[str]:
        """Other keys of the item's listings that share a word with the property ("bundle size" and "number of pieces" beside "pack size"): the shopper's
        classifier scores such a key as a subject too, and a question that names two subjects gets the reply that the questions should come separately."""
        try:
            base = _stem(label)
            own = _words(label)
            out = []
            for name in self.keys_of(number):
                if name == own or name.startswith(own + " "):
                    continue
                if {w for w in _stem(name) & base if len(w) >= 3}:
                    out.append(name)
            return out[:2]
        except Exception:
            return []

    def question(self, number: int, label: str, samples: list[str] | None = None, apart_items: list[str] | None = None, near: list[str] | None = None) -> str:
        names = [self.items[n - 1].name for n in self.asked_items(number, label) if 1 <= n <= len(self.items)]
        named = ", the ".join(names[:-1]) + (" and the " if len(names) > 1 else "") + names[-1]
        if len(names) > 1 and apart_items is None:
            apart_items = []          # the question names every product that carries the property: no "Apart from" those
        apart = (self.lookalikes(number, label) + [k for k in (near or []) if k not in self.lookalikes(number, label)])[:2]
        if apart_items is None and not apart and samples is None:
            # the first wording: another requested item carries the same property
            apart_items = self.sharing_items(number, label)
        if apart_items and not apart:
            return (f"Apart from the {' and the '.join(apart_items)}, what {label} would you like for the {named}"
                    + (f", for example {samples[0]} or {samples[1]}?" if samples else "?"))
        text = (f"Apart from the {' and the '.join(apart)}, which {label} would you like for the {named}" if apart
                else f"What {label} would you like for the {named}")
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def head_label(self, number: int, label: str) -> str:
        """The property's name without its last, kind-like word ("packaging" for "packaging type") when the item's listings carry that shorter
        name as a key of its own: the shopper's classifier then meets one subject where the longer name met two ("... type" keys share a word)."""
        parts = _words(label).split()
        if len(parts) < 2 or parts[-1] not in _KIND_PROPS | {"style", "form", "variety", "option", "options", "family"}:
            return ""
        head = " ".join(parts[:-1])
        try:
            keys = {_words(k) for k in self.keys_of(number)} | self.known_keys(number)
        except Exception:
            return ""
        return head if head in keys else ""

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
            if self.msg_assumed and self.refused_message(answer):
                # the store refused it: its cap is the number of messages that went through
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                self.msg_assumed = False
                self.trace.append({"kind": "msg_cap_learned", "cap": self.msg_limit})
                break
            words = self.spoken(answer)
            self.learn_reply_key(answer)
            self.replies.append(words)
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
            if need in self.resolved:
                # a pointer already followed: the plain requirement written in its place stands
                continue
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
                KNOWN_REFS.add(need)
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
            self.budget = self.in_price_unit(heard["budget"]) if self.budget_fitted else heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        about = lambda n, q: (f"items {', '.join(str(x) for x in self.asked_for[q])}" if len(self.asked_for.get(q, [])) > 1 else f"item {n}")
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about {about(n, q)}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.ask_json(prompt, list(models), 1500) or {}
        try:
            self.settle_items(got, sent)
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "settle_items", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return got

    def settle_items(self, got: dict[str, Any], sent: list[tuple[int, str, str]]) -> None:
        """A message that asks one property of several products gets a single reveal, meant for just one product. When the reader gave the same requirement to
        several items, only the item(s) whose name the answer carries keep it (all keep it when the answer names none of them)."""
        needs = [e for e in _items(got.get("needs")) if isinstance(e, dict)]
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in needs:
            groups.setdefault(re.sub(r"\s+", " ", str(entry.get("need") or "").lower()).strip(), []).append(entry)
        for text, group in groups.items():
            numbers = {_count(e.get("item")) or 1 for e in group}
            if not text or len(group) < 2 or len(numbers) < 2:
                continue
            value = set(words(text.split(":", 1)[-1], drop_stop=False))
            carrying = [a for _, _, a in sent if value and value <= set(words(a, drop_stop=False))] or [a for _, _, a in sent]
            heard = set(words(" ".join(carrying), drop_stop=False))
            score = {}
            for n in numbers:
                if 1 <= n <= len(self.items):
                    name = set(words(self.items[n - 1].name))
                    score[n] = len({w for w in name if _has(w, heard)}) / max(1, len(name))
            best = max(score.values(), default=0.0)
            winners = {n for n, sc in score.items() if sc == best}
            if best > 0 and len(winners) < len(numbers):
                got["needs"] = [e for e in _items(got.get("needs")) if not (isinstance(e, dict) and re.sub(r"\s+", " ", str(e.get("need") or "").lower()).strip() == text
                                                                           and (_count(e.get("item")) or 1) not in winners)]
                self.trace.append({"kind": "need_settled", "need": text[:60], "items": sorted(winners)})

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
                if isinstance(value, str) and len(value.split()) >= 3 and value not in self.rows:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 3 and v not in self.rows)
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 4 and inner not in self.rows:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold or got.get("price_moved") is True:
            self.stock_at = self.step
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and self.budget_fitted:
            budget = self.in_price_unit(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            if self.budget is None or budget > self.budget:
                self.audit_again = True
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
            self.audit_again = True
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
            self.note_at = self.step
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
                    record = self.full.get(ident) or {}
                    pp = self.page_parts(record)
                    state, why = verify_must(need, block, pp["title"], pp["overview"], pp["kind"])
                    mark = " [meets it]" if state == "pass" else (f" [does NOT meet it: {why}]" if state == "fail" else "")
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]) + mark)
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
        for ident in order:
            if ident in skip or self.sold_out(ident) or self.set_aside(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def line_item(self, ident: str) -> int | None:
        """The requested item a listing was found for: by its aisle label, else by the words of its title."""
        rec = self.full.get(ident) or self.rows.get(ident) or {}
        parts = self.page_parts(rec)
        best, best_score = None, 0.0
        stems = frozenset(words(parts["kind"])) if parts["kind"] else frozenset()
        title = set(words(parts["title"]))
        for number, item in enumerate(self.items, 1):
            names = {w for w in set(words(item.shelf or "")) | set(words(item.name)) if len(w) >= 3}
            score = len({w for w in stems if len(w) >= 3} & names) + 0.2 * len(set(words(item.name)) & title)
            if names:     # of two items a label fits, the one whose name it covers more of
                score += 0.5 * len({w for w in stems if len(w) >= 3} & names) / len(names)
            if parts["kind"] and _words(parts["kind"]) in (_words(self.shelf_of(item)), _words(item.name)):
                score += 10
            if score > best_score:
                best, best_score = number, score
        return best

    def verify_line(self, number: int, ident: str) -> list[str]:
        """What the requirements of item `number` and the listing's own attributes show, by the words alone."""
        rec = self.full.get(ident) or {}
        if not rec or not 1 <= number <= len(self.items):
            return []
        item = self.items[number - 1]
        parts = self.page_parts(rec)
        if not parts["facts"]:
            return []
        problems: list[str] = []
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            state, why = verify_must(must, parts["facts"], parts["title"], parts["overview"], parts["kind"])
            if state == "fail":
                problems.append(f"{ident} (item {number}, {item.name}): requirement \"{must}\": {why}")
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            cap = self.price_cap_problem(must, ident)
            if cap:
                problems.append(f"{ident} (item {number}, {item.name}): requirement \"{must}\": {cap}")
        unit = self.unit_cap_problem(number, ident)
        if unit:
            problems.append(f"{ident} (item {number}, {item.name}): {unit}")
        shelf = self.shelf_of(item)
        if shelf and parts["kind"] and not self.same_aisle(shelf, parts["kind"], item):
            problems.append(f"{ident} (item {number}, {item.name}): it stands in the aisle \"{parts['kind']}\", the store's aisle for this kind is \"{shelf}\"")
        return problems

    def unit_cap(self, item: "Item") -> float | None:
        """The per-unit price cap of an item in the unit the store's prices use (the request's unit converted once the store's unit is known)."""
        if item.price_cap is None:
            return None
        return self.in_price_unit(item.price_cap) if self.budget_fitted else item.price_cap

    def unit_cap_problem(self, number: int, ident: str) -> str:
        """The listing's price against its item's per-unit cap; '' when it holds, no cap was stated or the price is unknown."""
        item = self.items[number - 1]
        cap, price = self.unit_cap(item), self.price_of(ident)
        if cap is None or price is None or price <= cap + 1e-9:
            return ""
        return f"its price is {price:g}, above the {cap:g} the request allows for one unit"

    def price_cap_problem(self, must: str, ident: str) -> str:
        """A requirement on the listing's own price ("price: at most 900 cents", "cost: under 12 dollars") against its price per unit; '' when it holds or
        the requirement is not about the price."""
        prop, cons = split_must(must)
        if not prop or not cons or not re.search(r"\b(?:price|cost)\b", prop.lower()):
            return ""
        rel = _REL_LEAD.match(cons)
        figures = _NUM.findall(cons)
        price = self.price_of(ident)
        if not rel or not figures or price is None:
            return ""
        limit = float(figures[0])
        if re.search(r"dollar|\$|usd(?!-cent)|euro|pound", cons.lower()) and not re.search(r"cent", cons.lower()):
            limit = self.in_price_unit(limit) if self.budget_fitted else limit
        kind = rel.group(1).lower()
        if kind in ("at most", "no more than", "up to", "less than") and price > limit + 1e-9:
            return f"its price is {price:g}, above {limit:g}"
        if kind in ("at least", "no less than", "more than") and price < limit - 1e-9:
            return f"its price is {price:g}, below {limit:g}"
        return ""

    def verify_cart(self, ids: list[str]) -> list[str]:
        """What the requirements and the listings' own attributes show, by the words alone: one line per cart listing that an attribute
        naming a requirement's property contradicts, or that stands in another aisle than the item's; an item with no line of its own;
        a line whose look-alike (same title and price) states the item's wording in full. Attributes not read, other units, synonyms
        and everything else are left to the review."""
        problems: list[str] = []
        self.sync_pair_codes(ids)
        for ident in ids:
            number = self.line_item(ident)
            if number is not None:
                problems += self.verify_line(number, ident)
                item = self.items[number - 1]
                if item.pairs_with and item.fit_code and self.fit_rank(item, ident) >= 2:
                    problems.append(f"{ident} (item {number}, {item.name}): its compatibility text does not name {item.fit_code}, the model of the item-{item.pairs_with} line in the cart")
        return problems + self.coverage_problems(ids) + self.twin_problems(ids) + self.quantity_problems(ids)

    def quantity_problems(self, ids: list[str]) -> list[str]:
        """A cart line (for an item whose aisle is its own) whose unit count differs from the count the request asks for."""
        out: list[str] = []
        for ident in ids:
            have = self.cart_lines.get(ident)
            if not isinstance(have, int) or isinstance(have, bool) or self.sold_out(ident):
                continue
            mine = [n for n in range(1, len(self.items) + 1) if self.in_aisle(n, ident)]
            if len(mine) != 1:
                continue
            item = self.items[mine[0] - 1]
            if item.qty and have != item.qty:
                out.append(f"{ident} (item {mine[0]}, {item.name}): the cart holds {have} unit(s), the request asks for {item.qty}")
        return out

    def coverage_problems(self, ids: list[str]) -> list[str]:
        live = [i for i in ids if not self.sold_out(i)]
        out: list[str] = []
        if live and len(live) < len(self.items):
            out.append(f"the cart holds {len(live)} listing(s) for {len(self.items)} requested items; each item needs a line of its own")
        kinds = {i: self.page_parts(self.full.get(i) or self.rows.get(i) or {})["kind"] for i in live}
        if live and all(kinds.values()):
            for number, item in enumerate(self.items, 1):
                shelf = self.shelf_of(item)
                if shelf and not any(self.same_aisle(shelf, k, item) for k in kinds.values()):
                    out.append(f"item {number} ({item.name}): no cart line comes from the store's aisle for it, \"{shelf}\"")
        return out

    def twins_of(self, ident: str) -> list[str]:
        """Other listings with the same aisle, title (but for a code in brackets or digits) and price: look-alikes whose attributes differ."""
        rec = self.rows.get(ident) or {}
        parts = self.page_parts(rec)
        if not parts["title"]:
            return []
        norm = lambda t: re.sub(r"\[[^\]]*\]|\d+", "", t or "").lower().strip()
        price = self.price_of(ident)
        found = []
        for j in self.seen_order:
            if j == ident or j in self.ref_pages or j in self.ref_rows or self.sold_out(j):
                continue
            q = self.page_parts(self.rows.get(j) or {})
            if q["kind"] == parts["kind"] and norm(q["title"]) == norm(parts["title"]) and self.price_of(j) == price:
                found.append(j)
        return found[:6]

    def name_in_facts(self, item: "Item", facts: dict[str, Any]) -> bool:
        """Whether one attribute value of a listing holds every word of the item's name (two words or more): "tapered roller bearing"."""
        phrase = [w for w in words(item.name)]
        if len(phrase) < 2:
            return False
        return any(set(phrase) <= set(words(str(v), drop_stop=False)) or all(_has(w, set(words(str(v), drop_stop=False))) for w in phrase) for v in facts.values()
                   if isinstance(v, (str, int, float)))

    def strong_musts(self, item: "Item") -> list[str]:
        """The requirements whose value is only the start of the item's own name ("tapered roller" for "tapered roller bearings"), with the
        whole name as the value: a listing whose attribute states the whole name is the better one when look-alikes differ in it."""
        phrase = words(item.name)
        found = []
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            prop, cons = split_must(must)
            cw = words(cons)
            if prop and cw and len(phrase) > len(cw) and phrase[:len(cw)] == cw:
                found.append(f"{prop}: {' '.join(phrase)}")
        return found

    def twin_score(self, number: int, ident: str) -> tuple[int, int, int] | None:
        rec = self.full.get(ident)
        if not rec:
            return None
        item = self.items[number - 1]
        parts = self.page_parts(rec)
        if not parts["facts"]:
            return None
        strong = sum(1 for m in self.strong_musts(item) if verify_must(m, parts["facts"], parts["title"], parts["overview"], parts["kind"])[0] == "pass")
        strong += 1 if self.name_in_facts(item, parts["facts"]) else 0
        fits = 1 if (not item.fit_code or _fit_state(rec, item.fit_code)[0] == "names") else 0
        return (-len(self.verify_line(number, ident)), fits, strong)

    def twin_problems(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for ident in ids:
            number = self.line_item(ident)
            if number is None:
                continue
            mine = self.twin_score(number, ident)
            if mine is None:
                continue
            for twin in self.twins_of(ident):
                theirs = self.twin_score(number, twin)
                if theirs and theirs[0] >= mine[0] and theirs[1] >= mine[1] and theirs[2] > mine[2]:
                    out.append(f"{ident} (item {number}, {self.items[number - 1].name}): its attributes do not state the item's name in full; its look-alike {twin} "
                               f"(same title and price) does{' and names the model code' if theirs[1] else ''}: take {twin} instead")
                    break
        return out

    def read_twins(self) -> None:
        """Open the unread look-alikes of a cart line whose item has a requirement worded as only the start of its name (a few steps)."""
        done = self.__dict__.setdefault("_twins_read", set())
        for ident in self.cart_ids():
            number = self.line_item(ident)
            if number is None or ident in done or len(words(self.items[number - 1].name)) < 2:
                continue
            done.add(ident)
            unread = [t for t in self.twins_of(ident) if t not in self.full]
            if unread and self.left() > self.tail() + 8:
                self.fill_facts(unread[:4], self.tail() + 6)

    def cart_over_budget(self) -> str:
        if not isinstance(self.budget, (int, float)) or self.brief.get("budget_scope") == "each":
            return ""
        total, known = 0.0, 0
        for ident, count in self.cart_lines.items():
            price = self.price_of(ident)
            if price is not None and not self.sold_out(ident):
                total += price * count
                known += 1
        if known and known == len([i for i in self.cart_lines if not self.sold_out(i)]) and self.budget < total < 20 * self.budget:
            return f"the cart's total (price times count, all lines) is {total:g}, over the budget of {self.budget:g}"
        return ""

    def verify_hold(self) -> str:
        """Before a checkout: the code check's findings, at most twice for the same findings (a wrong finding must not stall the episode)."""
        try:
            for n in range(1, len(self.items) + 1):
                self.add_named_need(n, self.pool_of(n))
            self.read_twins()
            cart_now = self.cart_ids()
            self.sync_pair_codes(cart_now)
            for ident in cart_now:
                number = self.line_item(ident)
                item = self.items[number - 1] if number else None
                if item and item.pairs_with and item.fit_code and self.fit_rank(item, ident) >= 2:
                    self.follow_pair_chains(number, [ident], limit=1)
            problems = self.verify_cart(cart_now)
            over = self.cart_over_budget()
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "verify_hold", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""
        if over:
            problems.append(over)
        if problems:
            took = self.safe_retake("the cart fails Kyo's word check (" + "; ".join(x[:140] for x in problems[:2]) + ")")
            if took:
                return took
        else:
            took = self.safe_retake("", cheaper=True) if self.brief.get("pick") == "cheapest" else ""
            if not took:
                try:
                    took = self.retake_confirmed()
                except Exception as exc:
                    self.trace.append({"kind": "guarded", "fn": "retake_confirmed", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    took = ""
            if took:
                return took
        if not problems:
            return ""
        key = "|".join(sorted(problems))
        held = self.__dict__.setdefault("_verify_held", {})
        if held.get(key, 0) >= 2 or sum(held.values()) >= 5:
            return ""
        held[key] = held.get(key, 0) + 1
        self.trace.append({"kind": "verify_hold", "problems": [x[:160] for x in problems[:4]]})
        return ("Kyo compared the cart with the requirements by their words and found: " + "; ".join(problems[:4])
                + ". Look for a listing in the right aisle whose attribute states the required value (search again with the value in the query); "
                "if the store has none, keep the best line and check out again.")

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        limit = self.soft() if limit is None else limit
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
        try:
            code_found = self.verify_cart(ids)
        except Exception:
            code_found = []
        prompt += ("\n\nCODE CHECK (Kyo compared requirements and attributes word by word; units, synonyms and unread attributes are left to you): "
                   + ("; ".join(code_found[:5]) if code_found else "no mismatch found"))
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
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
        # a listing the review names as wrong is not picked again; with the cart in Kyo's hands the pick is redone now
        named = {i for i in ids if any(i in p for p in problems) and sum(1 for j in ids if self.line_item(j) == self.line_item(i)) == 1}
        if named and len(self.items) >= 2 and len(named) < len(ids) + 1:
            self.vetoed |= named
            self.trace.append({"kind": "vetoed", "ids": sorted(named)})
            if self.picked and named & set(self.picked):
                again = self.safe_retake("the review found " + ", ".join(sorted(named)) + " wrong for its item")
                if again:
                    return text + ". " + again
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    # ---------- following references ----------
    def pick_search_key(self, named: str) -> str | None:
        """The search tool's argument that takes the words to look for: the one the brief named, else the required text
        argument that speaks of a query, else any required text argument, else the first text argument not about paging."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        if named in texts:
            return named
        required = [k for k in schema.get("required") or [] if k in texts]
        if required:
            speaking = [k for k in required if re.search(r"quer|search|word|phrase|term|text|find|keyword", k + " " + str((props.get(k) or {}).get("description") or ""), re.I)]
            return (speaking or required)[0]
        plain = [k for k in texts if not re.search(r"after|page|token|cursor|offset|skip|next|start", k, re.I)]
        return plain[0] if plain else None

    def page_tool(self) -> str | None:
        """The tool whose view counts as opening a listing's page: the offer view the brief named; else the one single-listing tool
        that is not the attributes view; else the single-listing view."""
        tool = self.roles.get("offer")
        if not tool:
            used = {self.roles.get(r) for r in ("cart", "buy", "refuse", "ask", "search", "cart_view", "compare", "open")} - {None}
            others = [t for t in self.schemas if t not in used and self.single_target(t)]
            tool = others[0] if len(others) == 1 else self.roles.get("open")
        return tool if tool and self.single_target(tool) else None

    def resolve_room(self) -> int:
        # a big basket needs most of the steps for itself: 36 steps for one line down to 23 for five (a three-page chain needs about 20)
        return min(RESOLVE_STEPS, max(6, self.max_steps * 2 // 5), max(16, 48 - 5 * len(self.items)))

    def can_resolve(self) -> bool:
        return (not self.finished and not self.broke and not self.no_credit and bool(self.search_key) and bool(self.page_tool())
                and self.resolve_steps < self.resolve_room() and self.left() > 12 and self.elapsed() < CLOCK_SOFT - 25
                and self.cost < self.hard() and self.extra < FOLLOWER_SPEND and self.resolve_calls < RESOLVE_CALLS
                and self.calls < ALL_CALLS - 4)

    def pending_refs(self) -> list[tuple[int, str]]:
        """Requirements that take their value from another listing and have not been followed to it (nor given up)."""
        live = {need for _, need in self.needs_now()}
        return [(n, need) for n, need in self.refs
                if need in live and need not in self.ref_values and self.chains.get(need, {}).get("state") != "failed"]

    def rcall(self, name: str, args: dict[str, Any]) -> Any:
        """One store call made by the reference follower: its result, or None when it could not be made."""
        faults: list[str] = []
        args = _shape(args, self.schemas.get(name), "args", faults)
        if faults or self.finished or self.left() <= 8:
            return None
        before = set(self.rows)
        reply = self.act(name, args)
        self.resolve_steps += 1
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(name, seen)
        if name == self.roles.get("search"):
            self.searches += 1
            # listings first met through the follower's searches are no samples of the requested items' shelves
            if not self.__dict__.get("_plain"):
                self.ref_rows |= set(self.rows) - before
        return seen

    def search_with(self, query: str, more: dict[str, Any] | None = None, kind: str = "") -> Any:
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        if not tool or not self.search_key or not query.strip():
            return None
        props = schema.get("properties") or {}
        fixed = {k: v for k, v in (self.brief.get("search_args") or {}).items() if k in props and k != self.search_key}
        args: dict[str, Any] = {**fixed, self.search_key: query.strip(), **(more or {})}
        kind_words = {w for w in _stem(kind) if len(w) >= 3}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = props.get(key) or {}
            options = spec.get("enum")
            if isinstance(options, list) and options:
                names = [o for o in options if isinstance(o, str)]
                # a category argument: the option that is the kind of product; a mode argument: the option that searches by words
                by_kind = [o for o in names if kind_words and _stem(o) & kind_words]
                by_mode = [o for o in names if re.search(r"text|word|keyword|phrase|query|search|find|term|match", o, re.I)]
                args[key] = (by_kind or by_mode or options)[0]
            elif "boolean" in _kinds(spec):
                args[key] = False
            elif "integer" in _kinds(spec) or "number" in _kinds(spec):
                args[key] = int(spec["minimum"]) if isinstance(spec.get("minimum"), (int, float)) else 1
        return self.rcall(tool, args)

    def next_rows(self, query: str, seen: Any, kind: str = "") -> Any:
        """The next page of a search the follower ran; None on the last page."""
        tool = self.roles.get("search")
        plan = self.paging(tool, seen) if tool else None
        if not plan or plan[3] is None or plan[0] <= plan[1]:
            return None
        return self.search_with(query, {plan[2]: plan[3]}, kind)

    @_guarded(lambda self, target, level: [])
    def ladder(self, target: dict[str, Any], level: int) -> list[str]:
        """Searches that should turn up a described listing, the most exact first. A store's search matches a listing's title, shop
        and property values, so the shop's name with the kind and the bare values narrows the shop's listings to the one. A number and
        its unit may be spelled together or apart in the listing, and one value the listing's own text does not carry stops the whole
        search, so the later rungs leave a value out. Queries stay short (stores read only the first words)."""
        seller = [w for w in re.findall(r"[A-Za-z0-9]+", target["seller"]) if len(w) > 1 or w.isdigit()][:3]
        kind = [w for w in re.findall(r"[A-Za-z0-9]+", target["kind"]) if not re.fullmatch(r"(?i)products?|listings?|items?|pages?|the|a|an", w)][:4]
        values = [re.findall(r"[A-Za-z0-9]+", v) for v in (target.get("values") or [])]
        if not any(values):
            # no bare values given: the words of the described properties
            spoken: list[str] = []
            for attribute in target["attributes"]:
                for w in re.findall(r"[A-Za-z0-9]+", attribute):
                    if w.lower() not in [x.lower() for x in spoken]:
                        spoken.append(w)
            values = [spoken[:4]] if spoken else []
        everything = [w for v in values for w in v]

        def words(*groups: list[str]) -> str:
            seen: list[str] = []
            for group in groups:
                for w in group:
                    if w.lower() not in [x.lower() for x in seen]:
                        seen.append(w)
            return " ".join(seen[:8])

        def spelled(parts: list[str]) -> list[list[str]]:
            # "16 gb" also as "16gb" and as "16"
            for i in range(len(parts) - 1):
                if re.fullmatch(r"\d+", parts[i]) and re.fullmatch(r"[A-Za-z]+", parts[i + 1]):
                    return [parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:], parts[:i + 1] + parts[i + 2:]]
            return []

        other = kind[:-1] + [kind[-1][:-1] if kind[-1].lower().endswith("s") and len(kind[-1]) > 3 else kind[-1] + "s"] if kind else []
        if level == 0:
            rungs = [words(seller, kind, everything[:6])] + [words(seller, kind, v[:6]) for v in spelled(everything)[:1]] + [words(seller, kind)]
        elif level == 1:
            rungs = [words(seller, other)] if other else []
            rungs += [words(seller, kind, v[:6]) for v in spelled(everything)[1:]]
            if len(values) >= 2:
                rungs += [words(seller, kind, [w for j, v in enumerate(values) if j != i for w in v][:6]) for i in range(len(values))]
            rungs += [words(seller, v) for v in values] + [words(seller, everything[:6])]
        else:
            rungs = [words(seller or kind), words(kind, everything[:4])]
        out: list[str] = []
        for query in rungs:
            if query and query not in out:
                out.append(query)
        return out[:5]

    @_guarded(lambda self, rows, target, strict=True: [i for i, _ in rows])
    def rank_rows(self, rows: list[tuple[str, dict[str, Any]]], target: dict[str, Any], strict: bool = True) -> list[str]:
        """The rows of a search for a described listing, the likeliest first: the shop named in the row, then the kind of product,
        then the described values. A row that does not name the shop is another shop's listing: when the shop is known and no row names
        it, nothing is worth opening (strict), else the rows are tried as they come."""
        seller = " ".join(w for w in re.findall(r"[A-Za-z0-9]+", target["seller"]) if len(w) > 1 or w.isdigit()).lower()
        kind_words = {w for w in _stem(target["kind"]) if len(w) >= 3}
        wanted: set[str] = set()
        for attribute in (target.get("values") or target["attributes"]):
            tokens = _tokens(attribute)
            wanted |= {w for w in tokens if len(w) >= 2}
            # a value and its unit may be written together in a listing ("16 gb", "16gb")
            wanted |= {x + y for x, y in zip(tokens, tokens[1:])}
        scored = []
        try:
            frags = self.frags_for(target)[0]
        except Exception as exc:     # the description's fragments only refine the order of the rows
            self.trace.append({"kind": "guarded", "fn": "rank_rows.frags", "error": f"{type(exc).__name__}: {exc}"[:160]})
            frags = []
        for index, (ident, record) in enumerate(rows):
            text = " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)).lower()
            shop = 1 if seller and self.shop_in(seller, record) else 0
            tb = bag(text)
            hits = sum(1 for f in frags if frag_level(f, tb) >= 1)
            scored.append((shop, len(kind_words & _stem(text)), hits, len(wanted & set(_tokens(text))), index, ident))
        scored.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3], t[4]))
        if seller:
            named = [t for t in scored if t[0]]
            if named:
                scored = named
            elif strict:
                return []
        return [t[5] for t in scored]

    @staticmethod
    def shop_in(seller: str, record: dict[str, Any]) -> bool:
        """Whether a listing record names the shop (whole words: Kozenan is not Kozenanmo); the record's own shop field when it has one."""
        if not seller:
            return False
        field = Kyo.page_parts(record)["seller"]
        hay = (field or " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))).lower()
        return re.search(r"(?<![a-z0-9])" + re.escape(seller) + r"(?![a-z0-9])", hay) is not None or (
            not field and re.search(r"(?<![a-z0-9])" + re.escape(seller.split()[0]) + r"(?![a-z0-9])", hay) is not None)

    @staticmethod
    def names_shop(rows: list[tuple[str, dict[str, Any]]], target: dict[str, Any]) -> bool:
        """Whether any row of a search names the shop a described listing belongs to."""
        seller = " ".join(w for w in re.findall(r"[A-Za-z0-9]+", target["seller"]) if len(w) > 1 or w.isdigit()).lower()
        if not seller:
            return False
        return any(Kyo.shop_in(seller, record) for _, record in rows)

    def read_page(self, chain: dict[str, Any], ident: str) -> dict[str, Any] | None:
        """Have a matched page read: what it states for the property, where its note sends the reader, the store's pairing rule."""
        record = self.record_of(ident)
        prompt = PAGE_READ + f"\n\nPROPERTY: {chain['prop']}\nRELATION: {chain['relation']}\n\nPAGE ID: {ident}\nPAGE RECORD:\n{record}"
        got = self.ask_json(prompt, [LIGHT], 900, quiet=True)
        if got is None:
            return None
        value, words, rule = _text(got.get("value")), _text(got.get("value_words")), _text(got.get("rule"))
        if value:
            # a value must lie in the page's own words, and those in the record; the pairing rule is no page value
            if (words and (_flat(words) not in _flat(record) or _flat(value) not in _flat(words)
                           or (rule and (_flat(words) in _flat(rule) or _flat(rule) in _flat(words))))) or (not words and _flat(value) not in _flat(record)):
                value = ""
        return {"value": value, "next": _target(got.get("next")) if got.get("next") else None, "rule": rule, "partner": _text(got.get("partner"))}

    def other_view(self, tool: str) -> str | None:
        """The store's other single-listing view: a tool that takes one listing id and is no cart, order, search or message tool."""
        used = {self.roles.get(r) for r in ("cart", "buy", "refuse", "ask", "search", "cart_view", "compare")} - {None}
        return next((t for t in self.schemas if t != tool and t not in used and self.single_target(t)), None)

    @staticmethod
    def sheet_only(seen: Any) -> bool:
        """A single-listing result that is an attributes sheet and nothing else: a facts block with at most a few plain fields (an id, a name)."""
        if not isinstance(seen, dict) or not _fact_block(seen):
            return False
        return len([v for v in seen.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]) <= 4

    def read_sheet(self, ident: str) -> None:
        """Read a listing's attributes sheet (one store step) for the follower."""
        spent = self.step
        self.fill_facts([ident], 8)
        self.resolve_steps += self.step - spent

    def aisle_labels(self) -> list[str]:
        """The names of the aisles the store files its listings under, read from the rows seen so far."""
        key = self.category_key()
        if not key:
            return []
        return sorted({v.strip() for r in self.rows.values() for v in [r.get(key)] if isinstance(v, str) and v.strip()})

    @staticmethod
    def page_parts(rec: dict[str, Any]) -> dict[str, Any]:
        """The parts of a listing's record the matcher reads: the seller's note, the spec sheet, title, description, shop and aisle. The keys
        the light model named for this store (PART_KEYS) come first; a key it did not name is read by its name."""
        parts: dict[str, Any] = {"note": "", "facts": _fact_block(rec) or {}, "title": "", "overview": "", "seller": "", "kind": ""}
        named = {k for k in PART_KEYS.values() if k}
        for part, key in PART_KEYS.items():
            if key and isinstance(rec.get(key), str) and rec[key]:
                parts[part] = rec[key]
        for key, value in rec.items():
            if not isinstance(value, str) or key in named:
                continue
            k = str(key).lower()
            if re.search(r"note|print|remark|comment|memo", k):
                if len(value) > len(parts["note"]):
                    parts["note"] = value
            elif re.search(r"seller|shop|store|vendor", k):
                parts["seller"] = parts["seller"] or value
            elif re.search(r"maker|brand", k):
                continue
            elif re.search(r"categ|aisle|kind|shelf", k):
                parts["kind"] = parts["kind"] or value
            elif re.search(r"title|name|headline", k):
                parts["title"] = parts["title"] or value
            elif re.search(r"about|desc|overview|summary", k):
                parts["overview"] = parts["overview"] or value
        if not parts["kind"] and _KIND_KEY:
            hinted = rec.get(_KIND_KEY[0])
            if isinstance(hinted, str):
                parts["kind"] = hinted
        return parts

    def clause_of(self, need: str, target: dict[str, Any]) -> str:
        """The words that list the pointed-at listing's properties, as the shopper wrote them: found in the requirement, the shopper's
        replies or the request by the wording shoppers use for it ("from X's listing, with a, b and c" and a closing bracket);
        else the model's quote when it lies in the text; else the model's attributes."""
        texts = [need] + list(self.replies[-6:]) + [self.request]
        pat = re.compile(r"(?:(?:page|listing|offer)\s+of\s+the\s+(.+?)\s+)?(?:from|by)\s+([^\s,]+?)(?:['\u2019]s)?\s+(?:listing|page|offer)s?\b,?\s*(?:with\s+|whose\s+|that\s+has\s+)?(.{1,300}?)\s*(?:\(\s*or\s+wherever|\bor\s+wherever|\(\s*or\b|[?]|$)", re.S)
        want = _shop(target.get("seller") or "").lower()
        first = ""
        for text in texts:
            for m in pat.finditer(str(text)):
                clause = re.sub(r"\s+", " ", m.group(3)).strip(" ,.;")
                if m.group(1) and len(m.group(1)) <= 60:
                    # the kind as the shopper wrote it ("control arms"): the model tends to turn it into the singular, which the store's search does not match
                    target["kind"] = re.sub(r"\s+", " ", m.group(1)).strip()
                if want and _shop(m.group(2)).lower() == want:
                    return clause
                first = first or clause
        if first and not want:
            return first
        quote = (target.get("clause") or "").strip()
        if quote and any(_flat(quote) in _flat(t) for t in texts):
            return quote
        return ", ".join(target.get("attributes") or [])

    def frags_for(self, target: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        """The properties that pick a described listing out, as matchable fragments (the kind's own name left out), and the aisle
        label the description names (cached per description until more rows have shown more aisle names)."""
        clause = target.get("clause") or ", ".join(target.get("attributes") or [])
        kind = target.get("kind") or ""
        sig = (clause, kind, len(self.rows) // 40, len(self.full))
        cache = self.__dict__.setdefault("_frag_cache", {})
        if sig not in cache:
            try:
                KEY_PHRASES[:] = self.key_phrases()
            except Exception:
                pass
            # a note that names the listing by its kind alone ("the Shidorko gear oil Transmission Fluids multipack") has its properties in the kind phrase
            made = descriptor(clause, self.aisle_labels(), kind) if clause.strip() else descriptor(kind, self.aisle_labels(), "")
            if not made[0] and clause.strip():
                made = descriptor(clause, self.aisle_labels(), "")     # the clause was nothing but the kind's words: they describe the listing
            cache[sig] = made
        return cache[sig]

    def key_phrases(self) -> list[tuple[str, ...]]:
        """The attribute names (one to three words) that the listings read so far carry, longest first."""
        found: set[tuple[str, ...]] = set()
        for rec in list(self.full.values())[:400]:
            for key in (_fact_block(rec) or {}):
                ws = tuple(words(str(key).replace("_", " ")))
                if 1 <= len(ws) <= 3:
                    found.add(ws)
        return sorted(found, key=lambda p: (-len(p), p))

    def frags_of(self, chain: dict[str, Any]) -> list[dict[str, Any]]:
        frags, label = self.frags_for(chain["target"])
        chain["_label"] = label
        return frags

    @staticmethod
    def bare_values(clause: str) -> list[str]:
        """The values in a descriptor ("light bulb colour blue, power source usb" -> "blue", "usb"): what follows the last property-name
        word of each fragment; a fragment with no such word is kept whole."""
        out: list[str] = []
        for frag in fragments(unparen(clause)):
            toks = [t for t in re.findall(r"[A-Za-z0-9.]+(?:-[A-Za-z0-9]+)*", frag["text"]) if t.lower() not in STOP]
            last = max([i for i, w in enumerate(toks) if w.lower() in GENERIC] or [-1])
            val = toks[last + 1:] if 0 <= last < len(toks) - 1 else toks
            if val:
                out.append(" ".join(val))
        return out

    def match_page(self, chain: dict[str, Any], ident: str) -> dict[str, Any]:
        """A page's match against the description; a page the matcher cannot read is simply not the listing."""
        try:
            return self._match_page(chain, ident)
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "match_page", "error": f"{type(exc).__name__}: {exc}"[:200]})
            blank = {"n": 0, "note": 0, "facts": 0, "soft": 0, "title": 0, "no": 0, "maybe": 0, "unseen": 0, "has_note": False, "answers": False,
                     "detail": ["error"], "full": False, "maybe_full": False, "key": (False, False, False, False, 0, 0, 0, False)}
            return {"ok": False, "full": False, "ask": False, "judge": blank, "sheet": False,
                    "parts": {"note": "", "facts": {}, "title": "", "overview": "", "seller": "", "kind": ""}}

    def _match_page(self, chain: dict[str, Any], ident: str) -> dict[str, Any]:
        """Whether a page is the listing the chain looks for: its shop and aisle are the described ones and the properties the pointer
        lists are stated by the page (note, spec sheet or title) with none ruled out by it. `ask` = the spec sheet could settle it."""
        rec = self.full.get(ident) or self.rows.get(ident) or {}
        parts = self.page_parts(rec)
        frags = self.frags_of(chain)
        t = chain["target"]
        seller = " ".join(w for w in re.findall(r"[A-Za-z0-9]+", t.get("seller") or "") if len(w) > 1 or w.isdigit()).lower()
        text = " ".join(str(v) for v in rec.values() if isinstance(v, str)).lower()
        shop_ok = (not seller) or self.shop_in(seller, rec)
        label = chain.get("_label") or ""
        kind_ok = True
        if label and parts["kind"]:
            kind_ok = bool(set(words(label)) & set(words(parts["kind"]))) or _flat(label) == _flat(parts["kind"])
        j = judge(frags, parts["note"], parts["facts"], parts["title"], parts["overview"], chain["prop"], parts["kind"])
        has_sheet = bool(parts["facts"])
        ok = shop_ok and kind_ok and j["key"][0]
        full = ok and (j["full"] if frags else j["answers"])
        # a page whose note restates the description is settled by the note; the spec sheet is read for a page with no note (a silent
        # kind of look-alike has none), a page the note half confirms, or one whose title shows part of the description
        n = max(1, j["n"])
        open_ = (not parts["note"]) or j["note"] * 2 >= n or j["title"] > 0 or j["soft"] > 0
        return {"ok": ok, "full": full, "ask": ok and not full and not has_sheet and open_, "parts": parts, "judge": j, "sheet": has_sheet}

    def known_sellers(self) -> set[str]:
        """The shops the rows seen so far name."""
        found: set[str] = set()
        for record in list(self.rows.values())[-400:]:
            name = self.page_parts(record)["seller"].strip()
            if name:
                found.add(name)
        return found

    def pointer_of(self, chain: dict[str, Any], parts: dict[str, Any], read: dict[str, Any] | None) -> dict[str, Any] | None:
        """Where a matched page's note sends the reader for the property: the sentence of the note that names the property and points
        (found by code), its shop (the reader's pick when the sentence holds it, else the first shop-like name after the pointing words)
        and the words that describe the other listing (taken from the sentence, so none is lost or reworded)."""
        note, prop = parts["note"], chain["prop"]
        nx = (read or {}).get("next")
        sentence = pointer_sentence(note, prop)
        if not sentence and nx and nx.get("seller"):
            sentence = next((x for x in sentences(note) if re.search(r"\b" + re.escape(nx["seller"]) + r"\b", x, re.I)), "")
            pw = [w for w in words(prop) if w != "type"] or words(prop)
            if sentence and not POINT_CUE.search(sentence) and not (pw and sum(1 for w in pw if w in set(words(sentence))) * 2 >= len(pw)):
                sentence = ""
        if not sentence:
            return None
        skip = frozenset(w for lab in self.aisle_labels() for w in words(lab)) | frozenset(words(prop))
        known = frozenset(self.known_sellers())
        tgt = pointer_target(sentence, (nx or {}).get("seller") or "", parts["seller"], skip, known)
        if not tgt["seller"]:
            return None
        clause = tgt["clause"]
        kind_text = tgt["kind"] or (nx or {}).get("kind") or ""
        if not clause and not kind_text:
            clause = (nx or {}).get("clause") or ""
        plain, label = strip_kind(unparen(clause), self.aisle_labels() + ([kind_text] if kind_text else []))
        plain = re.sub(r"\s+([,;])", r"\1", plain).strip(" ,;.")
        while True:
            q = re.sub(r"^(?:the|a|an|product|products|listing|listings|item|items|from|by|of|which|that|who|is|are|has|have|with|,)\b[\s,]*", "", plain, flags=re.I)
            if q == plain:
                break
            plain = q
        if not plain:
            # the kind phrase carries the properties: its aisle label is the kind, the rest of its words describe the listing
            plain, label2 = strip_kind(kind_text, self.aisle_labels())
            label = label or label2
        kind = label or kind_text
        return {"seller": _shop(tgt["seller"]), "kind": kind, "attributes": [], "values": self.bare_values(plain), "clause": plain}

    def judge_page(self, chain: dict[str, Any], ident: str, deep: bool = False, forced: bool = False) -> dict[str, Any] | None:
        """Open a candidate page with the page tool (the view the order's process check counts), match it against the pointer's
        description by code, and when it is the listing have it read. A first look (not deep) leaves the spec sheet unread: a page
        whose note does not settle the description waits as pending, and only the likeliest pending pages are read in full."""
        key = _dump(chain["target"]).lower() + "|" + ident
        old = chain["verdicts"].get(key)
        if old is not None and not (deep and old.get("pending")) and not forced:
            return old
        tool = self.page_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return None
        if ident not in self.page_seen:
            seen = self.rcall(tool, {target: ident})
            if seen is None:
                return None
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            other = self.other_view(tool) if self.sheet_only(seen) else None
            if other:
                # the brief's labels are turned round or one is missing: the view that came back is the attributes sheet, so the page
                # is the store's other single-listing view
                self.roles["open"], self.roles["offer"] = tool, other
                self.trace.append({"kind": "page_view_repaired", "sheet": tool, "page": other})
                tool, target = other, self.single_target(other)
                seen = self.rcall(tool, {target: ident}) if target else None
                if seen is None:
                    return None
                self.note_view(tool, seen)
                if isinstance(seen, dict) and self.ident(seen):
                    self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
        facts = self.facts_tool() or self.other_view(tool)
        got = self.match_page(chain, ident)
        sig = _dump(chain["target"]).lower()
        chain["cands"].setdefault(sig, {})[ident] = (got["judge"]["key"], got["ok"] and got["judge"]["maybe_full"])
        verdict: dict[str, Any] = {"is_target": False, "value": "", "unconfirmed": [], "next": None, "rule": "", "partner": ""}
        if got["ask"] and facts and facts != tool and ident not in self.full and self.can_resolve():
            if not deep:
                j = got["judge"]
                verdict["pending"] = True
                chain["pending"].setdefault(_dump(chain["target"]).lower(), {})[ident] = (j["note"] + j["facts"] + j["soft"] + j["title"], bool(got["parts"]["note"]))
                chain["verdicts"][key] = verdict
                return verdict
            # the note leaves a property open: the spec sheet may state it (one step)
            self.read_sheet(ident)
            got = self.match_page(chain, ident)
            chain["cands"].setdefault(sig, {})[ident] = (got["judge"]["key"], got["ok"] and got["judge"]["maybe_full"])
        if got["full"] or (forced and got["ok"]):
            verdict = self.read_matched(chain, ident, got, facts, tool)
        self.trace.append({"kind": "page_match", "id": ident, "ok": got["ok"], "full": got["full"], "detail": got["judge"]["detail"][:6],
                           "target": verdict["is_target"], "value": verdict["value"][:40], "next": (verdict["next"] or {}).get("seller", "")})
        chain["verdicts"][key] = verdict
        if verdict["is_target"]:
            self.ref_pages.add(ident)    # a verified page of a chain: a reference, not a sample of any shelf
        return verdict

    def read_matched(self, chain: dict[str, Any], ident: str, got: dict[str, Any], facts: str | None, tool: str) -> dict[str, Any]:
        """A page that matched: where it sends the reader, or the value it states (the spec sheet's own figure first)."""
        parts = got["parts"]
        read = self.read_page(chain, ident)
        verdict: dict[str, Any] = {"is_target": True, "value": "", "unconfirmed": [], "next": None, "rule": "", "partner": ""}
        pointer = self.pointer_of(chain, parts, read)
        if not pointer and read and read.get("next") and read["next"].get("seller"):
            # the reader sees a pointer that the sentence parse did not: it counts when the note names that shop (not this page's own)
            nx = read["next"]
            name = _shop(nx["seller"])
            if name and name.lower() != (parts["seller"] or "").strip().lower() and re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", parts["note"] or ""):
                clause = nx.get("clause") or ""
                pointer = {"seller": name, "kind": nx.get("kind") or "", "attributes": [], "values": self.bare_values(clause), "clause": clause}
        if pointer:
            verdict["next"] = pointer
            return verdict
        value = fact_value(parts["facts"], chain["prop"]) if parts["facts"] else ""
        if not value and not parts["facts"] and facts and facts != tool and ident not in self.full and self.can_resolve():
            self.read_sheet(ident)
            parts = self.page_parts(self.full.get(ident) or self.rows.get(ident) or {})
            value = fact_value(parts["facts"], chain["prop"]) if parts["facts"] else ""
        if not value and read:
            value = read["value"]
        verdict["value"] = value
        verdict["rule"] = store_rule(parts["note"]) or (read or {}).get("rule", "")
        verdict["partner"] = (read or {}).get("partner", "")
        if not value and not pointer:
            # the listing matches the description but says nothing for the property: a look-alike of a silent kind
            verdict["is_target"] = False
        return verdict

    def find_target(self, chain: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """Search for the listing the chain looks for, open the likeliest rows' pages and have each read, until one is it."""
        target = chain["target"]
        sig = _dump(target).lower()
        tried: set[str] = chain["tried"].setdefault(sig, set())
        chain.setdefault("names", ())
        done_queries: list[str] = chain["queries"].setdefault(sig, [])
        limit = RESOLVE_OPENS * (1 + chain["level"])
        kind = target["kind"]

        def room() -> bool:
            return (self.can_resolve() and self.resolve_steps < chain["limit"] and chain["state"] == "open"
                    and self.resolve_steps < chain["hop_start"] + HOP_STEPS)

        for query in self.ladder(target, chain["level"]):
            if len(tried) >= limit:
                break
            if query in done_queries or not room():
                continue
            done_queries.append(query)
            seen = self.search_with(query, None, kind)
            pages = 0
            while seen is not None and pages < 3:
                if self.names_shop(self.records_in(seen), target):
                    chain["named"] = True
                for ident in [i for i in self.rank_rows(self.records_in(seen), target, chain["level"] < 2) if i not in tried and i not in chain["pages"]]:
                    if len(tried) >= limit or not room():
                        break
                    tried.add(ident)
                    verdict = self.judge_page(chain, ident)
                    if verdict is None and ident not in chain["retried"]:
                        chain["retried"].add(ident)
                        tried.discard(ident)
                    # a search that lists one row only for the description (the store matches every word in its attributes) has that hit's
                    # spec sheet read at once: the offer page alone does not show the attributes, and a note that only points on says
                    # nothing of the page's own description
                    if (verdict and not verdict["is_target"] and pages == 0 and len(self.records_in(seen)) == 1 and room()
                            and ident not in self.full and self.match_page(chain, ident)["ok"]):
                        self.read_sheet(ident)
                        chain["verdicts"].pop(_dump(chain["target"]).lower() + "|" + ident, None)
                        verdict = self.judge_page(chain, ident)
                    if verdict and verdict["is_target"]:
                        if chain["relation"] == "partner" and verdict["value"] and not verdict["next"] and not verdict["partner"] and not verdict["rule"]:
                            # the store's pairing rule is on the last page: this one carries none (kept in case no page does)
                            chain["fallback"] = chain["fallback"] or (ident, verdict)
                            continue
                        return ident, verdict
                if len(tried) >= limit or not room():
                    break
                seen = self.next_rows(query, seen, kind)
                pages += 1
        # no first look settled it: read the spec sheets of the likeliest pages whose notes left the description open
        pend = {i: v for i, v in chain["pending"].get(sig, {}).items()
                if (chain["verdicts"].get(_dump(chain["target"]).lower() + "|" + i) or {}).get("pending")}
        for ident, _ in sorted(pend.items(), key=lambda kv: (-kv[1][0], not kv[1][1]))[:4]:
            if not room():
                break
            verdict = self.judge_page(chain, ident, deep=True)
            if verdict and verdict["is_target"]:
                if chain["relation"] == "partner" and verdict["value"] and not verdict["next"] and not verdict["partner"] and not verdict["rule"]:
                    chain["fallback"] = chain["fallback"] or (ident, verdict)
                    continue
                return ident, verdict
        # the listings read meanwhile may have shown attribute names that split the description differently: the candidates already read
        # are judged again (no store step is needed for a page that was read)
        try:
            names = tuple(self.key_phrases())
        except Exception:
            names = ()
        if names != chain.get("names") and tried:
            chain["names"] = names
            KEY_PHRASES[:] = list(names)
            self._frag_cache = {}
            for ident in sorted(tried):
                key = _dump(chain["target"]).lower() + "|" + ident
                old = chain["verdicts"].get(key)
                if old is not None and not old.get("is_target") and not old.get("pending") and ident in self.page_seen and room():
                    chain["verdicts"].pop(key, None)
                    verdict = self.judge_page(chain, ident)
                    if verdict and verdict["is_target"]:
                        if chain["relation"] == "partner" and verdict["value"] and not verdict["next"] and not verdict["partner"] and not verdict["rule"]:
                            chain["fallback"] = chain["fallback"] or (ident, verdict)
                            continue
                        return ident, verdict
        # a description that gives a figure in another unit than the page does cannot be settled by code: the model picks among the
        # candidates that agree in everything else
        pick = self.adjudicate(chain, sig) if room() else ""
        if pick:
            verdict = self.judge_page(chain, pick, deep=True, forced=True)
            if verdict and verdict["is_target"]:
                return pick, verdict
        return None

    def adjudicate(self, chain: dict[str, Any], sig: str) -> str:
        """The candidate of a hop that the description picks out when the page writes a figure in another unit; '' when none."""
        if sig in chain["adjudged"]:
            return ""
        near = sorted(((key, ident) for ident, (key, maybe) in chain["cands"].get(sig, {}).items() if maybe), reverse=True)[:4]
        if not near or self.resolve_calls >= RESOLVE_CALLS - 2:
            return ""
        chain["adjudged"].append(sig)
        target = chain["target"]
        prompt = (ADJUDICATE + f"\n\nDESCRIPTION: the shop {target.get('seller') or '(any)'}, kind {target.get('kind') or '(any)'}, properties: "
                  + (target.get("clause") or ", ".join(target.get("attributes") or [])) + "\n\nCANDIDATES:\n"
                  + "\n".join(f"{i}: {self.record_of(i)}" for _, i in near))
        got = self.ask_json(prompt, [LIGHT], 400, quiet=True)
        pick = _text((got or {}).get("pick"))
        self.trace.append({"kind": "adjudicate", "candidates": [i for _, i in near], "pick": pick})
        return pick if pick in {i for _, i in near} else ""

    def confirm_partner(self, chain: dict[str, Any], verdict: dict[str, Any]) -> str:
        """The value a pairing rule names for the page's value, read once more by the heavy model: the direction of a rule is easy to turn round."""
        if not verdict["rule"] or not verdict["value"]:
            return verdict["partner"]
        prompt = PARTNER + f"\n\nPROPERTY: {chain['prop']}\nVALUE ON THE PAGE: {verdict['value']}\nRULE AS QUOTED: {verdict['rule']}"
        got = self.ask_json(prompt, [HEAVY, LIGHT], 900)
        return _text((got or {}).get("partner")) or verdict["partner"]

    def end_chain(self, chain: dict[str, Any], verdict: dict[str, Any]) -> str:
        """The chain ended on a page that states the value."""
        chain["value"], chain["rule"], chain["partner"] = verdict["value"], verdict["rule"], verdict["partner"]
        if chain["relation"] == "partner":
            chain["partner"] = self.confirm_partner(chain, verdict)
        chain["state"] = "done"
        return self.apply_chain(chain)

    def advance_chain(self, chain: dict[str, Any]) -> str:
        """Take one chain as far as it goes: read the pointer, then page after page until a page states the value."""
        if chain["state"] != "open":
            return ""
        if chain["desc"] is None:
            if chain["desc_tries"] >= 2:
                chain["state"] = "failed"
                return ""
            chain["desc_tries"] += 1
            prompt = DESCRIBE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENT:\n" + chain["need"]
            if self.replies:
                prompt += "\n\nTHE SHOPPER'S REPLIES (the requirement may come from one of them; its comparison words are there):\n" + "\n".join(r[:600] for r in self.replies[-6:])
            got = self.ask_json(prompt, [LIGHT, HEAVY], 700, quiet=True)
            if got is None:
                # a failed call is not an answer: the next pass asks again
                return ""
            target = _target(got)
            if got.get("pointer") not in (True, "true", "True") or not target:
                chain["state"] = "failed"
                self.trace.append({"kind": "resolve_failed", "need": chain["need"][:160], "why": "not read as a pointer"})
                return ""
            target["clause"] = self.clause_of(chain["need"], target)
            if not target["values"]:
                target["values"] = self.bare_values(target["clause"])
            relation = _text(got.get("relation")).lower()
            chain["desc"] = got
            chain["target"] = target
            chain["relation"] = relation if relation in ("same", "partner", "differ", "at_least", "at_most", "near") else "same"
            # the reader sometimes calls every comparison "same": the requirement's own words decide when they show a clear one
            cue = _relation_cue(chain["need"])
            if cue and chain["relation"] == "same":
                self.trace.append({"kind": "relation_cue", "need": chain["need"][:100], "reader": chain["relation"], "words": cue})
                chain["relation"] = cue
            chain["tolerance"] = _text(got.get("tolerance"))
            chain["prop"] = _text(got.get("property")) or chain["need"].split(":", 1)[0].strip()
        while chain["state"] == "open" and self.can_resolve() and self.resolve_steps < chain["limit"]:
            if len(chain["pages"]) >= RESOLVE_HOPS:
                chain["state"] = "failed"
                break
            if chain["need"] not in {n for _, n in self.needs_now()}:
                break    # the requirement was withdrawn while the follower ran
            found = self.find_target(chain)
            if found is None:
                if not self.can_resolve() or self.resolve_steps >= chain["limit"] or chain["state"] != "open":
                    break
                chain["level"] += 1
                if chain["level"] >= 2 and chain["target"]["seller"] and not chain["named"]:
                    # no search has shown a single listing of that shop: it is not to be found here, and more searches only spend steps
                    chain["state"] = "failed"
                    break
                if chain["level"] > 2:
                    if chain["fallback"]:
                        # no page gave the pairing rule: the last page that stated a value ends the chain
                        ident, verdict = chain["fallback"]
                        chain["pages"].append(ident)
                        return self.end_chain(chain, verdict)
                    chain["state"] = "failed"
                continue
            ident, verdict = found
            if ident in chain["pages"]:
                chain["state"] = "failed"    # the pages lead round in a circle
                break
            chain["pages"].append(ident)
            self.trace.append({"kind": "resolve_page", "id": ident, "value": verdict["value"][:60], "next": verdict["next"], "target": chain["target"]})
            if verdict["value"] and not verdict["next"]:
                return self.end_chain(chain, verdict)
            if verdict["next"] and _dump(verdict["next"]).lower() != _dump(chain["target"]).lower():
                chain["target"] = verdict["next"]
                chain["level"] = 0
                chain["fallback"] = None
                chain["hop_start"] = self.resolve_steps
                chain["named"] = False
                continue
            chain["state"] = "failed"
        if chain["state"] == "failed":
            self.trace.append({"kind": "resolve_failed", "need": chain["need"][:160], "why": "no way on", "pages": chain["pages"]})
            if chain["pages"]:
                chain["progress"] = ("pages opened in full: " + " -> ".join(chain["pages"])
                                     + (f"; the last one sends you to {_dump(chain['target'])}" if chain["target"] else ""))
        return ""

    def apply_chain(self, chain: dict[str, Any]) -> str:
        """The chain ended on a value: write the plain requirement in place of the pointer."""
        try:
            return self.write_chain(chain)
        except Exception as exc:
            chain["state"] = "failed"
            self.trace.append({"kind": "resolve_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def write_chain(self, chain: dict[str, Any]) -> str:
        need, relation, value = chain["need"], chain["relation"], chain["value"]
        several = "/" in value or "," in value
        note = ""
        if relation == "partner":
            if chain["partner"]:
                text = chain["partner"]
            else:
                rule = f" ({chain['rule']})" if chain["rule"] else ""
                text = f"the value the store's pairing rule gives for {value}{rule}"
                note = " (the partner could not be read from the pages; look the pairing rule up)"
        elif relation == "differ":
            text = ("none of: " if several else "anything but ") + value
        elif relation == "at_least":
            text = "at least " + value
        elif relation == "at_most":
            text = "at most " + value
        elif relation == "near":
            text = f"within {chain['tolerance']} of {value}" if chain["tolerance"] else f"close to {value}"
        else:
            text = value + (" (the page lists several; a listing that has any one of them counts)" if several else "")
        head = need.split(":", 1)[0].strip()
        prop = head if ":" in need and 0 < len(head.split()) <= 5 else chain["prop"]
        must = f"{prop}: {text}"
        live = False
        for item in self.items:
            if need in item.dropped:
                continue
            for group in (item.musts, item.told):
                for index, existing in enumerate(group):
                    if existing == need:
                        group[index] = must
                        live = True
        self.resolved[need] = must
        self.refs = [(n, nd) for n, nd in self.refs if nd != need]
        self.ref_values[need] = text
        if not live:
            chain["state"] = "done"    # the requirement was withdrawn while the follower ran: nothing to write
            return ""
        self.audit_again = True
        chain["summary"] = f"{must} (read from the pages {' -> '.join(chain['pages'])}, each opened in full){note}"
        self.version += 1
        self.trace.append({"kind": "resolved", "need": need[:160], "must": must, "pages": chain["pages"]})
        rule = f" The last page's pairing rule: {chain['rule'].rstrip('. ')}." if chain["rule"] and relation == "partner" else ""
        return (f"[Kyo followed a reference] {need[:200]} -> pages {' -> '.join(chain['pages'])} (all opened in full).{rule} "
                f"The requirement now reads: {must}{note}. Pick listings that meet it; the pages themselves are references, "
                "and are candidates only when they meet every requirement of their own item too.")

    def resolve_pending(self, stage: str) -> str:
        """Follow every requirement that still points at another listing; returns what to tell the actor ('' when nothing was settled)."""
        lines: list[str] = []
        self.scope = "resolve"
        try:
            pending = self.pending_refs()
            share = max(4, self.resolve_room() // max(1, len(pending)))
            for number, need in pending:
                if not self.can_resolve():
                    break
                chain = self.chains.setdefault(need, {
                    "number": number, "need": need, "state": "open", "desc": None, "target": None, "prop": "", "relation": "same", "tolerance": "",
                    "pages": [], "tried": {}, "queries": {}, "verdicts": {}, "level": 0, "value": "", "rule": "", "partner": "", "summary": "",
                    "blank": 0, "desc_tries": 0, "pending": {}, "cands": {}, "adjudged": [], "fallback": None, "retried": set(), "limit": 0, "hop_start": 0, "progress": "", "told": False, "decline_held": False, "named": False})
                # every chain has its share of the follower's steps in a pass
                chain["limit"] = self.resolve_steps + share
                chain["hop_start"] = self.resolve_steps
                try:
                    line = self.advance_chain(chain)
                except Exception as exc:
                    # the follower is an addition: whatever goes wrong in it must never cost the episode
                    chain["state"] = "failed"
                    self.trace.append({"kind": "resolve_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    line = ""
                if line:
                    lines.append(line)
                elif chain["state"] == "failed" and chain.get("progress") and not chain.get("told"):
                    chain["told"] = True
                    lines.append(f"[Kyo could not finish following] {need[:160]}: {chain['progress']}. Search that shop and kind, open the page in full, "
                                 "follow its note to the next listing, and open every page on the way.")
        finally:
            self.scope = ""
        self.trace.append({"kind": "resolve_pass", "stage": stage, "settled": len(lines), "steps": self.resolve_steps})
        return "\n".join(lines)

    def after_follow(self, text: str) -> str:
        """A line of requirements with the pointers already followed replaced by what they came to."""
        for need, must in self.resolved.items():
            if need in text:
                text = text.replace(need, must)
        return text

    # ---------- model codes ----------
    def codes(self) -> list[str]:
        return sorted({it.fit_code for it in self.items if it.fit_code}, key=lambda c: (-len(c), c))

    @_guarded(lambda self, args: (args, []))
    def strip_codes(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Search words without the shopper's model code (a code is not a searchable word once a search for it found nothing)."""
        out, removed = dict(args), []
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            new, gone = value, []
            for code in self.codes():
                cut = re.sub(r"(?<![A-Za-z0-9])" + _code_re(code) + r"(?![A-Za-z0-9])", " ", new, flags=re.I)
                if cut != new:
                    gone.append(code)
                    new = cut
            new = re.sub(r"\s+", " ", new).strip()
            if new and new != value:
                out[key] = new
                removed += gone
        return out, removed

    def fit_note(self, observation: Any) -> str:
        try:
            return self.fit_lines(observation)
        except Exception as exc:
            self.trace.append({"kind": "fit_note_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def fit_lines(self, observation: Any) -> str:
        """For the listings a result shows: which name the shopper's model code in the first sentence of their text."""
        codes = self.codes()
        rows = self.records_in(observation)[:16] if codes else []
        if not rows:
            return ""
        parts = []
        for code in codes:
            names: list[str] = []
            later: list[str] = []
            absent = 0
            for ident, record in rows:
                state, sentence = _fit_state(self.full.get(ident) or record, code)
                if state == "names":
                    names.append(f"{ident} \"{sentence}\"" if len(names) < 3 else ident)
                elif state == "later":
                    later.append(ident)
                elif state == "no":
                    absent += 1
            parts.append(f"{code}: first sentence names it: {'; '.join(names) or 'none'}"
                         + (f"; named only later in the text: {', '.join(later)}" if later else "") + f"; not named: {absent}")
        return ("\n[fit] " + " | ".join(parts) + " (for items tied to the code only; a sentence that lists exceptions or a family pattern needs your own reading)")

    # ---------- endings without a purchase ----------
    def decline_hold(self) -> str:
        try:
            return self.decline_reasons()
        except Exception as exc:
            self.trace.append({"kind": "decline_gate_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def decline_reasons(self) -> str:
        """Why an ending without a purchase waits now ('' lets it go). Each reason is a step the store allows that has not been taken
        yet (a reference not followed, a change to the task not answered); the holds are few (DECLINE_HOLDS in all) and none comes
        when too few steps are left to act on it."""
        if self.decline_holds >= DECLINE_HOLDS or self.left() <= max(8, self.tail() + 5) or self.elapsed() > CLOCK_SOFT - 20:
            return ""
        before = self.step
        reasons: list[str] = []
        # 1. a requirement that points at another listing has not been followed to its value
        if self.pending_refs() and self.can_resolve():
            report = self.resolve_pending("decline")
            if report:
                reasons.append(report)
        waiting = self.pending_refs()
        if waiting:
            reasons.append("Not followed to its value yet: " + " | ".join(need[:160] for _, need in waiting[:2])
                           + ". Search the shop's listings by the shop's name and the kind of product, open each page in full, follow its note to the next listing.")
        live = {need for _, need in self.needs_now()}
        for chain in self.chains.values():
            if chain["state"] == "failed" and chain.get("progress") and chain["need"] in live and not chain["decline_held"]:
                chain["decline_held"] = True
                reasons.append(f"A reference was not followed to its value: {chain['need'][:160]} ({chain['progress']}). Search that shop and kind, "
                               "open the page in full and follow its note to the next listing.")
        # 2. the task changed (a note, a refusal) and nothing has been searched or carted since
        if self.note_at >= 0 and self.work_at <= self.note_at and self.notes and self.note_held != self.note_at:
            self.note_held = self.note_at
            reasons.append(f"The task changed at step {self.note_at}: {self.notes[-1][:200].rstrip('. ')}. Nothing has been searched or carted under the new state. "
                           "Look for a cart that fits it (a cheaper or other listing, a replacement for a sold-out line) before ending; "
                           "if nothing meets every requirement under the new state, send the ending call again.")
        if not reasons:
            return ""
        self.decline_holds += 1
        used = self.step - before
        head = "Not run yet (no step used). " if used == 0 else f"Not run yet (Kyo used {used} step(s) for you). "
        self.trace.append({"kind": "decline_held", "n": self.decline_holds, "why": [r[:80] for r in reasons]})
        return head + " ".join(reasons) + " The ending waits until this is dealt with."

    # ---------- questions ----------
    def label_owner(self, label: str) -> int | None:
        """item_of_label; else, for an aisle label whose words no item names ("Natural Sweeteners" for "coconut sugar"), the one item no aisle label
        names when this is the biggest aisle no item names. Only the code that reads attribute keys uses it: a reference page's aisle is still no
        item's (set_aside)."""
        found = self.item_of_label(label)
        if found is not None or not label:
            return found
        try:
            cache = self.__dict__.setdefault("_owner_cache", {})
            sig = len(self.rows)
            if cache.get("sig") != sig:
                cat_key = self.category_key()
                count: dict[str, int] = {}
                for ident, rec in self.rows.items():
                    if ident in self.ref_pages or ident in self.ref_rows:
                        continue
                    lab = str(rec.get(cat_key) or "").strip() if cat_key else ""
                    if lab:
                        count[lab] = count.get(lab, 0) + 1
                named = {self.item_of_label(lab) for lab in count} - {None}
                free_items = [n for n in range(1, len(self.items) + 1) if n not in named]
                free_labels = sorted((lab for lab in count if self.item_of_label(lab) is None), key=lambda lab: -count[lab])
                cache.clear()
                cache.update({"sig": sig, "owner": free_items[0] if len(free_items) == 1 else None, "label": free_labels[0] if free_labels else None})
            return cache["owner"] if cache["label"] is not None and label.strip() == cache["label"] else None
        except Exception:
            return None

    def item_of_label(self, label: str) -> int | None:
        """The requested item a category label of the key table belongs to."""
        low, stems = _words(label), {w for w in words(label) if len(w) >= 3}
        if not low:
            return None
        best, best_score = None, 0
        for number, item in enumerate(self.items, 1):
            if low in (_words(item.shelf or ""), _words(item.name)):
                return number
            names = {w for w in set(words(item.shelf or "")) | set(words(item.name)) if len(w) >= 3}
            score = len({w for w in stems if len(w) >= 3} & names)
            if score:
                score += 0.5 * score / len(names)
            if score > best_score:
                best, best_score = number, score
        return best

    def set_aside(self, ident: str) -> bool:
        """Whether a listing was met only while following a reference and is of no kind the shopper asked for (so it is no candidate
        for the review or the swap)."""
        if ident not in self.ref_rows and ident not in self.ref_pages:
            return False
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        return not label or self.item_of_label(label) is None

    def settled_prop(self, number: int, name: str) -> bool:
        """Whether the request or an answer already settles this property for the item."""
        keys = _stem(name)
        for must in self.items[number - 1].musts + self.items[number - 1].told:
            prop = _stem(must.split(":", 1)[0])
            if prop and keys and (keys <= prop or prop <= keys):
                return True
        return False

    def fill_questions(self, ordered: list[tuple[int, str]], table: list[tuple[str, str, int, list[str]]], count: int) -> list[tuple[int, str]]:
        """The planner's picks, topped up from the key table (the keys most listings carry, turns between the items) until
        every message the shopper allows has a question."""
        if len(ordered) >= count or not table:
            return ordered
        queues: dict[int, list[str]] = {}
        for label, name, _, _ in table:
            number = self.label_owner(label)
            if number is not None:
                queues.setdefault(number, []).append(name)
        out = list(ordered)
        taken = {(n, _words(label)) for n, label in out}
        taken |= {(m, key) for (_, key), others in getattr(self, "also", {}).items() for m in others}
        while len(out) < count:
            progressed = False
            for number in sorted(queues, key=lambda n: (sum(1 for m, _ in out if m == n), n)):
                pick = next((name for name in queues[number] if (number, _words(name)) not in taken and not self.settled_prop(number, name)), None)
                if pick:
                    out.append((number, pick))
                    taken.add((number, _words(pick)))
                    progressed = True
                    break
            if not progressed:
                break
        return out

    def refused_message(self, answer: Any) -> bool:
        """Whether the store refused a message to the shopper because the cap is spent."""
        if not isinstance(answer, dict):
            return False
        if self.reply_key:
            node = _reply_node(answer, self.reply_key)
            # the key every answer so far came under carries no words: the store did not pass the message on
            return node is None or not (isinstance(node.get(self.reply_key), str) and len(node[self.reply_key].split()) >= 2)
        # before the first answer only a fault-like key counts: the shopper's own words may well say "too many" or "no more"
        refusal = " ".join(str(v) for k, v in answer.items() if re.search(r"fault|error|issue|refus|reject|denied|quota|limit", str(k), re.I)).lower()
        return bool(re.search(r"budget|exceed|limit|quota|too many|no more|used all|all (?:your|the) messages|out of messages", refusal))

    def learn_reply_key(self, answer: Any) -> None:
        """The key the shopper's words come under (at the top of the answer or one wrapper down), from the first answer that carried any:
        a later answer without words there is a refusal."""
        if self.reply_key or not isinstance(answer, dict):
            return
        texts = [(len(v.split()), k) for k, v in answer.items() if isinstance(v, str) and len(v.split()) >= 2]
        for value in answer.values():
            if isinstance(value, dict):
                texts += [(len(v.split()), k) for k, v in value.items() if isinstance(v, str) and len(v.split()) >= 2]
        if texts:
            self.reply_key = max(texts)[1]

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain the follower gave up on gets two rounds of this older reading at most, and one hold
        failed = {n for n in pending if self.chains.get(n, {}).get("state") == "failed"}
        rounds = 2 if failed and failed == set(pending) else 8
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < rounds and self.cost < self.hard():
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
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < (1 if failed else 3) and self.left() > 5
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
        # a line is read again when its offer was last read before the line was put in, before the last change to the task
        # or refusal, or a good while ago; reading every line at every held checkout spent the steps of five-line baskets
        floor = self.stock_at
        held = [i for i in self.cart_ids() if i not in self.gone]
        # a basket of up to three lines is read as before (every line more than two steps old: a silent sell-out or price change can come
        # at any time); a bigger one is read when something changed for the line or a good while has passed
        age = 2 if len(held) <= 3 else REREAD_AGE
        stale = [i for i in held if self.offer_seen.get(i, -99) < max(floor, self.line_added.get(i, -1)) or self.step - self.offer_seen.get(i, -99) > age]
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

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > self.soft():
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident) or self.set_aside(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1200)
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
        # one review for the task as it stands; it is made again (three times at most) only after a change that can widen the fit set:
        # a budget raised, a requirement dropped, a reference followed to its value
        if self.audits >= 3 or (self.audits >= 1 and not self.audit_again) or self.cost > self.soft():
            return ""
        self.audits += 1
        self.audit_again = False
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        # the listings whose attributes were read come first (an early find must not drop out behind many later search rows), each group
        # cheapest first, at most 30 in all
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i) and not self.set_aside(i)]
        price = lambda i: (self.price_of(i) is None, self.price_of(i) if self.price_of(i) is not None else 0.0)
        opened = sorted([i for i in live if i in self.full], key=price)[:20]
        rest = sorted([i for i in live if i not in self.full], key=price)[:30 - len(opened)]
        records = [("OPENED " + _dump(_clip(self.full[i], 200))[:1200]) for i in opened] + [("ROW " + _dump(_clip(self.rows.get(i), 120))[:500]) for i in rest]
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (opened ones first, each group cheapest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2600)
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
            plan = [(tool, {target: i}) for i in rows[:2]]
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
        add_mode, remove_mode = _cart_modes(props[mode]["enum"]) if mode else (None, None)
        sets = (not self.cart_adds) if self.cart_adds is not None else self.cart_sets(name)
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: remove_mode, **({count: have} if count else {})})
            elif count and sets:
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not sets:
                # the store adds units: the surplus is taken off, the shortfall put in
                if mode and count and have > want:
                    calls.append({target: ident, mode: remove_mode, count: have - want})
                    continue
                if mode and count and have < want:
                    calls.append({target: ident, mode: add_mode, count: want - have})
                    continue
                if count and have > want and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= want - have:
                    calls.append({target: ident, count: want - have})
                    continue
                return None
            calls.append({target: ident, **({mode: add_mode} if mode else {}), **({count: want} if count else {})})
        return calls

    def procedure_kind(self) -> str:
        """The router: the structure of the request decides, never a name. Any item tied to a model code -> "fitment"; two or more items -> "basket";
        a single product keeps the general flow ("")."""
        if not PROCEDURES:
            return ""
        if any(it.fit_code for it in self.items):
            return "fitment"
        if len(self.items) >= 2:
            return "basket"
        return ""

    def line_review(self, ids: list[str]) -> list[str]:
        """The light model reads every cart line against its item and names the listings whose records contradict it; they are vetoed."""
        lines = []
        for ident in ids:
            number = self.line_item(ident)
            if number is None or self.sold_out(ident):
                continue
            item = self.items[number - 1]
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            if not needs:
                continue                  # nothing but the kind, which the store's category settles
            head = f"LINE item {number}: {item.name} x{item.qty}; requirements: " + "; ".join(needs)
            parts = self.page_parts(self.whole_record(ident))
            facts = "; ".join(f"{k}: {v}" for k, v in list((parts["facts"] or {}).items())[:14] if isinstance(v, (str, int, float)))[:500]
            price = self.price_of(ident)
            lines.append(head + f"\nRECORD {ident}: {parts['title'][:160]} | price {price:g}" if price is not None else head + f"\nRECORD {ident}: {parts['title'][:160]}")
            try:
                cat_key = self.category_key()
                label = str((self.full.get(ident) or self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            except Exception:
                label = ""
            lines[-1] += f" | {parts['overview'][:200]} | attributes: {facts}" + (f" | category: {label}" if label else "")
        if not lines:
            return []
        got = self.ask_json(LINEREVIEW + "\n\n" + "\n".join(lines), [LIGHT, HEAVY], 1500) or {}
        bad, ignored = [], []
        for entry in _items(got.get("bad")):
            if not isinstance(entry, dict) or str(entry.get("id")) not in ids:
                continue
            ident = str(entry.get("id"))
            number = self.line_item(ident)
            said = str(entry.get("requirement") or "").strip() or str(entry.get("why") or "")
            # a veto must name one of the line's requirements: the kind of product (the category's call) and the fit (the code's) are not the review's
            if number is not None and self.names_need(number, said):
                bad.append(ident)
            else:
                ignored.append(ident)
        bad = list(dict.fromkeys(bad))
        self.trace.append({"kind": "line_review", "lines": len(lines), "bad": bad, "ignored": ignored,
                           "said": [(str(e.get("id")), str(e.get("requirement") or "")[:60], str(e.get("why") or "")[:100]) for e in _items(got.get("bad")) if isinstance(e, dict)][:4]})
        return bad

    def names_need(self, number: int, text: str) -> bool:
        """Whether a reviewer's words name one of the item's requirements: the requirement itself, or every word of its property's name."""
        item = self.items[number - 1]
        have = set(words(text, drop_stop=False))
        low = _words(text)
        for need in item.musts + item.told:
            if need in item.dropped:
                continue
            if _words(need) and _words(need) in low:
                return True
            prop, _ = split_must(need)
            pw = words(prop)
            if pw and all(_has(w, have) for w in pw):
                return True
        return False

    def run_procedure(self) -> str:
        """Fitment and basket requests: after the interview Kyo carts the verified pick itself, reviews every line, repairs what is contradicted and checks out
        when the independent check passes. '' when the request is not one of those, the steps are too few, a reference is still to be followed, or the
        procedure was used; otherwise what it did, for the actor to continue from."""
        kind = self.procedure_kind()
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if (not kind or self.__dict__.get("_proc_done") or not self.asked or self.finished or not cart or not buy
                or self.left() < 20 or self.elapsed() > CLOCK_SOFT - 60):
            return ""
        if self.pending_refs() and self.can_resolve():
            return ""
        self._proc_done = True
        self._guarding = True
        notes: list[str] = []
        try:
            for round_ in range(PROC_ROUNDS):
                if self.left() < 14 or self.finished:
                    break
                took = self.safe_retake(f"the {kind} procedure", final=True)
                if took:
                    notes.append(took[:240])
                ids = [i for i in self.cart_ids() if not self.sold_out(i)]
                if len(ids) < len(self.items):
                    self.trace.append({"kind": "procedure_gap", "round": round_, "lines": len(ids), "items": len(self.items)})
                    break
                self.reread_offers()
                if any(self.sold_out(i) for i in self.cart_ids()):
                    continue
                bad = self.line_review(ids)
                if not bad:
                    break
                self.vetoed |= set(bad)
                notes.append("review: " + ", ".join(bad) + " contradict their item")
                for ident in bad:
                    number = self.line_item(ident)
                    if number is not None:
                        choice = self.model_choose(number, {ident})
                        if choice:
                            self.__dict__.setdefault("model_picks", {})[number] = choice
        finally:
            self._guarding = False
        ids = [i for i in self.cart_ids() if not self.sold_out(i)]
        self.trace.append({"kind": "procedure", "route": kind, "lines": len(ids), "items": len(self.items)})
        if len(ids) < len(self.items) or self.finished or self.left() < 6:
            return ("Kyo carried the request out as far as it could (" + "; ".join(notes) + "). Cart now: " + self.cart_text()) if notes else ""
        self.fill_facts(ids, 3)
        self.check_cart(ids, self.hard())
        if not self.verdicts.get(self.check_key(ids)):
            return "Kyo set the cart (" + "; ".join(notes) + ") but the independent check is not satisfied yet. Cart now: " + self.cart_text()
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies: list[str] = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(ids, self.hard())
                if not self.verdicts.get(self.check_key(ids)) or self.left() < 1 or self.finished or any(self.sold_out(i) for i in ids):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "procedure_buy", "lines": ids, "reply": _clip(reply, 200)})
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
        return "Kyo carried the " + kind + " request out and checked out (" + "; ".join(notes) + "). Store reply: " + " | then: ".join(replies)

    def model_choose(self, number: int, exclude: set[str]) -> str:
        """The light model's choice of the replacement for a contradicted line, among the item's cheapest candidates that pass the code checks (the requirements
        the code cannot verify are the ones it reads). '' when it names none."""
        item = self.items[number - 1]
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        pool = [i for i in self.pool_of(number) if i not in exclude and i not in self.vetoed and self.pick_key(number, i, True) is not None]
        pool = sorted(pool, key=price)[:10]
        if len(pool) < 2:
            return pool[0] if pool else ""
        needs = [m for m in item.musts + item.told if m not in item.dropped]
        head = f"ITEM {number}: {item.name} x{item.qty}; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
        if item.fit_code:
            head += f"; must fit {item.fit_code}"
        records = []
        for ident in pool:
            parts = self.page_parts(self.whole_record(ident))
            facts = "; ".join(f"{k}: {v}" for k, v in list((parts["facts"] or {}).items())[:14] if isinstance(v, (str, int, float)))[:500]
            records.append(f"CANDIDATE {ident}: {parts['title'][:160]} | price {price(ident):g} | {parts['overview'][:200]} | attributes: {facts}")
        if self.ref_values:
            head += "; values read from other listings: " + " | ".join(f"{m[:80]} -> {v[:80]}" for m, v in list(self.ref_values.items())[:3])
        got = self.ask_json(CHOOSE + "\n\n" + head + "\n" + "\n".join(records), [LIGHT, HEAVY], 1500) or {}
        pick = str(got.get("pick") or "")
        self.trace.append({"kind": "model_choose", "item": number, "candidates": len(pool), "pick": pick})
        return pick if pick in pool else ""

    def basket_guard(self) -> str:
        """A request of two items or more that is about to end without a purchase: Kyo puts its own pick in the cart first (once per episode). In a recent
        race every one of 180 such endings failed: the store had a fitting basket. '' when there is no pick, too few steps, or the guard was used."""
        if len(self.items) < 2 or self.__dict__.get("_basket_guarded") or self.finished or self.left() < 8 or self.elapsed() > CLOCK_SOFT + 10:
            return ""
        if not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        self._basket_guarded = True
        self._guarding = True
        try:
            took = self.safe_retake(f"ending without a purchase fails a basket of {len(self.items)} items for certain", final=True)
        finally:
            self._guarding = False
        self.trace.append({"kind": "basket_guard", "carted": bool(took)})
        if not took:
            return ""
        return ("Not run yet (no step used). " + took + " A basket ends without a purchase only when some requested item has no listing at all: "
                "check out this cart unless a line clearly breaks a requirement the shopper stated.")

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
        self.check_cart(list(lines), self.hard())
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
                self.check_cart(list(lines), self.hard())
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
            fresh = {i: c for i, c in lines if c and c > 0}
            # a line sent again unchanged was not put in anew
            for ident, count in fresh.items():
                if self.cart_lines.get(ident) != count:
                    self.line_added[ident] = self.step
            self.cart_lines = fresh
            return
        sets = (not self.cart_adds) if self.cart_adds is not None else self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if not removal and count is not None and count > 0:
                self.last_cart = (ident, count, have)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                if self.cart_lines.get(ident) != have:
                    self.line_added[ident] = self.step
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def cart_repeat(self, name: str, args: dict[str, Any]) -> str:
        """A cart call that would leave the cart as Kyo believes it already is (the line put in at the count it holds, a line taken off that is not
        in it): under a store that adds units it would double the line, so it is not sent. '' when the call changes something or cannot be read."""
        if name != self.roles.get("cart") or not self.cart_touched:
            return ""
        props = self.schemas.get(name, {}).get("properties") or {}
        mode_key = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not mode_key or any(isinstance(v, list) for v in args.values()):
            return ""
        add_mode, remove_mode = _cart_modes(props[mode_key]["enum"])
        ident = next((v for k, v in args.items() if k != mode_key and isinstance(v, str) and v in self.rows), None)
        if not ident:
            return ""
        count = next((_count(v) for k, v in args.items() if k != mode_key and not isinstance(v, str) and _count(v) is not None), None)
        have = self.cart_lines.get(ident, 0)
        if args.get(mode_key) == add_mode and have > 0 and have == (count if count is not None else 1) and self.cart_adds is not False:
            return f"the cart already holds {ident} x{have}; this store adds units, so sending it again would make it x{have + (count or 1)}."
        if args.get(mode_key) == remove_mode and have == 0:
            return f"{ident} is not in the cart."
        return ""

    def sync_cart(self, observation: Any) -> None:
        """The basket as the store shows it replaces the basket as Kyo believed it. A store that adds units instead of setting them, or a line
        the store dropped, shows up here; the add-or-set rule is learned from the first difference after a plain add."""
        if not isinstance(observation, (dict, list)):
            return
        records = [(i, r) for i, r in self.records_in(observation) if i in self.rows]
        if not records:
            empty = any(isinstance(v, list) and not v for v in (observation.values() if isinstance(observation, dict) else [observation]))
            if not empty or any(isinstance(v, list) and v and all(isinstance(r, dict) for r in v) for v in (observation.values() if isinstance(observation, dict) else [])):
                return
            seen: dict[str, int] = {}
        else:
            seen = {}
            for ident, rec in records:
                # the line's count: a figure under a key that speaks of units or pieces; never the price, a total or an amount of money
                figures = [(str(k), _count(v)) for k, v in rec.items()
                           if k != self.price_key and not isinstance(v, (bool, str)) and _count(v) is not None and 0 < _count(v) <= 50
                           and not re.search(r"price|cent|cost|total|amount|sum|money", str(k), re.I)]
                unitish = [c for k, c in figures if re.search(r"unit|count|qty|quant|piece|how_many|num", k, re.I)]
                if unitish:
                    count = unitish[0]
                elif len(figures) == 1:
                    count = figures[0][1]
                else:
                    count = self.cart_lines.get(ident, 1)      # nothing says how many: the belief stands
                seen[ident] = count
        if self.last_cart and self.cart_adds is None:
            ident, sent, had = self.last_cart
            if ident in seen and had > 0 and sent > 0:
                if seen[ident] == had + sent and had + sent != sent:
                    self.cart_adds = True
                elif seen[ident] == sent:
                    self.cart_adds = False
                if self.cart_adds is not None:
                    self.trace.append({"kind": "cart_rule", "adds": self.cart_adds})
        self.last_cart = None
        if seen != self.cart_lines:
            self.trace.append({"kind": "cart_synced", "believed": dict(self.cart_lines), "shown": dict(seen)})
            for ident, count in seen.items():
                if self.cart_lines.get(ident) != count:
                    self.line_added[ident] = self.step
            self.cart_lines = seen

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
        made = 0
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            made += 1
            if self.turn_cap and made > self.turn_cap:
                out.append((cid, f"Not run (no step used): the store allows {self.turn_cap} calls in one turn; send the rest in your next turn."))
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
            if name == self.roles.get("search") and self.code_free:
                args, cut = self.strip_codes(args)
                if cut:
                    trimmed = f"\n[Kyo] Left the model code {', '.join(cut)} out of the search words: a code is not searchable in this store; read each description's compatibility sentence."
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            if target and name == self.roles.get("open") and name != self.roles.get("offer") and str(args.get(target)) in self.full and not self.pending_refs():
                # the attributes of a listing do not change: a second read is answered from memory and costs no step
                ident = str(args[target])
                out.append((cid, f"Not run (no step used): {ident} was read already; its record as read:\n" + self.show(self.full[ident])))
                continue
            repeat = self.cart_repeat(name, args)
            if repeat:
                out.append((cid, "Not run (no step used): " + repeat))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            if self.is_refuse(name, args) and self.roles.get("buy") and self.picked_ready():
                buy_name = self.roles["buy"]
                buy_faults: list[str] = []
                buy_args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy_name), "args", buy_faults)
                if not buy_faults and buy_name in allowed:
                    self.trace.append({"kind": "decline_to_order", "cart": list(self.picked)})
                    name, args = buy_name, buy_args
                    trimmed += "\n[Kyo] Ending without a purchase was turned into the checkout: the cart holds Kyo's pick, which passed its checks, and a store listing that fits every requirement is worth more than no order."
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
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    if self.pending_refs() and self.can_resolve() and self.resolve_holds < 3:
                        # a requirement still points at another listing: Kyo follows it before the order goes in
                        self.resolve_holds += 1
                        before = self.step
                        report = self.resolve_pending("order")
                        if report or self.step != before:
                            out.append((cid, "Not run yet. " + (report or "Kyo looked for the listing a requirement points at and could not settle its value yet.")
                                        + "\nCart now: " + self.cart_text() + "\nCheck every cart line against the requirements as they now read, fix the cart, then check out."))
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
                            self.sync_cart(self.cart_view)
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    if any(i in self.gone for i in self.cart_ids()):
                        replaced = self.retake_gone()
                        if replaced:
                            out.append((cid, "Not run yet. " + (swap + "\n" if swap else "") + replaced + "\nCart now: " + self.cart_text()))
                            continue
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    held_by_code = self.verify_hold()
                    if held_by_code:
                        out.append((cid, "Not run yet (no step used). " + held_by_code + "\nCart now: " + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.soft():
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    took = self.safe_retake("ending without a purchase earns nothing when every item has a listing that meets its requirements", final=self.left() < 24)
                    if took:
                        out.append((cid, "Not run yet. " + took + "\nCart now: " + self.cart_text()))
                        continue
                    held_back = self.decline_hold()
                    if held_back:
                        out.append((cid, held_back + "\nCart now: " + self.cart_text()))
                        continue
                    if self.left() > 5:
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        gaps = [n for n in range(1, len(self.items) + 1) if self.lead_items and str(n) not in {str(k) for k in self.lead_items}]
                        if gaps and self.left() > 25 and self.gap_holds < 3 and self.elapsed() < CLOCK_SOFT - 40:
                            self.gap_holds += 1
                            self.trace.append({"kind": "gap_held", "items": gaps, "n": self.gap_holds})
                            names = "; ".join(f"item {n} ({self.items[n - 1].name}" + (f", aisle \"{self.shelf_of(self.items[n - 1])}\"" if self.shelf_of(self.items[n - 1]) else "") + ")" for n in gaps)
                            out.append((cid, f"Not run yet (no step used). The review found fitting listings for the other items but none for {names}, and {self.left()} steps are left: "
                                        "a missing fit is a search that has not found it yet, not a reason to end. Browse that item's aisle page by page (the search tool's next-page token), "
                                        "try the value words alone and the property's other names, and open what might fit. End without a purchase only after that."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], self.hard())
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
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
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                held = self.basket_guard()
                if held:
                    out.append((cid, held))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
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
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if not ending:
                self.work_at = self.step
                # a listing the actor itself meets is a candidate again, whatever the follower did with it
                met = {i for i, _ in self.records_in(observation)}
                self.ref_rows -= met
                self.ref_pages -= met
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
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
                self.sync_cart(observation)
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
            if name == self.roles.get("search"):
                self.searches += 1
                self.learn_price_key(name, observation)
                query = " ".join(v for v in args.values() if isinstance(v, str)).lower()
                if (self.codes() and not self.records_in(observation) and len(re.findall(r"\w+", query)) <= 3
                        and any(_names_code(query, code) for code in self.codes())):
                    self.code_free = True
                    text += "\n[Kyo] No rows: a model code is not a searchable word in this store. Search by the product's words and read each description's compatibility sentence."
            if name in readers:
                text += self.fit_note(observation)
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and self.note_at != self.step:
                # a refusal that no note explained may be news about stock
                self.stock_at = self.step
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

    # ---------- Kyo's own pick ----------
    def same_aisle(self, shelf: str, kind: str, item: "Item | None" = None) -> bool:
        """Whether a listing's aisle label is the aisle an item's kind is filed under: the same label; when the item's label is itself one of the store's
        labels and the item's name is made of that label's words ("SSDs" for "Internal SSDs") or the brief gave the label, no other label is that aisle;
        else a shared word of 4+ letters (a name with extra words, "gaming laptop bag" -> "Gaming Laptops", only guesses its aisle)."""
        a, b = _words(shelf), _words(kind)
        if a == b:
            return True
        if a in {_words(label) for label in self.aisle_labels()} and (item is None or item.shelf or set(words(item.name)) <= set(words(shelf))
                                                                  or item.name in self.__dict__.get("_shelf_sure", ())):
            return False
        return bool({w for w in _stem(shelf) if len(w) >= 4} & {w for w in _stem(kind) if len(w) >= 4})

    def in_aisle(self, number: int, ident: str) -> bool:
        """Whether a listing stands in the aisle of requested item `number` (the store's label for the item's kind)."""
        item = self.items[number - 1]
        kind = self.page_parts(self.rows.get(ident) or {})["kind"]
        shelf = self.shelf_of(item)
        if shelf and kind:
            return self.same_aisle(shelf, kind, item)
        return self.line_item(ident) == number

    def name_ratio(self, item: "Item", rec: dict[str, Any]) -> float:
        """The share of the words of the item's name that the listing's title, aisle, description or attribute values carry."""
        want = words(item.name)
        if not want:
            return 1.0
        parts = self.page_parts(rec)
        text = " ".join([parts["title"], parts["kind"], parts["overview"][:300]] + [str(v) for v in parts["facts"].values() if isinstance(v, (str, int, float))])
        have = set(words(text, drop_stop=False))
        return sum(1 for w in want if _has(w, have)) / len(want)

    def fit_rank(self, item: "Item", ident: str) -> int:
        """0 when the listing's own text names the item's model code in its first sentence, 1 when no text can say, 2 and up when it cannot fit."""
        if not item.fit_code:
            return 0
        if self.__dict__.get("_pair_fit", {}).get((ident, item.fit_code)):
            return 0                                  # confirmed through another part's page
        order = ("names", "unknown", "later", "no")
        found = [order.index(_fit_state(r, item.fit_code)[0]) for r in (self.rows.get(ident), self.full.get(ident)) if r]
        return min(found) if found else 1

    def whole_record(self, ident: str) -> dict[str, Any]:
        """Everything seen of a listing: the search row, the offer page and the property table merged (the table's snapshot alone lacks the description,
        which is where a model code is written)."""
        return {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}

    def read_description(self, ident: str) -> None:
        """Read a listing's offer page (one step) when only its property table was seen: the description is where a model code or a compatibility
        sentence is written."""
        tool = self.offer_tool()
        target = self.single_target(tool) if tool else None
        if not tool or not target or self.left() < 12 or self.finished or ident in self.__dict__.setdefault("_desc_read", set()):
            return
        self._desc_read.add(ident)
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
        if faults:
            return
        reply = self.act(tool, args)
        if reply is not None and not self.finished:
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get("observation"))

    def model_code_of(self, ident: str, number: int) -> str:
        """The model code a listing of a main item carries: a token of capitals and digits in the first sentence of one of its texts, preferring the one that
        the listings of the items paired with it name most often. '' when none."""
        if not self.page_parts(self.whole_record(ident))["overview"]:
            self.read_description(ident)
        rec = self.whole_record(ident)
        found: list[str] = []
        for value in rec.values():
            if isinstance(value, str) and value.strip():
                found += [t for t in re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,}[0-9]{2,}[A-Za-z0-9]*)(?![A-Za-z0-9])", _first_sentence(value)) if t not in found]
        if not found:
            return ""
        paired = [n for n, it in enumerate(self.items, 1) if it.pairs_with == number]
        texts = [" ".join(str(v) for v in (self.rows.get(i) or {}).values() if isinstance(v, str)) for n in paired for i in self.pool_of(n)]
        return max(found, key=lambda t: (sum(1 for x in texts if t in x), -found.index(t)))

    def sync_pair_codes(self, ids: list[str]) -> None:
        """A paired item's required code follows the main item's line in the cart (the actor may have carted another one than Kyo's pick)."""
        for it in self.items:
            if not it.pairs_with:
                continue
            main = next((i for i in ids if self.line_item(i) == it.pairs_with and not self.sold_out(i)), None)
            if main:
                code = self.model_code_of(main, it.pairs_with)
                if code:
                    it.fit_code = code

    def pick_main(self, number: int, ranked: list[str]) -> str:
        """Of a main item's ranked listings (best first, up to five), the first for whose model code every paired item has a seen listing that passes
        Kyo's whole check (it names the code or a chain confirmed it, meets its requirements and price caps, is in stock); '' when none does."""
        paired = [n for n, it in enumerate(self.items, 1) if it.pairs_with == number]
        chosen = self.__dict__.get("model_picks", {}).get(number)
        if chosen and chosen in ranked and chosen not in self.vetoed:
            try:
                mine, best = self.pick_key(number, chosen, True), self.pick_key(number, ranked[0], True)
                better = self.code_judges(number) and (mine is None or (best is not None and best[0] < mine[0]))
            except Exception:
                better = False
            if not better:
                return chosen                # the light model's choice for this line (the review contradicted the pick before), unless a better-confirmed one exists

        def serves(n: int, confirmed: bool) -> bool:
            for i in self.pool_of(n):
                key = self.pick_key(n, i, True)
                if key is not None and (not confirmed or key[0] < 0.5):
                    return True
            return False

        def cheapest_of(n: int) -> float:
            prices = [p for i in self.pool_of(n) for p in [self.price_of(i)] if p is not None and self.pick_key(n, i, True) is not None]
            return min(prices) if prices else 0.0

        brief, budget = self.__dict__.get("brief") or {}, self.__dict__.get("budget")
        cheapest = brief.get("pick") == "cheapest"
        budget = budget if isinstance(budget, (int, float)) and not isinstance(budget, bool) and brief.get("budget_scope") != "each" else None
        # what the budget leaves for this main and its accessories once the other items take their cheapest passing listing
        share = None
        if budget is not None:
            rest = [n for n in range(1, len(self.items) + 1) if n != number and n not in paired]
            share = budget - sum(cheapest_of(n) * self.items[n - 1].qty for n in rest)
        # first a main whose paired items have a listing that confirms their requirements, then one whose paired items merely pass; among those, the
        # cheapest whole basket for a cheapest request, else the best-ranked one whose cheapest basket fits the budget
        for confirmed in (True, False):
            found: list[tuple[float, str]] = []
            for ident in ranked[: 8 if cheapest else 5]:
                code = self.model_code_of(ident, number)
                if not code:
                    continue
                saved = {n: self.items[n - 1].fit_code for n in paired}
                try:
                    for n in paired:
                        self.items[n - 1].fit_code = code
                    # every paired item has a listing that passes the whole check with this code (fit, requirements, price caps, stock)
                    if all(serves(n, confirmed) for n in paired):
                        if not cheapest and share is None:
                            return ident
                        total = (self.price_of(ident) or 0.0) * self.items[number - 1].qty + sum(cheapest_of(n) * self.items[n - 1].qty for n in paired)
                        found.append((total, ident))
                finally:
                    for n, c in saved.items():
                        self.items[n - 1].fit_code = c
            if found:
                if cheapest:
                    return min(found)[1]
                fitting = [i for t, i in found if share is None or t <= share + 1e-9]
                return fitting[0] if fitting else min(found)[1]
        return ""

    def follow_pair_chains(self, number: int, pool: list[str], limit: int = 3) -> bool:
        """For listings of a paired item whose own text does not name the model code: follow the part each one points to ("see part X") and accept the
        listing when that part's page has a sentence naming the listing's own part code together with the model code. Cheapest first, at most `limit`
        pointers, only while steps allow. True when one was accepted."""
        item = self.items[number - 1]
        code = item.fit_code
        fits = self.__dict__.setdefault("_pair_fit", {})
        done = self.__dict__.setdefault("_pair_followed", set())
        token = r"(?<![A-Za-z0-9])([A-Z]{2,}[0-9]{3,})(?![A-Za-z0-9])"
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        accepted = False
        tried = 0

        def stated_elsewhere(own_code: str, skip: str) -> str:
            """Another listing (seen as a row or a page) with a sentence naming this part's own code together with the model code."""
            for other, record in list(self.full.items()) + list(self.rows.items()):
                if other == skip:
                    continue
                body = " ".join(str(v) for v in record.values() if isinstance(v, str))
                if own_code not in body:
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", body):
                    if own_code in sentence and _names_code(sentence, code):
                        return other
            return ""

        for ident in sorted(pool, key=price):
            if tried >= limit or self.left() < self.tail() + 12 or self.finished:
                break
            rec = self.whole_record(ident)
            if not rec or not self.buyable(ident) or _fit_state(rec, code)[0] == "names":
                continue
            title = self.page_parts(rec)["title"]
            own = re.findall(token, title)
            if not own:
                continue
            own_code, prefix = own[-1], re.match(r"[A-Z]+", own[-1]).group(0)
            seen_at = stated_elsewhere(own_code, ident)
            if seen_at:
                fits[(ident, code)] = True
                accepted = True
                self.trace.append({"kind": "pair_chain", "item": number, "listing": ident, "via": seen_at, "code": code, "steps": 0})
                continue
            text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
            # a pointer: another part code of the same family in a sentence that names no model code
            pointers = [t for s in re.split(r"(?<=[.!?])\s+", text) if not re.search(r"(?<![A-Za-z0-9])" + re.escape(re.match(r"[A-Z]+", code).group(0)) + r"[0-9]", s)
                        for t in re.findall(token, s) if t != own_code and t.startswith(prefix)]
            if (ident, code) in done:
                continue
            done.add((ident, code))
            tried += 1
            if not pointers:
                # no pointer: the part's own code searched once (a row's description carries "Part <code> also fits ...")
                self.search_with(own_code, kind=self.shelf_of(item) or item.name)
                seen_at = stated_elsewhere(own_code, ident)
                if seen_at:
                    fits[(ident, code)] = True
                    accepted = True
                    self.trace.append({"kind": "pair_chain", "item": number, "listing": ident, "via": seen_at, "code": code, "steps": 1})
                continue
            target = pointers[0]
            page = next((i for i, r in self.rows.items() if target in self.page_parts(r)["title"]), None)
            if page is None:
                self.search_with(target, kind=self.shelf_of(item) or item.name)
                page = next((i for i, r in self.rows.items() if target in self.page_parts(r)["title"]), None)
            if page is None:
                continue
            if page not in self.full:
                self.fill_facts([page], self.tail() + 6)
            body = " ".join(str(v) for v in (self.full.get(page) or self.rows.get(page) or {}).values() if isinstance(v, str))
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                if own_code in sentence and _names_code(sentence, code):
                    fits[(ident, code)] = True
                    accepted = True
                    self.trace.append({"kind": "pair_chain", "item": number, "listing": ident, "via": page, "code": code})
                    break
        return accepted

    def can_read(self) -> bool:
        try:
            tool = self.facts_tool()
            return bool(tool and self.single_target(tool) and self.facts_from.get(tool) is not False)
        except Exception:
            return False

    def pick_key(self, number: int, ident: str, loose: bool = False) -> tuple | None:
        """None when the listing cannot serve the item (sold out, does not name the model code, a requirement fails by its words);
        else a sort key, smaller is better: requirements not confirmed, name words missing, price, model code unconfirmed, look-alike strength."""
        item = self.items[number - 1]
        rec = self.full.get(ident) or self.rows.get(ident)
        if not rec or not self.buyable(ident) or self.set_aside(ident) or ident in self.vetoed:
            return None
        if self.unit_cap_problem(number, ident):
            return None
        fit = self.fit_rank(item, ident)
        if fit >= 2:
            return None
        read = ident in self.full
        if not read and not loose and self.can_read():
            return None
        if read and self.verify_line(number, ident) and not self.__dict__.get("_desperate"):
            return None
        needs = [m for m in item.musts + item.told if m not in item.dropped]
        parts = self.page_parts(rec)
        open_needs = len(needs)
        open_count = len(needs)
        weak = 0.0
        if read:
            # a requirement counts as met when an attribute naming its property states the value in full (a title that carries the words, or a bare
            # figure for "1 l", leaves it open): of two listings the one that states it wins, price decides between equals
            levels = [self.confirm_level(m, parts["facts"]) for m in needs]
            if self.__dict__.get("_price_first"):
                # the budget fallback: a title that states the requirement counts as met (the cheapest listing that says it, not the cheapest that has the attribute)
                levels = [min(lv, 0.5) if lv >= 1.0 and self.text_supports(m, parts, ident) else lv for m, lv in zip(needs, levels)]
            else:
                # a quantity the code cannot compare (another unit, a limit, a title figure): the light model's reading makes it half-confirmed
                levels = [min(lv, 0.5) if lv >= 1.0 and quantities(split_must(m)[1]) and self.text_supports(m, parts, ident) else lv for m, lv in zip(needs, levels)]
            open_needs = sum(levels)
            open_count = sum(1 for lv in levels if lv >= 1.0)
            weak = sum(levels)
        ratio = self.name_ratio(item, rec)
        tier = 0 if ratio >= 0.999 else (1 if ratio >= 0.6 else 2)
        strong = 0
        if read:
            strong = sum(1 for m in self.strong_musts(item) if verify_must(m, parts["facts"], parts["title"], parts["overview"], parts["kind"])[0] == "pass")
            strong += 1 if self.name_in_facts(item, parts["facts"]) else 0
        price = self.price_of(ident)
        price = price if price is not None else float("inf")
        if self.__dict__.get("_price_first"):
            return (open_count, tier, price, weak, fit, -strong)
        return (open_needs, tier, price, fit, -strong)

    def text_supports(self, must: str, parts: dict[str, Any], ident: str = "") -> bool:
        """The listing's title or description states the requirement. A value made of words: every word is there. A value with a quantity: the light
        model's answer for this listing (`text_states`, units converted), when it has been asked. Used only where the pick has nothing better to rank by
        (the budget fallback)."""
        prop, cons = split_must(must)
        if not cons or _is_ref(must) or _is_ref(cons):
            return False
        if quantities(cons):
            return bool(self.__dict__.get("_text_states", {}).get((must, ident)))
        if _NUM.search(cons) or _REL_LEAD.match(cons):
            return False
        text = " ; ".join([parts.get("title") or "", (parts.get("overview") or "")[:300]])
        return _contains(words(cons), text)

    def text_states(self, number: int, pool: list[str]) -> None:
        """Ask the light model which of the cheapest listings of an item state each quantity requirement in their title or description (one call per
        requirement, at most 12 listings); the answers are kept for `text_supports`."""
        item = self.items[number - 1]
        cache = self.__dict__.setdefault("_text_states", {})
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        for must in item.musts + item.told:
            prop, cons = split_must(must)
            if must in item.dropped or not cons or _is_ref(must) or _is_ref(cons) or not quantities(cons):
                continue
            todo = [i for i in sorted(pool, key=price) if (must, i) not in cache and i in self.rows and self.buyable(i)][:12]
            if not todo or self.calls >= ALL_CALLS - 4 or self.elapsed() > CLOCK_SOFT:
                continue
            lines = []
            for i in todo:
                parts = self.page_parts(self.full.get(i) or self.rows.get(i) or {})
                facts = "; ".join(f"{k}: {v}" for k, v in list((parts["facts"] or {}).items())[:10] if isinstance(v, (str, int, float)))[:300]
                lines.append(f"{i}: {parts['title'][:160]} | {parts['overview'][:160]}" + (f" | attributes: {facts}" if facts else ""))
            got = self.ask_json(TEXTSTATE + "\n\nREQUIREMENT: " + must + "\n\nLISTINGS:\n" + "\n".join(lines), [LIGHT, HEAVY], 1500) or {}
            said = {str(x) for x in _items(got.get("states"))}
            for i in todo:
                cache[(must, i)] = i in said
            self.trace.append({"kind": "text_states", "item": number, "must": must[:60], "listings": len(todo), "states": len(said & set(todo))})

    def confirm_level(self, must: str, facts: dict[str, Any]) -> float:
        """0 when an attribute whose name holds the requirement's property words states its value (every value word, or all the attribute
        says is inside the requirement: "red" for "only red"); 0.5 when an attribute of another name does; 1 when none does."""
        prop, cons = split_must(must)
        if not prop or not cons:
            return 1.0
        if _is_ref(must) or _is_ref(cons) or _is_ref(prop):
            return 0.0                    # a reference not yet followed says nothing about any listing
        if _REL_LEAD.match(cons):
            return 0.0 if verify_must(must, facts)[0] == "pass" else 1.0
        pw = [w for w in words(prop, drop_stop=False) if w not in ("a", "an", "the", "of", "and", "or", "for", "with", "in", "on", "to", "by")]
        alts = [words(a) for a in re.split(r",|;|/|\bor\b", re.sub(r"\(.*?\)", " ", cons)) if a.strip()] or [words(cons)]
        alts = [a for a in alts if a]
        if not alts:
            return 1.0
        best = 1.0
        # a value of several words names ONE thing ("lightning to usb"): in a list of entries ("micro usb, usb-c, lightning") its words must sit in one
        # entry; a value joined by "and" may spread over the entries
        joined = re.search(r"\band\b|&|\+", cons, re.I) is not None
        for key, value in fact_pairs(facts or {}):
            have = set(words(value, drop_stop=False))
            if not have:
                continue
            entries = [set(words(e, drop_stop=False)) for e in re.split(r",|;|\|", value) if e.strip()]
            exact = any(all(_has(w, have) for w in alt) and (len(alt) < 2 or joined or len(entries) < 2 or any(all(_has(w, e) for w in alt) for e in entries))
                        for alt in alts)
            inside = (any(have <= set(alt) and not any(re.search(r"\d", w) for w in set(alt) - have) for alt in alts)
                      and any(not re.fullmatch(r"\d+(?:\.\d+)?", w) for w in have))
            if not (exact or inside):
                continue
            if pw and set(pw) <= set(key):
                several = len([e for e in re.split(r",|;|/|&|\band\b", value) if e.strip()]) >= 2
                if several and len(alts) == 1 and not {w for w in have if w not in STOP} <= set(alts[0]):
                    best = min(best, 0.25)         # the attribute names more than the requirement does: the entry is one of several
                    continue
                return 0.0
            best = min(best, 0.5)
        return best

    def strict_confirm(self, must: str, facts: dict[str, Any]) -> bool:
        return self.confirm_level(must, facts) == 0.0

    def relax(self, number: int, pool: list[str]) -> bool:
        """Last resort, after the narrower searches found nothing: a requirement that none of three or more read listings of the aisle (naming the
        model code) meets, and some of them contradict, is dropped for this item: an aisle with nothing to buy is not what a shopper asks for, so the request was misread."""
        item = self.items[number - 1]
        read = [i for i in pool if i in self.full and self.fit_rank(item, i) < 2 and self.buyable(i) and not self.set_aside(i)]
        unread = [i for i in pool if i not in self.full and self.buyable(i) and self.fit_rank(item, i) < 2]
        if len(read) < 3 or (len(unread) > 2 and len(read) < 40):
            return False
        changed = False
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            states = []
            incomparable = False
            for i in read:
                parts = self.page_parts(self.full[i])
                state = verify_must(must, parts["facts"], parts["title"], parts["overview"], parts["kind"])[0]
                states.append(state)
                if state == "unknown" and self.names_property(must, parts["facts"]):
                    incomparable = True
            if "fail" in states and "pass" not in states and not incomparable:
                fixed = self.repair_must(must, read)
                if fixed:
                    # the request's own words around the property name state what the shopper wants: the requirement is rewritten with them
                    for lst in (item.musts, item.told):
                        if must in lst:
                            lst[lst.index(must)] = fixed
                    self.trace.append({"kind": "repaired", "item": number, "must": must[:80], "to": fixed[:80]})
                else:
                    item.dropped.append(must)
                    self.trace.append({"kind": "relaxed", "item": number, "must": must[:80], "read": len(read)})
                changed = True
        return changed

    @staticmethod
    def names_property(must: str, facts: dict[str, Any]) -> bool:
        """Whether one of the listing's attributes is named for the requirement's property."""
        prop, _ = split_must(must)
        pw = [w for w in words(prop, drop_stop=False) if w not in ("a", "an", "the", "of", "and", "or", "for", "with", "in", "on", "to", "by")]
        return bool(pw) and any(set(pw) <= set(k) for k, _ in fact_pairs(facts or {}))

    def repair_must(self, must: str, read: list[str]) -> str:
        """For a requirement that every read listing contradicts: the words of the request around the property's name, as the value, when
        some read listing meets that ("services: included with the fluid" -> "services: with services"); '' when none does."""
        prop, cons = split_must(must)
        pw = [w for w in words(prop, drop_stop=False)]
        if not pw or _is_ref(must):
            return ""
        tokens = re.findall(r"[A-Za-z0-9]+", self.request.lower())
        stems = [words(t, drop_stop=False) for t in tokens]
        last = pw[-1]
        spots = [i for i, st in enumerate(stems) if st and st[0] == last]
        windows: list[list[str]] = []
        for i in spots:
            for lo, hi in ((i - 1, i), (i - 2, i), (i, i + 1), (i, i + 2), (i - 1, i - 1), (i + 1, i + 1)):
                if lo >= 0 and hi < len(tokens):
                    windows.append(tokens[lo:hi + 1])
        seen: set[str] = set()
        for window in windows:
            value = " ".join(window)
            if value in seen or value == cons.lower() or not words(value):
                continue
            seen.add(value)
            candidate = f"{prop}: {value}"
            passes = 0
            for i in read:
                parts = self.page_parts(self.full[i])
                if verify_must(candidate, parts["facts"], parts["title"], parts["overview"], parts["kind"])[0] == "pass":
                    passes += 1
            if 0 < passes < len(read):
                return candidate
        return ""

    def add_named_need(self, number: int, pool: list[str]) -> None:
        """A make or brand the request attaches to the item ("one ball joint for a Honda") that the brief left out: when it is the value of an
        attribute on the read listings of the aisle, it becomes a requirement on that attribute (never for a word the brief already carries)."""
        item = self.items[number - 1]
        read = [i for i in pool if i in self.full]
        done = self.__dict__.setdefault("_named_done", {})
        if len(read) < 3 or done.get(number) == len(read):
            return
        done[number] = len(read)
        covered = set(words(" ".join([item.name] + item.musts + item.told), drop_stop=False))
        name_words = set(words(item.name))
        if not name_words:
            return
        for clause in re.split(r"(?<=[.!?;])\s+", self.request):
            for m in re.finditer(r"\bfor (?:a |an |the |my )?([A-Z][a-z]{2,})\b(?!['\u2019]s)", clause):
                token = m.group(1).lower()
                tw = set(words(token, drop_stop=False))
                if not tw or tw <= covered or token in {w.lower() for w in re.findall(r"[A-Za-z]+", item.name)}:
                    continue
                # the make belongs to the item whose name stands right before it ("two genuine control arms and two lower ball joints for a Toyota")
                before = set(words(" ".join(clause[:m.start()].split()[-6:])))
                scores = [len(set(words(it.name)) & before) / max(1, len(set(words(it.name)))) for it in self.items]
                best = max(scores)
                if best < 0.5 or scores[number - 1] < best or scores.count(best) > 1:
                    continue
                # the attribute that carries the make is learned from any read listing of the store (the item's own cheap listings may all be
                # of other makes: that is the point of the requirement); the item's listings must carry that attribute
                hits: dict[str, int] = {}
                for i, rec in self.full.items():
                    if i in self.ref_pages:
                        continue
                    for key, value in (self.page_parts(rec)["facts"] or {}).items():
                        if isinstance(value, str) and token in {w.lower() for w in re.findall(r"[A-Za-z]+", value)}:
                            hits[str(key)] = hits.get(str(key), 0) + 1
                carried = {str(k) for i in read for k in (self.page_parts(self.full[i])["facts"] or {})}
                hits = {k: c for k, c in hits.items() if k in carried}
                if hits:
                    key = max(hits.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
                    need = f"{key.replace('_', ' ').lower()}: {token}"
                    item.musts.append(need)
                    self.trace.append({"kind": "named_need", "item": number, "need": need})

    def pool_of(self, number: int) -> list[str]:
        base = [i for i in dict.fromkeys(self.seen_order) if i in self.rows and i not in self.ref_pages and i not in self.ref_rows]
        pool = [i for i in base if self.in_aisle(number, i)]
        return pool or [i for i in base if self.line_item(i) == number]

    def read_pool(self, number: int, pool: list[str], room: int) -> int:
        """Read the attributes of the cheapest unread listings of an item that can still serve it (up to `room`), stopping at the price of a
        verified one already read. Returns how many were read."""
        item = self.items[number - 1]
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        best = None
        for i in sorted(pool, key=price):
            key = self.pick_key(number, i) if i in self.full else None
            if key is not None and key[0] == 0:
                best = price(i)
                break
        picks: list[str] = []
        focus = [i for i in self.__dict__.setdefault("_focus", {}).get(number, []) if i in pool]
        for i in focus:
            if i not in self.full and self.buyable(i) and self.fit_rank(item, i) < 2 and i not in picks:
                picks.append(i)
            if len(picks) >= room:
                break
        for i in sorted(pool, key=price):
            if len(picks) >= room:
                break
            if i in self.full or i in picks or not self.buyable(i) or self.fit_rank(item, i) >= 2:
                continue
            if best is not None and price(i) > best:
                break
            picks.append(i)
        if not picks:
            return 0
        before = len(self.full)
        self.fill_facts(picks, self.tail() + 6)
        return len(self.full) - before

    def explore_queries(self, number: int) -> list[str]:
        """Searches for an item's aisle, the narrowest first: the aisle label (as the store writes it, the search does not stem) with every
        requirement's value words, with each value alone, then the label alone. The search matches an aisle label and attribute values, so
        the narrow ones list the cheapest listings that state the values."""
        item = self.items[number - 1]
        shelf = self.shelf_of(item)
        label = " ".join(re.findall(r"[a-z0-9]+", shelf.lower())) if shelf else ""
        values: list[str] = []
        for need in item.musts + item.told:
            if need in item.dropped:
                continue
            prop, cons = split_must(need)
            if not cons or _REL_LEAD.match(cons) or _is_ref(need) or _is_ref(cons) or re.search(r",|;|/|\bor\b", cons):
                continue
            tokens = re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", cons.lower()))
            if tokens and len(tokens) <= 5:
                values.append(" ".join(tokens))
        # the product's own words as written (a store may match whole words: "leaves", not the stem "leave"), up to its qualifier ("with dried leaves")
        head = re.split(r"\b(?:with|for|in|that|which|made|having|of)\b", item.name.lower())[0]
        plain = [t for t in re.findall(r"[a-z0-9]+", head) if t not in STOP] or [t for t in re.findall(r"[a-z0-9]+", item.name.lower()) if t not in STOP]
        base = label or " ".join(plain)
        out = []
        if len(values) >= 2:
            out.append(base + " " + " ".join(values))
        out += [base + " " + v for v in values]
        out += [base + " " + t for v in values for t in v.split() if len(t) >= 3 and " " in v]
        out.append(base)
        return [q for q in dict.fromkeys(out) if q.strip()]

    def explore(self, number: int) -> bool:
        """Run the item's next narrower search (one page; up to four searches per item) for listings the actor never met."""
        done = self.explored.get(number, 0)
        queries = self.explore_queries(number)
        if done >= 5 or done >= len(queries) or self.left() < 24 or self.finished:
            return False
        # the allowance: about a third of the steps left when the first narrower search runs; with three or more items each item gets its share
        if self.explore_budget is None:
            self.explore_budget = max(10, self.left() // 3)
        if self.explore_spent >= self.explore_budget or (len(self.items) >= 3 and self.explore_item.get(number, 0) >= max(4, self.explore_budget // len(self.items))):
            return False
        item = self.items[number - 1]
        shelf = self.shelf_of(item)
        query = queries[done]
        before = len(self.rows)
        self._plain = True
        pages = 1
        try:
            seen = self.search_with(query, kind=shelf or item.name)
            if seen is not None and done < len(queries) - 1:
                # the narrow searches list rows whose attributes hold the requirement's words: those are read first (cheapest first among them)
                rows = [i for i, _ in self.records_in(seen) if i in self.rows]
                focus = self.__dict__.setdefault("_focus", {}).setdefault(number, [])
                focus[:0] = [i for i in sorted(rows, key=lambda r: self.price_of(r) if self.price_of(r) is not None else float("inf")) if i not in focus]
            # a full page of the narrowest searches (cheapest first) may hold no listing that states everything: up to two more pages
            while done < 2 and seen is not None and pages < 3 and len(self.records_in(seen)) >= 8 and self.left() >= 30:
                seen = self.next_rows(query, seen, shelf or item.name)
                if seen is None:
                    break
                pages += 1
        finally:
            self._plain = False
        self.explored[number] = done + 1
        self.explore_spent += pages
        self.explore_item[number] = self.explore_item.get(number, 0) + pages
        self.trace.append({"kind": "explore", "item": number, "query": query, "pages": pages, "new_rows": len(self.rows) - before})
        return True

    def pick_plan(self, loose: bool = False, only: set[int] | None = None, held: set[str] | None = None) -> dict[int, str] | None:
        """One listing per requested item, each passing the code checks, or None when an item has none. Reads the attributes of the cheapest
        candidates (steps only, no model call) and browses an aisle when nothing in it can serve the item."""
        taken: set[str] = set(held or ())
        plan: dict[int, str] = {}
        count = len(self.items)
        room = 5 if count <= 3 else 3
        reads = 0
        cap = max(12, min(40, self.left() * 2 // 3))
        order = sorted((n for n in range(1, count + 1) if not only or n in only), key=lambda n: len(self.pool_of(n)))
        mains = {it.pairs_with for it in self.items if it.pairs_with}
        if mains:
            # the items others must work with are picked first: their model code decides which listings of the others can serve
            order = [n for n in order if n in mains] + [n for n in order if n not in mains]
            for it in self.items:
                if it.pairs_with and not (only and it.pairs_with not in only):
                    held_main = next((i for i in (held or ()) if self.line_item(i) == it.pairs_with), None)
                    if held_main:
                        it.fit_code = self.model_code_of(held_main, it.pairs_with) or it.fit_code
        for position, number in enumerate(order):
            ranked: list[tuple[tuple, str]] = []
            # each item gets its share of the reads still allowed, so the last item is not left with none
            ceiling = reads + max(6, (cap - reads) // (len(order) - position))
            for round_ in range(10):
                pool = [i for i in self.pool_of(number) if i not in taken]
                if self.__dict__.get("_price_first") or any(quantities(split_must(m)[1]) for m in self.items[number - 1].musts + self.items[number - 1].told):
                    try:
                        self.text_states(number, pool)
                    except Exception as exc:
                        self.trace.append({"kind": "guarded", "fn": "text_states", "error": f"{type(exc).__name__}: {exc}"[:200]})
                if self.left() > self.tail() + 10 and reads < min(cap, ceiling):
                    reads += self.read_pool(number, pool, room)
                try:
                    self.drop_counted(number, pool)
                except Exception as exc:
                    self.trace.append({"kind": "guarded", "fn": "drop_counted", "error": f"{type(exc).__name__}: {exc}"[:200]})
                try:
                    self.add_named_need(number, pool)
                except Exception:
                    pass
                ranked = sorted((k, i) for i in pool for k in [self.pick_key(number, i, loose)] if k is not None)
                # settled: a listing whose attributes state every requirement is read
                if ranked and ranked[0][0][0] < 0.5:
                    break
                if ranked and self.good_enough(number, ranked[0][1], pool):
                    break
                # otherwise a narrower search may list one that states what is still open
                if not self.explore(number):
                    # listings of the aisle that were never read may state what is still open: they are read before any requirement is doubted
                    item_ = self.items[number - 1]
                    if (reads < min(cap, ceiling) and self.left() > self.tail() + 10 and round_ < 9
                            and any(i not in self.full and self.buyable(i) and self.fit_rank(item_, i) < 2 for i in pool)):
                        continue
                    if (not ranked or ranked[0][0][0] >= 0.5) and self.left() > self.tail() + 30:
                        # the searches are used up and nothing read confirms the item: its shelf, listed under what one unit may cost
                        swept = self.shelf_sweep(number, max(4, min(10, ceiling - reads)))
                        if swept:
                            reads += swept
                            continue
                    if (not ranked or ranked[0][0][0] >= 0.5) and self.relax(number, pool):
                        continue
                    break
            if not ranked and self.items[number - 1].pairs_with and self.items[number - 1].fit_code:
                # nothing names the model code itself: a listing may fit it through another part's page
                pool = [i for i in self.pool_of(number) if i not in taken]
                if self.follow_pair_chains(number, pool):
                    ranked = sorted((k, i) for i in pool for k in [self.pick_key(number, i, loose)] if k is not None)
            if not ranked:
                item = self.items[number - 1]
                pool = self.pool_of(number)
                self.trace.append({"kind": "pick_gap", "item": number, "name": item.name, "pool": len(pool), "read": sum(1 for i in pool if i in self.full),
                                   "fit": sum(1 for i in pool if self.fit_rank(item, i) < 2),
                                   "failing": [self.verify_line(number, i)[:1] for i in pool if i in self.full and self.fit_rank(item, i) < 2][:3]})
                return None
            choice = ranked[0][1]
            chosen = self.__dict__.get("model_picks", {}).get(number)
            # the light model decides among the listings confirmed as well as the best one, never against a better-confirmed one
            by_model = bool(chosen and chosen in {i for _, i in ranked} and chosen not in self.vetoed
                            and (not self.code_judges(number) or next(k for k, i in ranked if i == chosen)[0] <= ranked[0][0][0]))
            if by_model:
                choice = chosen
            if number in mains:
                choice = self.pick_main(number, [i for _, i in ranked]) or choice
                code = self.model_code_of(choice, number)
                for it in self.items:
                    if it.pairs_with == number and code:
                        it.fit_code = code
                self.trace.append({"kind": "pair_main", "item": number, "listing": choice, "code": code})
            plan[number] = choice
            taken.add(choice)
        return plan

    def drop_counted(self, number: int, pool: list[str]) -> None:
        """A requirement that only repeats the unit the request counts the item in ("three pairs of sneakers" -> "pack type: pair") is dropped once three
        read listings of the item's aisle show that fewer than a quarter of them have an attribute named for that property or a value holding that unit (footwear is sold by the
        pair without saying so); a shelf where a quarter of the read listings state "pack size: 1 pair" keeps it."""
        item = self.items[number - 1]
        read = [i for i in pool if i in self.full and self.fit_rank(item, i) < 2]
        if len(read) < 3:
            return
        for must in list(item.musts):
            if must in item.dropped or not _counted_unit(self.request, must):
                continue
            unit = _words(split_must(must)[1]).split()[-1].rstrip("s")
            shown = 0
            for i in read:
                facts = self.page_parts(self.full[i])["facts"] or {}
                if self.names_property(must, facts) or any(re.search(r"\b" + re.escape(unit) + r"s?\b", str(v), re.I) for v in facts.values()):
                    shown += 1
            # where the pack is a real choice most listings state it (hood scoops & vents 20 of 31); where the unit only counts, almost none do (sneakers 2 of 253)
            if shown < max(1.0, 0.25 * len(read)):
                item.dropped.append(must)
                self.trace.append({"kind": "counted_unit", "item": number, "must": must[:80], "read": len(read)})

    def good_enough(self, number: int, ident: str, pool: list[str]) -> bool:
        """The best listing of the item states every requirement under some attribute (none is left to the title or to nothing) and more
        reading would only look for a stricter key: after eight listings of the item are read, or after five when no read listing has an
        attribute named for the property at all (a key nobody carries cannot be found by reading on). Never for a cheapest, most or least
        request, and never when some item is tied to a model code (the fitment pick keeps reading for the strict match)."""
        try:
            if self.brief.get("pick") in ("cheapest", "most", "least"):
                return False
            item = self.items[number - 1]
            read = [i for i in pool if i in self.full]
            if any(it.fit_code for it in self.items) or len(read) < 5 or ident not in self.full:
                return False
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            if not needs:
                return True
            facts = self.page_parts(self.full[ident])["facts"]
            levels = [(m, self.confirm_level(m, facts)) for m in needs]
            if max(lv for _, lv in levels) > 0.5:
                return False
            if len(read) >= 8:
                return True
            open_ = [m for m, lv in levels if lv > 0.0]
            return all(not any(self.names_property(m, self.page_parts(self.full[i])["facts"]) for i in read) for m in open_)
        except Exception:
            return False

    def browse_args(self, shelf: str, cap: float | None) -> dict[str, Any] | None:
        """Arguments of the shelf-listing tool: the shelf's label in the text parameter named for a shelf, aisle, section or category (else its one
        required text parameter), and a price cap in the number parameter named for a price, a cap or a maximum (never one that counts rows)."""
        tool = self.roles.get("browse")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        about = lambda k: f"{k} {(props.get(k) or {}).get('description') or ''}"
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        named = [k for k in texts if re.search(r"shelf|aisle|section|categor|department|group", about(k), re.I)]
        required = [k for k in schema.get("required") or [] if k in texts]
        key = (named or required or [None])[0]
        if not key or not shelf:
            return None
        args: dict[str, Any] = {key: shelf}
        if cap is not None and cap > 0:
            numbers = [k for k, v in props.items() if isinstance(v, dict) and ("integer" in _kinds(v) or "number" in _kinds(v))]
            priced = [k for k in numbers if re.search(r"price|cost", about(k), re.I)]
            capped = [k for k in numbers if re.search(r"cap|max|under|budget|ceiling", about(k), re.I)
                      and not re.search(r"row|result|item|page|count|offset|skip|from|after|start", about(k), re.I)]
            name = (priced or capped or [None])[0]
            if name:
                args[name] = int(cap) if "integer" in _kinds(props[name]) else round(float(cap), 2)
        return args

    def budget_sweep(self, plan: dict[int, str] | None) -> bool:
        """A pick over the budget: each line's shelf is listed under the price one unit may cost when the other lines take their cheapest known listing
        (the dearest line first, up to four pages) and its cheapest unread listings are read. Once per line; True when a listing was read."""
        tool = self.roles.get("browse")
        if not tool or not plan or not isinstance(self.budget, (int, float)) or self.brief.get("budget_scope") == "each" or self.finished:
            return False
        done = self.__dict__.setdefault("_swept", set())
        price = lambda i: self.price_of(i) if i and self.price_of(i) is not None else None

        def floor(n: int) -> float:
            known = [p for i in self.pool_of(n) for p in [price(i)] if p is not None and self.pick_key(n, i, True) is not None]
            return min(known) if known else (price(plan.get(n)) or 0.0)

        floors = {n: floor(n) for n in range(1, len(self.items) + 1)}
        before = len(self.full)
        for n in sorted(plan, key=lambda m: -((price(plan[m]) or 0.0) * self.items[m - 1].qty)):
            if n in done or self.left() < 24:
                continue
            item = self.items[n - 1]
            cap = (self.budget - sum(floors[m] * self.items[m - 1].qty for m in floors if m != n)) / max(1, item.qty)
            if cap <= 0 or (price(plan[n]) or 0.0) <= cap:
                continue
            args = self.browse_args(self.shelf_of(item), cap)
            if not args:
                continue
            done.add(n)
            self._plain = True
            pages, seen = 0, None
            try:
                seen = self.rcall(tool, args)
                pages = 1 if seen is not None else 0
                while seen is not None and pages < 4 and self.left() > 24:
                    nxt = self.paging(tool, seen)
                    if not nxt or nxt[3] is None or nxt[0] <= nxt[1] * pages:
                        break
                    seen = self.rcall(tool, {**args, nxt[2]: nxt[3]})
                    pages += 1
            finally:
                self._plain = False
            pool = [i for i in self.pool_of(n) if price(i) is not None and price(i) <= cap + 1e-9]
            read = self.read_pool(n, pool, max(4, min(14, (self.left() - 20) // 2))) if pool else 0
            self.trace.append({"kind": "budget_sweep", "item": n, "cap": round(cap, 2), "pages": pages, "under_cap": len(pool), "read": read})
        return len(self.full) > before

    def code_judges(self, number: int) -> bool:
        """Whether the code's confirmation levels speak for every requirement of the item: none takes its value from another listing, a page or a
        store rule, and none is a relation (the code rates those as neither met nor failed, so only the model can tell listings apart on them)."""
        item = self.items[number - 1]
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            prop, cons = split_must(must)
            if (_is_ref(must) or _is_ref(cons) or _is_ref(prop) or _REL_LEAD.match(cons)
                    or re.search(r"\b(?:rule|page|listing|value)\b", cons, re.I)):
                return False
        return True

    def shelf_sweep(self, number: int, room: int) -> int:
        """An item whose narrower searches are used up while no read listing confirms its requirements: its shelf is listed (up to four pages) under the
        most one unit may cost (the request's per-unit cap, else the budget left after the other items' cheapest known listings, else no cap) and the
        cheapest unread listings that can serve it are read. Once per item (shared with the budget sweep); returns how many were read."""
        tool = self.roles.get("browse")
        done = self.__dict__.setdefault("_swept", set())
        if not tool or number in done or self.finished or self.left() < 30:
            return 0
        item = self.items[number - 1]
        cap = self.unit_cap(item)
        if cap is None and isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            floors = []
            for m in range(1, len(self.items) + 1):
                if m == number:
                    continue
                known = [p for i in self.pool_of(m) for p in [self.price_of(i)] if p is not None and self.pick_key(m, i, True) is not None]
                floors.append((min(known) if known else 0.0) * self.items[m - 1].qty)
            cap = (self.budget - sum(floors)) / max(1, item.qty)
            if cap <= 0:
                return 0
        args = self.browse_args(self.shelf_of(item), cap)
        if not args:
            return 0
        done.add(number)
        before = len(self.full)
        self._plain = True
        pages, seen = 0, None
        try:
            seen = self.rcall(tool, args)
            pages = 1 if seen is not None else 0
            while seen is not None and pages < 4 and self.left() > 30:
                nxt = self.paging(tool, seen)
                if not nxt or nxt[3] is None or nxt[0] <= nxt[1] * pages:
                    break
                seen = self.rcall(tool, {**args, nxt[2]: nxt[3]})
                pages += 1
        finally:
            self._plain = False
        pool = [i for i in self.pool_of(number) if cap is None or (self.price_of(i) is not None and self.price_of(i) <= cap + 1e-9)]
        read = self.read_pool(number, pool, room) if pool else 0
        self.trace.append({"kind": "shelf_sweep", "item": number, "cap": None if cap is None else round(cap, 2), "pages": pages, "listed": len(pool), "read": read})
        return len(self.full) - before

    def plan_total(self, plan: dict[int, str]) -> float:
        return sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in plan.items())

    def picked_ready(self) -> bool:
        """The cart holds just the lines of Kyo's last pick, in stock, at those counts, within the spend limit, and the word check finds nothing."""
        if not self.picked:
            return False
        try:
            now = self.cart_ids()
            if set(now) != set(self.picked) or any(self.cart_lines.get(i, self.picked[i]) != q for i, q in self.picked.items()):
                return False
            if any(self.sold_out(i) for i in now) or self.cart_over_budget():
                return False
            return not self.verify_cart(now)
        except Exception:
            return False

    def restore_picked(self) -> None:
        """Before the forced ending: a line of Kyo's pick that was taken out of the cart, for an item that has no other line now, goes back in."""
        if not self.picked or self.left() < 5 or not self.roles.get("cart"):
            return
        cart = self.roles["cart"]
        now = self.cart_ids()
        covered = {self.line_item(i) for i in now}
        back = {i: q for i, q in self.picked.items() if i not in now and not self.sold_out(i) and self.line_item(i) not in covered}
        if not back:
            return
        lines = {**{i: self.cart_lines.get(i, 1) for i in now}, **back}
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 2 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        self.trace.append({"kind": "picked_restored", "lines": back})

    def retake_gone(self) -> str:
        """A cart line the store has said is sold out: Kyo's pick takes its place (the other lines stay)."""
        only = {self.line_item(i) for i in self.cart_ids() if i in self.gone} - {None}
        if not only:
            return ""
        return self.safe_retake("a cart line is sold out and will not come back", only=only)

    def safe_retake(self, why: str, final: bool = False, cheaper: bool = False, only: set[int] | None = None) -> str:
        """retake / retake_cheaper that never raises: a failure in Kyo's own pick must not end the episode."""
        try:
            return self.retake_cheaper() if cheaper else self.retake(why, final, only)
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "retake", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def retake(self, why: str, final: bool = False, only: set[int] | None = None) -> str:
        """Kyo sets the cart to its own pick (one listing per item at the requested quantities) and says so; "" when it has no complete pick,
        when the pick is over the budget, or when the cart already is the pick."""
        cart = self.roles.get("cart")
        if not cart or self.finished or self.retakes >= (5 if final else 4) or self.left() < (6 if final else 14):
            return ""
        if not final and self.elapsed() > CLOCK_SOFT:
            return ""
        if not final and self.pending_refs() and self.can_resolve() and self.resolve_holds < 3:
            # a requirement still points at another listing: follow it before any pick is made
            self.resolve_holds += 1
            before = self.step
            report = self.resolve_pending("order")
            if report or self.step != before:
                return ("Kyo looked for the listing a requirement points at. " + (report or "It could not settle the value yet.")
                        + " Check every cart line against the requirements as they now read, then check out.")
        try:
            plan = self.pick_plan(loose=final, only=only, held={i for i in self.cart_ids() if only and self.line_item(i) not in only})
            desperate = False
            if not plan and final and len(self.items) >= 2 and self.left() >= 8:
                # two or more lines: an ending without a purchase fails for certain, a pick that relaxes the word check may not
                self._desperate = True
                try:
                    plan = self.pick_plan(loose=True, only=only, held={i for i in self.cart_ids() if only and self.line_item(i) not in only})
                finally:
                    self._desperate = False
                desperate = bool(plan)
                if desperate:
                    self.trace.append({"kind": "desperate_pick", "plan": plan})
        except Exception as exc:
            self.trace.append({"kind": "guarded", "fn": "pick_plan", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""
        self.trace.append({"kind": "pick_plan", "why": why[:160], "plan": plan,
                           "needs": {str(n + 1): [m[:60] for m in it.musts + it.told if m not in it.dropped] for n, it in enumerate(self.items)}})
        if not plan:
            return ""
        now = self.cart_ids()
        if only:
            # only these items change: the other cart lines stay as they are
            keep = {i: self.cart_lines.get(i, 1) for i in now if (self.line_item(i) not in only)}
            lines = {**keep, **{ident: self.items[n - 1].qty for n, ident in plan.items() if n in only}}
            total = sum((self.price_of(i) or 0.0) * c for i, c in lines.items())
        else:
            lines = {ident: self.items[n - 1].qty for n, ident in plan.items()}
            total = self.plan_total(plan)
        if (isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget
                and not desperate and not self.__dict__.get("_price_first")):
            # over the budget: the same pick with the price first among the listings that state every requirement
            self._price_first = True
            try:
                again = self.pick_plan(loose=final, only=only, held={i for i in self.cart_ids() if only and self.line_item(i) not in only})
                if (not again or self.plan_total(again) > self.budget) and not only and self.budget_sweep(again or plan):
                    # the shelves listed under what each line may cost: the cheap listings that state the requirements are read now
                    again = self.pick_plan(loose=final, only=only, held=set()) or again
            finally:
                self._price_first = False
            if again:
                plan = again
                self.trace.append({"kind": "price_first_pick", "plan": plan})
                if only:
                    keep = {i: self.cart_lines.get(i, 1) for i in now if (self.line_item(i) not in only)}
                    lines = {**keep, **{ident: self.items[n - 1].qty for n, ident in plan.items() if n in only}}
                    total = sum((self.price_of(i) or 0.0) * c for i, c in lines.items())
                else:
                    lines = {ident: self.items[n - 1].qty for n, ident in plan.items()}
                    total = self.plan_total(plan)
        if (isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget
                and (not desperate or self.__dict__.get("_guarding"))):
            return ""
        if set(lines) == set(now) and all(self.cart_lines.get(i, lines[i]) == lines[i] for i in lines):
            return ""
        drop = [i for i in now if i not in lines]
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 2 > self.left():
            return ""
        used = self.step
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.cart_said = _dump(reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        self.retakes += 1
        self.picked = dict(lines)
        self.built = [i for i in lines if not self.sold_out(i)]
        self.version += 1
        self.trace.append({"kind": "retaken", "lines": lines, "drop": drop, "why": why[:160]})
        picks = "; ".join(f"item {n} ({self.items[n - 1].name}): {ident} at {self.price_of(ident):g}" if self.price_of(ident) is not None else f"item {n}: {ident}"
                          for n, ident in plan.items() if not only or n in only)
        return (f"Kyo set the cart itself ({self.step - used} steps) because {why}. Its pick, one listing per item, each naming the model code where one is needed "
                f"and passing the word check of every requirement: {picks}. Check out again; change a line only if its record clearly breaks a requirement.")

    def worst_level(self, number: int, ident: str, told_only: bool = False) -> float:
        """The least confirmed requirement of a read listing for an item: 1 when some requirement is stated by no attribute at all. With `told_only`
        just the requirements the shopper told in answer (the request's own wording may have been misread; an answer is explicit)."""
        item = self.items[number - 1]
        rec = self.full.get(ident)
        needs = [m for m in (item.told if told_only else item.musts + item.told) if m not in item.dropped]
        if not rec or not needs:
            return 0.0
        facts = self.page_parts(rec)["facts"]
        return max(self.confirm_level(m, facts) for m in needs)

    def retake_confirmed(self) -> str:
        """A cart line for which no attribute states a requirement the shopper told, while Kyo's pick for that item (cheapest of those that state it) does."""
        if self.retakes >= 4 or self.left() < 14 or self.elapsed() > CLOCK_SOFT or self.pending_refs():
            return ""
        key = self.cart_mark() + f"#{self.version}"
        done = self.__dict__.setdefault("_confirm_done", set())
        if key in done:
            return ""
        done.add(key)
        weak: dict[int, tuple[str, float]] = {}
        for ident in self.cart_ids():
            number = self.line_item(ident)
            if number is None or ident not in self.full:
                continue
            item = self.items[number - 1]
            if not [m for m in item.told if m not in item.dropped]:
                continue
            k = self.pick_key(number, ident, True)
            if k is not None and self.worst_level(number, ident, True) >= 1.0:
                weak[number] = (ident, k[0])
        if not weak:
            return ""
        plan = self.pick_plan()
        if not plan:
            return ""
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        better = {}
        for n, (ident, level) in weak.items():
            alt = plan.get(n)
            if not alt or alt == ident:
                continue
            k = self.pick_key(n, alt)
            if k is not None and k[0] < level and self.worst_level(n, alt, True) < 1.0:
                better[n] = alt
        if not better:
            return ""
        return self.retake("a cart line's attributes leave a requirement unconfirmed and a listing that states it was found", only=set(better))

    def retake_cheaper(self) -> str:
        """A 'cheapest' request whose cart is dearer than Kyo's verified pick."""
        if self.brief.get("pick") != "cheapest" or self.retakes >= 4 or self.left() < 14:
            return ""
        key = self.cart_mark() + f"#{self.version}"
        done = self.__dict__.setdefault("_cheaper_done", set())
        if key in done:
            return ""
        done.add(key)
        try:
            plan = self.pick_plan()
        except Exception:
            return ""
        if not plan:
            return ""
        cart_total = sum((self.price_of(i) or 0.0) * c for i, c in self.cart_lines.items() if i in self.rows)
        # a cart is only changed for a listing whose attributes state every requirement the pick knows
        changed = [(n, i) for n, i in plan.items() if i not in self.cart_lines]
        if any((self.pick_key(n, i) or (1,))[0] != 0 for n, i in changed):
            return ""
        if cart_total and self.plan_total(plan) < cart_total:
            return self.retake("a cheaper listing meets every requirement of the 'cheapest' pick (the pick costs "
                               f"{self.plan_total(plan):g}, the cart {cart_total:g})")
        return ""

    def assemble_from_audit(self) -> str:
        """The cart is empty (or over the budget) at a forced finish although the review named fitting listings for every item: set the cart from
        those (cheapest first, each passing the word check, total inside the budget) and check out."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.__dict__.get("_assembled") or not self.lead_items or not cart or not buy or self.finished:
            return ""
        self._assembled = True
        plan: dict[int, str] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            if not index or not 1 <= index <= len(self.items):
                continue
            for ident in ids:
                if not self.sold_out(ident) and ident in self.rows and not self.verify_line(index, ident) and ident not in plan.values():
                    plan[index] = ident
                    break
        if len(plan) < len(self.items):
            return ""
        lines = {ident: self.items[index - 1].qty for index, ident in plan.items()}
        total = sum((self.price_of(i) or 0) * n for i, n in lines.items())
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget and total < 20 * self.budget:
            return ""
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + 3 > self.left():
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
        if faults or self.left() < 1:
            return ""
        self.gate("order")
        if self.finished:
            return ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "assembled", "lines": lines, "reply": _clip(reply, 200)})
        return "assembled"

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and self.lead_items and (not holding or self.cart_over_budget()):
            try:
                if self.assemble_from_audit() and (self.finished or self.left() <= 0):
                    return ""
            except Exception as exc:
                self.trace.append({"kind": "guarded", "fn": "assemble_from_audit", "error": f"{type(exc).__name__}: {exc}"[:200]})
            if self.finished:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (not holding or self.cart_over_budget()) and not self.finished and self.left() >= 8:
            try:
                self.retake("the episode is ending without a usable cart", final=True)
            except Exception as exc:
                self.trace.append({"kind": "guarded", "fn": "retake", "error": f"{type(exc).__name__}: {exc}"[:200]})
            if self.finished:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and self.picked and not self.finished:
            try:
                self.restore_picked()
            except Exception as exc:
                self.trace.append({"kind": "guarded", "fn": "restore_picked", "error": f"{type(exc).__name__}: {exc}"[:200]})
            if self.finished:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
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
        # a requirement the request itself points at another listing for is followed before the actor's first move
        followed = self.resolve_pending("start") if self.pending_refs() else ""
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        if followed:
            intro += "\n\n" + followed
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        try:
            self.work(self.loop_cap)
        except Exception as exc:
            # a failure inside one turn must not end the episode with whatever cart is held: the loop goes on once more
            self.trace.append({"kind": "work_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            if not self.finished and not self.broke:
                self.work(self.loop_cap)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.elapsed() < CLOCK_SOFT + 20:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, 0.01, CLOCK_SOFT + 25)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls,
                           "secs": round(self.elapsed()), "cuts": self.cuts, "quiet_light": self.quiet_light, "mute_ok": self.mute_ok,
                           "follower_steps": self.resolve_steps, "follower_calls": self.resolve_calls, "follower_cost": round(self.extra, 5),
                           "decline_holds": self.decline_holds, "searches": self.searches, "msg_used": self.msg_used, "msg_limit": self.msg_limit})
        return self.trace

    def early_pick(self) -> str:
        """The interview is over, the steps are running short and the cart lacks a line for some item: Kyo's pick searches for those items now,
        while enough steps remain to read what it finds and to check out (once per episode)."""
        if self.__dict__.get("_early_done") or not self.asked or self.finished:
            return ""
        have = {self.line_item(i) for i in self.cart_ids()}
        missing = {n for n in range(1, len(self.items) + 1) if n not in have}
        # a bundle (three items or more) has no steps to spare for the actor's own wandering: Kyo's pick starts when a quarter of the steps are used
        window = int(self.max_steps * BUNDLE_EARLY) if len(self.items) >= 3 else max(26, int(self.max_steps * 0.4))
        if not missing or not (20 <= self.left() <= window) or self.elapsed() > CLOCK_SOFT - 40:
            return ""
        if self.pending_refs() and self.can_resolve():
            return ""
        self._early_done = True
        self.trace.append({"kind": "early_pick", "missing": sorted(missing), "left": self.left()})
        return self.safe_retake("the steps are running short and the cart has no line for item(s) " + ", ".join(str(n) for n in sorted(missing)), only=missing)

    def short_cart(self) -> bool:
        """A cart that is being built and still lacks a line: ending now fails whatever is done, so the loop may run a little longer."""
        return self.cart_touched and 0 < len(self.cart_ids()) < len(self.items) and self.left() > 12

    def work(self, loop_cap: int, bonus: float = 0.0, clock: float = CLOCK_SOFT) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap
               and self.cost < self.hard() + bonus + (0.01 if self.short_cart() else 0.0)
               and self.elapsed() < clock + (30 if self.short_cart() else 0)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            took = self.early_pick()
            if took:
                self.chat.append({"role": "user", "content": "[Kyo] " + took + "\n" + self.status()})
                continue
            try:
                carried = self.run_procedure()
            except Exception as exc:
                carried = ""
                self.trace.append({"kind": "guarded", "fn": "run_procedure", "error": f"{type(exc).__name__}: {exc}"[:200]})
            if carried:
                self.chat.append({"role": "user", "content": "[Kyo] " + carried + "\n" + self.status()})
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
