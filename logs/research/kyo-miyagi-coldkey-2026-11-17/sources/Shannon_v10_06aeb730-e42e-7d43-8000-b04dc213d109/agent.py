"""Errand (kyo, third rebuild): shops for one customer inside an ORO store world.

An Errand keeps the task frame and the chat and owns eleven parts, each holding its own state: the relay (model and
store calls), the toolkit (what each tool shows), the brief (the request, read once), the ledger (listing records),
the scout (pages and extra reads), the interviewer (questions to the shopper), the clerk (steps the request asks
for), the bulletin (notes that ride on results), the reviewer (checks before a checkout or an ending), the till (the
cart) and the dispatcher (the model's tool calls, passed through named guards). A light model walks the store; a
heavy one reads the request and judges. Waits between resent model calls are event waits; no clock is read."""
from __future__ import annotations

import json
import random
import re
import threading
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT_MODEL = "anthropic/claude-haiku-5.5"
HEAVY_MODEL = "anthropic/claude-sonnet-5.5"
SPARE_MODEL = "openai/gpt-6-luna"

MAX_TURNS = 42
MAX_MODEL_CALLS = 60
SOFT_SPEND_USD = 0.038
HARD_SPEND_USD = 0.052
# no model call of any kind once an episode has spent this much: ORO runs every episode in a process of its own, and the
# run's key (about $5) has to last for all of them; past the soft limit the heavy model's calls go to the light one
EPISODE_CEILING_USD = 0.060
# the closing judgement (which listing comes closest) may still run this much past the ceiling, on the light model
FINISH_ALLOWANCE_USD = 0.012
# the turn, model-call and spend limits above suit a request of this many steps; a request allowing more steps scales them up
CAP_STEP_BUDGET = 46
RETRY_WAITS = (4, 10, 20, 35)
RETRY_WAIT_TOTAL_S = 45
# a model turn that fails even after its resends: the episode pauses and tries again, a few times at most
FAILED_TURNS_MAX = 4
FAILED_TURN_PAUSE_S = 8.0
VALUE_CHARS = 450
ROW_CHARS = 160  # a search row's text in the chat: enough to pick what to open, and it is resent every turn
OBS_CHARS = 15000
HISTORY_CHARS = 100000
GATE_ROUNDS = 3
AUTO_PAGE_LIMIT = 10
EPISODE_SECONDS = 300
FINISH_RESERVE_S = 60
CALL_SECONDS = {LIGHT_MODEL: 12.0, HEAVY_MODEL: 9.5}
OTHER_CALL_SECONDS = 8.0
FAILED_CALL_SECONDS = 30.0
STORE_CALL_SECONDS = 0.3
# a reply's time grows with its length: seconds = a base per call plus a share per output token (fitted on arena replies)
REPLY_BASE_SECONDS = 6.0
SECONDS_PER_OUT_TOKEN = {LIGHT_MODEL: 0.0056, HEAVY_MODEL: 0.0100}
# a reply that reports its price came through the paid router (as on the network), which answers far faster than a
# login seat: such replies are counted at these rates
PRICED_REPLY_BASE_SECONDS = 2.0
PRICED_SECONDS_PER_OUT_TOKEN = {LIGHT_MODEL: 0.0025, HEAVY_MODEL: 0.0060}
PRICED_CALL_SECONDS = {LIGHT_MODEL: 3.5, HEAVY_MODEL: 5.0}
CHEAPER_READS = 3
# a refused checkout (a line sold out at the till) needs a removal, a new line and another checkout
SWAP_STEPS = 3
# Kyo walks the pages a requirement points to: a few rounds of one search or a few page reads, while enough steps remain
WALK_ROUNDS = 5
WALK_MIN_STEPS = 10
# a walk reads at most this many pages in all, and at most WALK_OPENS rows per search (a narrow shop search puts the
# page it wants among its first rows)
WALK_PAGES = 8
WALK_OPENS = 2
# what a one-item request silent on the no-fit case gets: the in-stock listing within the budget that fails the fewest requirements
CLOSEST_DEFAULT = "order the in-stock listing of the requested kind within the budget that fails the fewest requirements, keeping every requirement the request states in its own words before any the shopper added later"
SAMPLE_PER_KIND = 6

_llm_client = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env_client = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


SHOPPER_GUIDE = """You shop for a single customer in an online store, and you can act only through the store's tools. An episode scores 1 or 0, and only an exactly right finish scores 1: either an order that holds every requested item in its requested quantity, with each listing meeting every requirement still in force when you check out and the total within the budget, or, when the store truly offers nothing that meets all of it, an ending through the no-purchase tool done the way the request asks. When the request says to order the closest listing in that case, send the ending call once your search is done: Kyo then picks that closest listing and buys it.

HOW THE STORE WORKS
- Every tool call costs one step, and the steps are counted; thinking costs nothing. Each store gives its tools, arguments and result fields names of its own, so read every tool's description and schema and send argument names and allowed values exactly as written.
- Searches can be restricted: a category filter, a keyword mode that looks only at the first words, few rows per page, a token or an offset for the next page. When the answer hinges on seeing the whole set, compare the total the store reports with the rows you hold and keep paging with the exact token or offset it returned. Keyword queries work best short and with the value first (the wanted value, then the product noun). When a search turns up candidates, page through it and open its rows in batches with the side-by-side tool before you switch to other words; a requirement whose value lives in the attributes (packaging, material) rarely shows in titles, so a page-by-page read of the category or of one good query beats many new queries. An empty result from a filtered search usually means a wrong filter value or parameter: search again with the label spelled exactly as the result rows spell it, or with a different filter parameter.
- A search row is just a summary. Stores split a listing's record in different ways: one view can show its attributes (properties), another only its offer and description, and a side-by-side view can carry either. As soon as the tool that shows attributes is known, the ledger names it. Requirements on properties are judged from those attributes, so read them for every listing you might buy; one call that shows several listings saves steps when it shows what you need. Do not open a listing again as long as nothing about it has changed.
- You never write to the shopper yourself: Kyo's interviewer asks the questions, and the answers arrive in the conversation and in the ledger.
- The request can also call for steps along the way (an approval, a message, something to check or read first), and the ledger lists them. Those due just before the order or the ending are done by Kyo when you call the checkout or the ending tool, so do not wait for them and never end without buying on their account; do any others with the store's tools (questions about products go through Kyo's interviewer).

WHEN A LISTING FITS
- The listing's category label settles what kind of product it is: only the label that matches the requested item as a whole counts, never a nearby one. When two labels could both be the item (a shorthand, a synonym or a narrower label), the item's category is the one whose label uses the request's own words for the product; search and buy there.
- A requirement on a property is met when the listing's attribute for that property states the value. How the key is spelled does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough; for a number range or limit, that element has to be a single figure or range of that property, and a combined size of several dimensions (such as 40x40x2 cm) is no single figure and meets nothing. Values are compared word by word: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). Other words do not count (an abbreviation or a code is not the word behind it, and "teal green" is not "sea green"). Numbers are compared once units are converted, limits include their edge value, and a range has to lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides it: when it states something else, the listing fails, whatever the other keys or the title say. A key for some other property, even a related one (a handle colour is not the body's colour), decides nothing for that requirement. After opened listings, Kyo adds [evidence] lines that name the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value given under another attribute (a list of styles or features that includes it, for example) or in the title or description counts, but not a value named only among purchase options or variants to choose between (an options or variants key, a list of sizes, colours or packs on offer); a title or description that names several audiences or uses the one listing serves at once ("boys girls", "men's and women's") meets each of them. A number of a different kind does not (a weight is not a capacity), and neither does a bare figure whose text does not say what it measures.
- When the category label itself carries a requirement's value (a season or an audience in the label), the category meets that requirement.
- A requirement can point at another listing in place of a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle it first: find that listing. Keyword searches match titles and descriptions, and a shop's name is seldom among those words, so search the listing's kind together with the property values the description gives (in its category when the search takes one), and among the rows pick the one whose seller, shop or brand field names that shop; when no row names it, the shop may show only on a listing's full page (its fine print or seller note), so open in full, one by one, the rows whose values match the description until a page names that shop (try other value words or the second page before giving up); open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page holds the value. When the requirement asks for a value compatible with, going with or pairing with that listing, the value is the one that page names as its match or counterpart (or the value of the further listing named there), not that listing's own value of the property. The order counts only once every page of that chain has been opened. When the wanted value is anything other than that other listing's value, the rule excludes only listings with that exact value, and on its own it is never a reason to end without buying.
- When a listing you are about to buy sends you to another page about it, look at that page before you buy.
- When one requested item has to fit another (an accessory for a device), pick the two together from the listings' own words (a list of what it fits, a model or series code both pages show). When no accessory meeting its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without buying while that switch is still untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") only settle ties.
- Never buy anything the store shows as sold out; the total (price times quantity over all lines) may not exceed the budget, though it may equal it.

CHOOSING A LISTING
- When the request wants the cheapest (lowest total, smallest price), the order has to hold the lowest-priced listing of ALL fitting listings. Find the whole fitting set (page through it: rows often arrive in no price order, and Kyo reads the next pages of a short result on its own), check candidates from the lowest price upward, and take the first that fits completely. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request wants the most or the least of some measure, compare every fitting listing on that measure and take the extreme one; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: spend no steps hunting for a better one.
- Cart each item as soon as you have opened a listing that meets every requirement for it (under a cheapest, most or least pick: once you have compared the fitting set), then carry on. Stores often sell out a carted line soon after it goes in, and carting early leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES ON THE WAY
- The shopper or the store can change things: a new budget, a requirement added or removed, a price change, a listing sold out. Such changes appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result carries it, and the ledger lists the requirements in force. A refusal does not end the episode: apply every change, fix the cart and check out again.
- Read a chosen listing's record once more shortly before checking out, because price and stock can change without warning.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out in a call of its own.
- End without buying only after a real search turns up no listing that meets everything. Running low on steps proves nothing of the sort: then check out the best cart you have. Word that ending the way the request asks, and when a remark is needed, keep it brief and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then your next move."""


BRIEF_PROMPT = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools, then reply with ONE JSON object (compact, unindented, nothing else) holding these keys:

"items": one entry for each distinct product the shopper wants bought, each {"name": the kind of product in the request's own words, "shelf": the store's category label for it when some tool lists labels (copied exactly), else "", "quantity": how many units of that one listing the shopper is ordering now (an integer, 1 when unstated; a pair or a set sold as one listing is 1; "ordering one" is 1 even when the story says several people need one), "musts": the firm requirements on that product, each "property: constraint" carrying the exact value, limit, exclusion or closed list; an exclusion keeps its "not" ("anything other than neckline v-neck" or "anything but a glass lid" gives "neckline: not v-neck", "lid material: not glass"), never the excluded value alone. Use the request's own word for the property ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch") and never swap it for a nearby property. A must whose value comes from another listing is ONE must holding that listing's whole description ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): the words that describe that other listing (its shop, its kind, the "with <property> <value> and <property> <value>" after it) stay inside that single must and never turn into musts of this product}. Never list price, budget, stock or quantity among the musts. Words that only say the order is one unit (\"in a single pack\", \"just one bottle\", \"a single piece\") give the quantity and no must, only a wish for a listing sold as one piece (not a multi-piece or multi-buy option); a must about the packing needs the request to name that property with its value (\"with the pack type as multipack\"). The story around the request (who will use it, where it will go, what went wrong before, the occasion behind it) is background, not a must, unless it states a property's value or limit outright; and words that are part of what the product is called (a season, an audience or a use named along with the product) go in "name"; but when the request states such a word as the product's property (a property name followed by the word, or "specifically for" the word), it is a must as well. A must always carries a value the request states: a sentence that only says some property matters, hints at it, or speaks of it in a figure of speech, without naming the value, is not a must; leave it to the interviewer. Likewise words that only elaborate a value already stated in the same sentence (a comparison, a simile, or praise joined to it, as in \"in the blue family, lovely and bright as a summer sky\", or a second phrase set beside the value that restates it in other words, as in \"high pile, the deep sort\") add no must of their own: that sentence gives one must. A product the request names in a single phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements. A product the request mentions twice (once in the story, once with its order count) is still ONE item, with the order count as its quantity.
"budget": the most the shopper will pay in total, as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it applies per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are unavailable.
"pick": "cheapest" when the request wants the lowest price or the smallest total, "most" or "least" when it wants the extreme of some measure, else "any". Remarks that money is short are no pick rule.
"pick_measure": what "most"/"least" maximises or minimises, else "".
"messages": how many messages the shopper allows the agent to send (an integer), or null when unstated.
"steps": how many steps, moves or tool calls the request allows (an integer), or null.
"lookups": the musts (copied exactly) whose value has to be read first from another listing the request points at, else [].
"rules": every instruction the request gives on how to work or how to end, each in at most 12 words.
"wishes": the things the shopper would only like.
"ask_first": the things the request explicitly says to check with the shopper before choosing (empty when there are none).
"process": the steps the request demands beyond finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what has to be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (just before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps: saying that nothing fits and putting forward or ordering the closest listing belong in "fallback", never here. A step needs the request to demand it in plain words; never add one the request does not demand. An offer or permission to ask (being welcome to check a detail first, to ask whatever helps, to take one's time) is no step: the interviewer asks the questions. Wanting to know, settle or be told something about the product itself (how it is made, what it is made of, how it looks or feels, even "when I order") is a preference about one of its properties for the interviewer to ask about, never a step. A step is an instruction about the process itself, such as being told the total or asked before the order is placed.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it is silent.
"fallback": what the request says to do when nothing fits every requirement, in its own terms and in at most 40 words (keep every rule it gives for choosing, such as how prices compare), or null when it says nothing about that.
"fallback_buy": true when that instruction names some listing to order, offer, suggest or put forward anyway (the closest one, the one falling short on the fewest counts, a cheaper or near one), else false; false only when it says to end without naming or buying any listing.
"roles": an object naming the exact tool for each job, or null where no tool does it: "search", "open" (shows one listing's attributes or properties; when there are two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that tool is not "open"), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"price_cap": {"tool": a search or listing tool, "arg": its argument that sets the highest price a listed row may have} when some tool takes such an argument, else null.
"category_arg": {"tool": that same tool, "arg": its argument that limits the rows to one store category} when it takes one, else null.
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


ASK_PLANNER = """You choose the questions a shopping agent asks a shopper. Shoppers hold a few firm preferences they never wrote down, each about one property of one requested product. A question brings a preference out only when it names that exact property, and each question is answered only about the first property it names, so every question uses up one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and a few example values. An unwritten preference can also concern a property that no listed key names.

Go through the properties in this order:
0. What the request itself tells you to check with the shopper (listed under ASK FIRST when there is any), each named by the property of the product it is about; skip one that concerns only stock, price, delivery or showing listings, since no product property answers it.
1. Hinted properties. The request's story often points to an unwritten preference without stating its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it goes in, the person who will use or wear it, what goes inside it, or what the product is like to eat, drink, wear, hold or hear in use (a hint about how something goes down points to a flavour or taste key, about how it feels to a texture or material key, about how it is made or put together, what a photo cannot show of it, to a construction or build key). For each such sentence, name the one property it points to and the item it is about, and rank them by how directly the sentence speaks about the product itself, strongest first: a sentence saying someone will check, judge or need to know something about the product (how it is made, what it does for its user, how it must look or work) comes before a passing remark; a remark about the shopper's own day, body, mood or surroundings that says nothing about the product (tired arms, a messy hallway, a busy week) is no hint at all.
A comparison or figure of speech about the product or its job is a hint too: name the property that the likeness is about. Name a hinted property by the listed key it points to (a hint about who will wear it to a gender or age key, about its looks to a colour or finish key, about where or when it is used to an occasion key); when no listed key fits a hint, name the property it points to in plain words all the same (how a food or drink goes down points to its flavour), since the opened listings are only a sample of the item's listings. A hint never points to shipping, delivery, price or stock.
2. After that, the listed keys, ranked. An unwritten preference is about a property that tells listings apart, so favour keys on which the listings differ (several different values among the listings that carry them) and keys that most of the item's listings carry; a key nearly all listings share one value of is a poor pick, and so is a key seen on just one or two listings. Among those, favour plain properties a shopper has a taste about, whose values are short and distinct (a colour, a material, a pattern, a style, a form, a packaging type, an occasion, a closure, a fitting or connector type), over keys whose values are long lists or free text (features, usage notes, descriptions). Sizes, measurements and counts (a size, a length, a capacity, a screen or display size, a memory, a pack size, a number of pieces) hold unwritten preferences as often as named kinds do: rank them among those by the same tests (how many listings carry the key, how much its values differ), never after them. The key that nearly every opened listing of the item carries, with values that differ, belongs among the first picks unless the request settles it.
Leave out a property the request already settles for that item, a second name for a property already picked (two keys whose names share the property word, such as "shape" and "container shape", are one property), a key listing purchase variants (a key often called options or variants, mixing sizes, colours and pack counts), identifiers (a key whose value names only the listing itself: its own model number, part number, series or serial code; a key naming a component or type the product is built with, such as its processor, chipset or engine, is a property people choose by and no identifier), warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, every word of the key in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never cut down to "colour" or "finish"); only the property's name, with no example values, no brackets and no second property.
Pick COUNT, best first (fewer only when fewer properties are left): an unused question is wasted, so after the strong picks fill the rest with the next best keys. Reply with ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


REPLY_READER = """A shopping agent asked a shopper some questions. Read every answer and reply with ONE JSON object and nothing else. A value the shopper rules out ("any but X", "everything except X") is the need "property: not X"; a value they insist on even in negative words ("I'll accept nothing but X", "I'd never live it down if I settled on anything but X", "anything apart from X gets binned") is "property: X".
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value must be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains the message asked about too many things or wants one question at a time; not answers that only say they did not follow it], "budget": a new total limit as a number written exactly as the shopper writes it (in the same unit, never converted), or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need after the property the answer itself names, which may not be the one asked about; several answers about one property make a single need. Copy the value word for word as the answer gives it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or close to the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or rephrase it. The properties that describe the other product are not needs of this item.
- "free": the property of each question the shopper said they do not care about.
- Homely or roundabout wording that names a value still states that value."""


NOTE_READER = """While a shopping agent was working, the store showed the note below (it may come from the shopper or from the store). Reply with ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports a malformed or rejected call, "other" for anything else, "budget": the new total limit as a number in the request's unit, exactly as the note writes it (never converted), or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [ids of listings said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A requirement that gets replaced is a drop of the old one plus an add of the new one. Report every change the note makes."""


CART_CHECK = """You are the final check before a shopping agent checks out. Judge from the records shown whether this cart is exactly right.
How to read them: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular or plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units are converted, limits include the edge, a range has to lie wholly inside, and for a number range or limit, that element has to be a single figure or range of that property, and a combined size of several dimensions (such as 40x40x2 cm) is no single figure and meets nothing); a requirement to be compatible with, go with or pair with another listing is met by the value that listing's page names as its match or counterpart; an attribute whose key names the requirement's own property decides, and when it states something else the listing fails; a key for some other property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value given under another attribute or in the title or description counts, unless it is named there only as one of several options or variants on offer to choose between (a listing that names several audiences or uses it serves at once, such as "boys girls", meets each of them; a number of a different kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself carries is met by that category; the category label has to match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Every requested item has to be its own cart line at the requested quantity, nothing else may be in the cart, nothing may show as out of stock, and the total of price times quantity over all lines has to stay within the budget. A listing's newest record wins.
When the pick rule is cheapest, most or least, look at the other listings shown too: if one of them clearly fits every requirement and beats the cart's choice under that rule, the cart is wrong.
Reply with ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [ids of listings that should replace a cart line, best first]}. Give ok=false only for a problem the records clearly show or for a requirement that no record addresses."""


MATCH_RULES = """the listing's attribute for a property decides when its key names that very property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge, for a number range or limit, that element has to be a single figure or range of that property, and a combined size of several dimensions (such as 40x40x2 cm) is no single figure and meets nothing); a requirement to be compatible with, go with or pair with another listing is met by the value that listing's page names as its match or counterpart; a key for some other property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value given under another attribute or in the title or description counts, unless it is named there only as one of several options or variants on offer to choose between (a listing that names several audiences or uses it serves at once, such as "boys girls", meets each of them; a bare figure that does not say what it measures counts for nothing); a requirement the category label itself carries is met by that category; the category label has to match the item, and of two labels that could both be the item, only the one using the request's own words for it counts; a rule of being unlike another listing's value counts as met unless the records show that same value; listings out of stock and listings over the budget do not fit."""

NO_BUY_REVIEW = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, using these reading rules: """ + MATCH_RULES + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can neither pass nor fail on a property the row does not show.
Reply with ONE JSON object and nothing else: {"fits": {"item number": [ids of listings that seem to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, within the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

LOOKUP_SPLITTER = """A shopper's request was turned into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that identify that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes wrongly taken as requirements on the product being bought. Going by how the request is worded, say which of the OTHER REQUIREMENTS below only describe that other listing. Reply with ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when each one is a real requirement on the product]}"""

LINK_TRACER = """A shopping agent is about to check out. Some requirements take their value from another listing rather than stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only once the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, page not read).
Reply with ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the page before describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the value the requirement asks for as that last page gives it: when the requirement wants a value compatible with, going with or pairing with that listing, the value the page names as its match or counterpart; otherwise the page's own value of the property; or ""}]}
Give each chain's links in order and stop at the first link whose page is not OPENED or not found, because its words are still unknown."""

LINK_WALKER = """A shopping agent has to settle requirements whose value sits on another listing's page. A listing pointed at is known by its own page: the fine print on its page often names its shop and details while the row's brand field names another shop, so a row alone rarely proves which listing it is. A page can also send the reader on to a further listing for the property. Using the records below (OPENED: the page read in full, fine print included; ROW: a search row only), follow each requirement's chain link by link from the requirement's own description.
Reply with ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "names_listing": true when the requirement points at one particular other listing whose page holds the value, which it names by a shop or brand together with its kind or details; false when it names no shop or brand, only asks to suit or match another item of this order, or only states a condition, "links": [{"id": the listing id when an OPENED record below is that link, else "", "about": how the requirement or the page before describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the value the requirement asks for as that last page gives it (when the requirement wants a value compatible with or pairing with that listing, the value the page names as its match), or ""}], "open": [at most 2 ids of rows that a search listed under SEARCHES DONE returned, best first, that may be the first link no OPENED page settles yet; [] when no such search has returned a likely row], "search": the arguments for the search tool below that would bring up that link: the shop's name with the product kind and at most ONE of the details the description gives (a search holding every detail finds nothing; after a search listed under SEARCHES DONE came back empty, drop the detail); give it whenever a chain is not done, and null only when every chain is done}
Give each chain's links in order and stop at the first link whose page is not OPENED, because its words are still unknown."""

STEP_GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below on the way. From the numbered RECORD of what the agent has done, decide whether each step due before this STAGE is done for the cart as it is now. A step is done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart's last change and its reply grants it; questions about a product's properties do no step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step only, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Reply with ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

CLOSEST_PICK = """Nothing a shopping agent found meets every requirement, and the shopper's request says what to do then: FALLBACK
Judge the listing records shown with these reading rules: """ + MATCH_RULES + """
For each OPENED listing of a requested kind, list the requirements it fails; being out of stock and costing more than the budget count as failed requirements too. Then choose, for each requested item, the listing the request's instruction leads to (the one failing the fewest requirements, with prices compared the way the request says). Choose only among listings still available to buy whose price stays within the budget: the checkout refuses any other. When the request makes one item fit or be compatible with another, choose them together: a listing whose page names the models or codes it serves must serve the code that the other chosen listing's page names.
Reply with ONE short JSON object and nothing else, "pick" first: {"pick": {"1": "listing id", ...one entry per item number}, "misses": {"listing id": ["short failed requirement", ...] for the picked listings and at most 3 others}, "reason": "one sentence"}"""

SWAP_FINDER = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, using these reading rules: """ + MATCH_RULES + """
Reply with ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number, as the record shows it} for every listing that seems to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

RESULT_SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Reply with ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched overall (not how many are shown), or null, "next": the key holding the value that requests the next page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, saying whether the listing can be bought now, or null}"""


def _dumps(datum: Any) -> str:
    try:
        return json.dumps(datum, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(datum)


def _clipped(datum: Any, max_len: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(datum, str):
        return datum[:max_len] + "..." if len(datum) > max_len else datum
    if isinstance(datum, list):
        return [_clipped(entry, max_len, depth + 1) for entry in datum]
    if isinstance(datum, dict):
        return {key: _clipped(entry, max_len, depth + 1) for key, entry in datum.items()}
    return datum


_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
# a family of model codes a listing serves ("AB-45xx"), and a full model code ("AB-4510")
_FAMILY_RE = re.compile(r"\b([A-Z]{2,})[- ]?(\d+)[xX]{1,4}\b")
_CODE_RE = re.compile(r"\b([A-Z]{2,})[- ]?(\d{2,6})\b(?![xX])")


def _parse_object(text: str) -> dict[str, Any] | None | bool:
    """The text read as one JSON object; False when it is not JSON at all."""
    try:
        reply_obj = json.loads(text)
    except ValueError:
        return False
    return reply_obj if isinstance(reply_obj, dict) else None


def _first_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence_hit = _FENCE_RE.match(text)
    if fence_hit:
        text = fence_hit.group(1)
    full_cart = _parse_object(text)
    if full_cart is not False:
        return full_cart
    remaining, right_at = text.find("{"), text.rfind("}")
    if not 0 <= remaining < right_at:
        return None
    line_props = _parse_object(text[remaining:right_at + 1])
    return line_props if line_props is not False else None


def _plain_words(key: str) -> str:
    split_text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", split_text).strip().lower()


def _token_set(text: str) -> frozenset[str]:
    return frozenset(word_text[:-1] if len(word_text) > 3 and word_text.endswith("s") else word_text for word_text in _plain_words(text).split())


def _key_matches(prop_name: frozenset[str], key: frozenset[str]) -> bool:
    if not (prop_name and key):
        return False
    return prop_name <= key or (len(key) >= 2 and key <= prop_name)


def _schema_types(schema: dict[str, Any]) -> list[str]:
    declared_type = schema.get("type")
    if isinstance(declared_type, str):
        return [declared_type]
    return [entry for entry in declared_type or [] if isinstance(entry, str)]


_TYPE_TESTS = {
    "string": lambda v_: isinstance(v_, str),
    "integer": lambda v_: isinstance(v_, int) and not isinstance(v_, bool),
    "number": lambda v_: isinstance(v_, (int, float)) and not isinstance(v_, bool),
    "boolean": lambda v_: isinstance(v_, bool),
    "array": lambda v_: isinstance(v_, list),
    "object": lambda v_: isinstance(v_, dict),
    "null": lambda v_: v_ is None,
}


def _is_type(datum: Any, kind_name: str) -> bool:
    type_test = _TYPE_TESTS.get(kind_name)
    return type_test(datum) if type_test else True


_NUM_RE = re.compile(r"\s*-?\d+(\.\d+)?\s*")


def _coerced(datum: Any, type_names: list[str]) -> Any:
    """The first conversion toward one of the schema's kinds that applies (the value as it was when none does)."""
    for kind_name in type_names:
        if kind_name == "integer" and isinstance(datum, float) and datum.is_integer():
            return int(datum)
        if kind_name in ("integer", "number") and isinstance(datum, str) and _NUM_RE.fullmatch(datum):
            item_ix = float(datum)
            return int(item_ix) if kind_name == "integer" or item_ix.is_integer() else item_ix
        if kind_name == "string" and isinstance(datum, (int, float)) and not isinstance(datum, bool):
            return str(datum)
        if kind_name == "boolean" and isinstance(datum, str) and datum.strip().lower() in ("true", "false"):
            return datum.strip().lower() == "true"
        if kind_name == "array" and not isinstance(datum, list) and datum is not None:
            return [datum]
    return datum


def _enum_check(datum: Any, schema: dict[str, Any], at_path: str, faults: list[str]) -> Any:
    enum_values = schema.get("enum")
    if not isinstance(enum_values, list) or datum in enum_values:
        return datum
    if isinstance(datum, str):
        same_enum = [opt_ for opt_ in enum_values if isinstance(opt_, str) and opt_.lower() == datum.strip().lower()]
        if same_enum:
            datum = same_enum[0]
    if datum not in enum_values:
        faults.append(f"{at_path}: allowed values are {enum_values}")
    return datum


def _length_check(datum: str, schema: dict[str, Any], at_path: str, faults: list[str]) -> str:
    max_chars, min_chars = schema.get("maxLength"), schema.get("minLength")
    if isinstance(max_chars, int) and len(datum) > max_chars:
        datum = datum[:max_chars]
    if isinstance(min_chars, int) and len(datum) < min_chars:
        faults.append(f"{at_path} is too short")
    return datum


def _range_check(datum: Any, schema: dict[str, Any], at_path: str, faults: list[str]) -> None:
    low_val, top_bound = schema.get("minimum"), schema.get("maximum")
    if isinstance(low_val, (int, float)) and datum < low_val:
        faults.append(f"{at_path} is below {low_val}")
    if isinstance(top_bound, (int, float)) and datum > top_bound:
        faults.append(f"{at_path} is above {top_bound}")


def _count_check(datum: list[Any], schema: dict[str, Any], at_path: str, faults: list[str]) -> None:
    max_entries, fewest = schema.get("maxItems"), schema.get("minItems")
    if isinstance(max_entries, int) and len(datum) > max_entries:
        faults.append(f"{at_path} allows at most {max_entries} entries")
    if isinstance(fewest, int) and len(datum) < fewest:
        faults.append(f"{at_path} needs at least {fewest} entries")


def _check_limits(datum: Any, schema: dict[str, Any], at_path: str, faults: list[str]) -> Any:
    """Enum, length, range and entry-count rules; a string over its length limit is cut."""
    datum = _enum_check(datum, schema, at_path, faults)
    if isinstance(datum, str):
        datum = _length_check(datum, schema, at_path, faults)
    if isinstance(datum, (int, float)) and not isinstance(datum, bool):
        _range_check(datum, schema, at_path, faults)
    if isinstance(datum, list):
        _count_check(datum, schema, at_path, faults)
    return datum


def _fit_schema(datum: Any, schema: Any, at_path: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return datum
    type_names = _schema_types(schema)
    if type_names and not any(_is_type(datum, kind_name) for kind_name in type_names):
        datum = _coerced(datum, type_names)
        if not any(_is_type(datum, kind_name) for kind_name in type_names):
            faults.append(f"{at_path} should be {'/'.join(type_names)}")
            return datum
    datum = _check_limits(datum, schema, at_path, faults)
    if isinstance(datum, list) and isinstance(schema.get("items"), dict):
        datum = [_fit_schema(entry, schema["items"], f"{at_path}[{i_}]", faults) for i_, entry in enumerate(datum)]
    if isinstance(datum, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            faults.extend(f"{at_path}.{tool_name} is not an allowed argument" for tool_name in datum if tool_name not in props)
        faults.extend(f"{at_path}.{tool_name} is required" for tool_name in schema.get("required") or [] if tool_name not in datum)
        datum = {key: _fit_schema(entry, props.get(key), f"{at_path}.{key}", faults) if key in props else entry for key, entry in datum.items()}
    return datum


def _listify(datum: Any) -> list[Any]:
    if isinstance(datum, list):
        return datum
    return [datum] if isinstance(datum, (str, dict)) and datum else []


_INT_RE = re.compile(r"\s*\d+(\.0+)?\s*")


def _as_int(datum: Any) -> int | None:
    if isinstance(datum, bool):
        return None
    if isinstance(datum, int):
        return datum
    if isinstance(datum, float):
        return int(datum) if datum.is_integer() else None
    if isinstance(datum, str) and _INT_RE.fullmatch(datum):
        return int(float(datum))
    return None


def _id_shaped(datum: Any, keyed: bool = False) -> bool:
    if not isinstance(datum, str):
        return False
    attrless = datum.strip()
    if not 3 <= len(attrless) <= 64 or " " in attrless:
        return False
    return keyed or bool(re.search(r"\d", datum))


def _guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    lead_key, lead_score = None, None
    for key in rows[0]:
        keyed = "id" in str(key).lower()
        datums = [row_rec.get(key) for row_rec in rows]
        if not all(_id_shaped(v_, keyed) for v_ in datums):
            continue
        datums = [v_.strip() for v_ in datums]
        if len(rows) > 1 and len(set(datums)) < len(datums):
            continue
        rank_score = (keyed, sum(map(len, datums)) / len(datums))
        if lead_score is None or rank_score > lead_score:
            lead_key, lead_score = key, rank_score
    return lead_key


def _mentions(node: Any, lid: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(_mentions(entry, lid, depth + 1) for entry in node.values())
    if not isinstance(node, list):
        return False
    for entry in node:
        if isinstance(entry, dict) and lid in [str(x_).strip() for x_ in entry.values() if isinstance(x_, str)]:
            return True
    return any(_mentions(entry, lid, depth + 1) for entry in node)


def _attrs_of(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best: dict[str, Any] | None = None
    for datum in node.values():
        if not isinstance(datum, dict) or len(datum) < 2:
            continue
        scalar_fields = {k_: v_ for k_, v_ in datum.items() if isinstance(v_, (str, int, float)) and not isinstance(v_, bool)}
        if len(scalar_fields) >= 2 and (best is None or len(scalar_fields) > len(best)):
            best = scalar_fields
    return best


class Want:
    """One requested product: its name, store label, quantity and needs (the request's, the shopper's, the dropped ones)."""

    def __init__(self, tool_name: str, label_text: str, units_wanted: int, firm_needs: list[str]):
        self.name = tool_name
        self.shelf = label_text
        self.qty = units_wanted
        self.musts = firm_needs
        self.heard_needs: list[str] = []
        self.dropped: list[str] = []
        self.dont_care: list[str] = []


_WAITS_ENABLED = [True]
def _wait(pause_s: float) -> None:
    """A pause between resends of a failed model call (an event wait, no clock read)."""
    if _WAITS_ENABLED[0]:
        threading.Event().wait(pause_s)


def _TEST_NO_PAUSE() -> None:
    _WAITS_ENABLED[0] = False


class _Pending:
    """One tool call of a model turn on its way through Kyo's guards."""

    def __init__(self, call_id: str, tool_name: Any, raw: Any) -> None:
        self.call_id, self.name, self.raw = call_id, tool_name, raw
        self.args: dict[str, Any] = {}
        self.extra = ""
        self.side_key: str | None = None
        self.id_arg: str | None = None
        self.is_final = False
        self.mark = ""


class Relay:
    """Model and store calls, step counting and the text shown back."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.credit_out = False
        self.no_credit = False
        self.waited = 0
        self.cost = 0.0
        self.spent_out = False
        self.llm_calls = 0
        self.clock_s = 0.0
        self.last_out_tokens: int | None = None
        self.last_priced = False
        self.priced_replies = 0
        self.plain_replies = 0
        self.trace: list[dict[str, Any]] = []

    def cap_scale(self) -> float:
        return max(1.0, self.owner.max_steps / CAP_STEP_BUDGET)

    def call_cap(self) -> int:
        return int(MAX_MODEL_CALLS * self.cap_scale())

    def soft_spend(self) -> float:
        # spend is not scaled with the step budget: the run's key is shared by every episode
        return SOFT_SPEND_USD

    def hard_spend(self) -> float:
        return HARD_SPEND_USD

    def left(self) -> int:
        """Steps still usable: the step budget, cut down when the counted clock says the episode time is running out."""
        return min(self.steps_left(), self.time_turns())

    def steps_left(self) -> int:
        return self.owner.max_steps - self.owner.steps_taken

    def time_turns(self) -> int:
        """How many more turns (one light-model reply and one store call each) fit before the finishing reserve, by counted seconds."""
        rates = PRICED_CALL_SECONDS if self.priced_replies > self.plain_replies else CALL_SECONDS
        turn_s = rates.get(LIGHT_MODEL, OTHER_CALL_SECONDS) + STORE_CALL_SECONDS
        return max(0, int((EPISODE_SECONDS - FINISH_RESERVE_S - self.clock_s) // turn_s))

    def messages_left(self) -> int:
        return self.owner.brief.max_messages - self.owner.interviewer.sent_count - max(0, self.owner.brief.reserve_messages - self.owner.clerk.gate_messages)

    def cart_key(self) -> str:
        return _dumps(sorted(self.owner.till.lines.items()))

    def kept_steps(self) -> int:
        needs_talk = any(s_["when"] == "order" and s_["messages"] > 0 for s_ in self.owner.brief.duty_items)
        return 2 if needs_talk and self.owner.interviewer.sent_count < self.owner.brief.max_messages and self.owner.brief.duties_at("order") else 1

    def ask_models(self, messages: list[dict[str, Any]], seats: list[str], max_tokens: int, tool_defs: list[dict[str, Any]] | None = None,
                   finishing: bool = False) -> dict[str, Any] | None:
        for model_id in seats:
            if self.llm_calls >= self.call_cap() + (2 if finishing else 0):
                return None
            if self.cost >= EPISODE_CEILING_USD + (FINISH_ALLOWANCE_USD if finishing else 0):
                self.spent_out = True
                return None
            if model_id == HEAVY_MODEL and (self.cost >= self.soft_spend() or (finishing and self.cost >= EPISODE_CEILING_USD)):
                if LIGHT_MODEL in seats:
                    continue
                model_id = LIGHT_MODEL
            self.llm_calls += 1
            message, status = self._tries(model_id, messages, max_tokens, tool_defs)
            if message is not None:
                # a credit refusal can be a passing burst (a shared in-flight budget): a reply that comes through lifts it
                self.no_credit = False
                return message
            if status == 402:
                self.no_credit = True
                return None
        return None

    def llm_call(self, model_id: str, messages: list[dict[str, Any]], max_tokens: int, tool_defs: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        body: dict[str, Any] = {"model": model_id, "messages": self._with_cache_marks(model_id, messages), "max_tokens": max_tokens, "temperature": 0}
        if tool_defs:
            body["tools"] = tool_defs
            body["tool_choice"] = "auto"
            if model_id == LIGHT_MODEL:
                # the walker's turns need little hidden reasoning: it costs output tokens (money and seconds) on every turn
                body["reasoning"] = {"effort": "low"}
        wire_result = _llm_client.post_verbose("/inference/chat/completions", body)
        payload = wire_result.data
        if isinstance(payload, dict) and payload.get("choices"):
            return self._read_choice(model_id, payload)
        return self._llm_failed(model_id, wire_result.error or {}, payload)

    def pause(self, pause_s: float) -> None:
        """A counted wait (no clock is read): the episode clock moves on by the pause."""
        self.clock_s += pause_s
        _wait(pause_s)

    def ask_json(self, prompt: str, seats: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model_id in seats:
            message = self.ask_models([{"role": "user", "content": prompt}], [model_id], max_tokens)
            reply_obj = _first_object((message or {}).get("content"))
            if reply_obj is not None:
                return reply_obj
            if self.credit_out or self.no_credit or self.spent_out or self.llm_calls >= self.call_cap():
                return None
        return None

    def env_call(self, tool_name: str, call_args: dict[str, Any]) -> dict[str, Any] | None:
        envelope = dict(self.owner.binding)
        envelope["step"] = self.owner.steps_taken + 1
        envelope["action"] = {"name": tool_name, "args": call_args}
        for _ in range(4):
            wire_result = _env_client.post_verbose("/environment/call", envelope)
            self.clock_s += STORE_CALL_SECONDS
            if isinstance(wire_result.data, dict):
                return self._took(tool_name, call_args, wire_result.data)
            if not self._retryable(wire_result.error or {}, envelope):
                return None
        return None

    def render(self, observed: Any) -> str:
        text = _dumps(_clipped(self._short_rows(observed), VALUE_CHARS))
        if len(text) > OBS_CHARS:
            text = text[:OBS_CHARS] + "...(cut)"
        return text

    def _short_rows(self, observed: Any) -> Any:
        """A search page as the walker sees it: each row of a list of listings keeps only a short text per field (the ledger keeps them whole)."""
        if isinstance(observed, list):
            if len(observed) >= 2 and all(isinstance(r_, dict) for r_ in observed) and sum(1 for r_ in observed if self.owner.ledger.record_of(r_, True, 1)) >= 2:
                return [_clipped(r_, ROW_CHARS) for r_ in observed]
            return [self._short_rows(r_) for r_ in observed]
        if isinstance(observed, dict):
            return {k_: self._short_rows(v_) for k_, v_ in observed.items()}
        return observed

    def trim_history(self) -> None:
        chars_total = sum(len(_dumps(n_.get("content"))) for n_ in self.owner.chat)
        if chars_total <= HISTORY_CHARS:
            return
        tool_defs = [lid_ for lid_, n_ in enumerate(self.owner.chat) if n_.get("role") == "tool"]
        for pos in tool_defs[: max(1, len(tool_defs) // 2)]:
            content = self.owner.chat[pos].get("content") or ""
            if len(content) > 600:
                lids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                lids = [lid_ for lid_ in dict.fromkeys(lids) if lid_ in self.owner.ledger.rows][:20]
                self.owner.chat[pos]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(lids)})"

    def _tries(self, model_id: str, messages: list[dict[str, Any]], max_tokens: int, tool_defs: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One model's attempts: a passing failure (no credit, a rate limit, a timeout, a server error) waits and resends,
        a timeout once at most, and all the waits of the episode together stay within WAIT_TOTAL."""
        resends = 0
        while True:
            message, status = self.llm_call(model_id, messages, max_tokens, tool_defs)
            self.clock_s += FAILED_CALL_SECONDS if message is None and status in (None, 408) else self._reply_seconds(model_id, message)
            if message is not None:
                return message, status
            may_resend = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
            if not may_resend or resends >= (1 if status in (None, 408) else len(RETRY_WAITS)) or self.waited + RETRY_WAITS[resends] > RETRY_WAIT_TOTAL_S:
                return None, status
            wait_s = RETRY_WAITS[resends] * (0.8 + 0.4 * random.random())
            self.waited += RETRY_WAITS[resends]
            self.clock_s += wait_s
            resends += 1
            _wait(wait_s)

    @staticmethod
    def _with_cache_marks(model_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Anthropic models: the first and the last message carry a cache mark."""
        if not model_id.startswith("anthropic/") or len(messages) <= 1:
            return messages
        sent = [dict(n_) for n_ in messages]
        for pos in (0, len(sent) - 1):
            content = sent[pos].get("content")
            if isinstance(content, str) and content:
                sent[pos]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return sent

    def _took(self, tool_name: str, call_args: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        self.owner.steps_taken = payload["step"] if isinstance(payload.get("step"), int) else self.owner.steps_taken + 1
        if payload.get("finished"):
            self.owner.done = True
        view = payload.get("observation")
        if tool_name in self.owner.kit.page_tools() and not (isinstance(view, dict) and view.get("error")):
            for datum in call_args.values():
                self.owner.ledger.pages_read.update(v_.strip() for v_ in (datum if isinstance(datum, list) else [datum]) if isinstance(v_, str))
        return payload

    def _step_conflict(self, body: str, envelope: dict[str, Any]) -> str:
        """A 409: the episode is over ("stop"), or the step number to resend with is known ("retry")."""
        low_val = body.lower()
        if "finish" in low_val or "quarantin" in low_val or "finaliz" in low_val:
            self.owner.done = True
            return "stop"
        found = re.search(r"expected (\d+)", body)
        if found:
            self.owner.steps_taken = int(found.group(1)) - 1
            envelope["step"] = self.owner.steps_taken + 1
            return "retry"
        if "reuse" in low_val or "different action" in low_val:
            self.owner.steps_taken += 1
            envelope["step"] = self.owner.steps_taken + 1
            return "retry"
        return "stop"

    def _reply_seconds(self, model_id: str, message: dict[str, Any] | None) -> float:
        """Counted seconds of one reply: by its output tokens when the reply reports them, else the model's flat rate."""
        out_tokens, self.last_out_tokens = self.last_out_tokens, None
        priced, self.last_priced = self.last_priced, False
        if priced:
            if message is None or out_tokens is None or model_id not in PRICED_SECONDS_PER_OUT_TOKEN:
                return PRICED_CALL_SECONDS.get(model_id, OTHER_CALL_SECONDS)
            return PRICED_REPLY_BASE_SECONDS + PRICED_SECONDS_PER_OUT_TOKEN[model_id] * out_tokens
        if message is None or out_tokens is None or model_id not in SECONDS_PER_OUT_TOKEN:
            return CALL_SECONDS.get(model_id, OTHER_CALL_SECONDS)
        return REPLY_BASE_SECONDS + SECONDS_PER_OUT_TOKEN[model_id] * out_tokens

    def _read_choice(self, model_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        usage_info = payload.get("usage") or {}
        out_tokens = usage_info.get("completion_tokens")
        self.last_out_tokens = out_tokens if isinstance(out_tokens, int) and not isinstance(out_tokens, bool) and out_tokens >= 0 else None
        # a cost of 0 marks a reply nobody paid for (a login-served call, timed like any unpriced reply)
        self.last_priced = isinstance(usage_info.get("cost"), (int, float)) and not isinstance(usage_info.get("cost"), bool) and usage_info["cost"] > 0
        if self.last_priced:
            self.priced_replies += 1
        else:
            self.plain_replies += 1
        if isinstance(usage_info.get("cost"), (int, float)):
            self.cost += float(usage_info["cost"])
        message = payload["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message, 200
        self.trace.append({"kind": "llm_fail", "model": model_id, "status": 0})
        return None, 0

    def _llm_failed(self, model_id: str, failure: dict[str, Any], payload: Any) -> tuple[None, Any]:
        status = failure.get("status")
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            status = _as_int(payload["error"].get("code"))
            failure = {"body": payload["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model_id, "status": status, "body": str(failure.get("body") or "")[:200]})
        return None, status

    def _retryable(self, failure: dict[str, Any], envelope: dict[str, Any]) -> bool:
        """A failed store call logged; True when it is worth sending once more."""
        status = failure.get("status")
        body = str(failure.get("body") or "")
        self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
        if status == 409:
            return self._step_conflict(body, envelope) == "retry"
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.credit_out = True
            return False
        return True


class Toolkit:
    """Which tool shows what: pages, attributes, side-by-side views."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.shows_attrs: dict[str, bool] = {}
        self.side_good_n = 0
        self.side_bad_n = 0
        self.field_sets: dict[str, set[str]] = {}
        self.seen_fields: dict[str, tuple[int, set[str]]] = {}

    def page_tools(self) -> set[str]:
        page_tool = self.owner.brief.roles.get("offer") or self.owner.brief.roles.get("open")
        return {t_ for t_ in (page_tool, self.owner.brief.roles.get("compare")) if t_}

    def attr_tool(self) -> str | None:
        listed_labels = next((t_ for t_, passed in self.shows_attrs.items() if passed and self.id_param_of(t_)), None)
        if listed_labels:
            return listed_labels
        tool = self.owner.brief.roles.get("open")
        return tool if self.shows_attrs.get(tool or "") is not False and self.id_param_of(tool) else None

    def side_shows(self) -> bool | None:
        return self.shows_attrs.get(self.owner.brief.roles.get("compare") or "")

    def side_bare(self) -> bool:
        view = self.side_shows()
        return view is False or (view is None and (self.shows_attrs.get(self.attr_tool() or "") is True or bool(self.owner.brief.roles.get("offer"))))

    def id_param_of(self, tool: str | None) -> str | None:
        schema = self.owner.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        chunks = [k_ for k_, v_ in props.items() if isinstance(v_, dict) and "string" in _schema_types(v_) and "enum" not in v_]
        id_param = next((k_ for k_ in schema.get("required") or [] if k_ in chunks), None) or (chunks[0] if len(chunks) == 1 else None)
        if not tool or not id_param or any(k_ != id_param for k_ in schema.get("required") or []):
            return None
        return id_param

    def side_param(self) -> str | None:
        schema = self.owner.schemas.get(self.owner.brief.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        id_param = next((k_ for k_, v_ in props.items() if isinstance(v_, dict) and "array" in _schema_types(v_)), None)
        if not id_param or any(k_ != id_param for k_ in schema.get("required") or []):
            return None
        return id_param

    def side_cap(self) -> int:
        param_specs = ((self.owner.schemas.get(self.owner.brief.roles.get("compare") or "") or {}).get("properties") or {}).get(self.side_param() or "") or {}
        declared = param_specs.get("maxItems") if isinstance(param_specs.get("maxItems"), int) and param_specs["maxItems"] > 0 else None
        if self.side_bad_n:
            learned = max(2, min(self.side_bad_n - 1, max(self.side_good_n, self.side_bad_n // 2)))
        else:
            learned = declared or max(4, self.side_good_n)
        return min(declared, learned) if declared else learned

    def safe_side_cap(self) -> int:
        """How many listings Kyo's own side-by-side reads send: no more than the store has already been seen to show."""
        return self.side_cap() if self.side_bad_n else min(self.side_cap(), max(self.side_good_n, 4))

    def offer_tool(self) -> str | None:
        return self.owner.brief.roles.get("offer") or self.owner.brief.roles.get("open")


class Brief:
    """The request read once, and the ledger and status lines built from it."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.parsed: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.wants: list[Want] = []
        self.budget: Any = None
        self.max_messages = 0
        self.duty_items: list[dict[str, Any]] = []
        self.reserve_messages = 0
        self.lookups: list[tuple[int, str]] = []
        self.lookup_values: dict[str, str] = {}

    def parse_task(self) -> None:
        prompt = BRIEF_PROMPT + "\nTOOLS:\n" + "\n".join(self._tools_text()) + "\n\nREQUEST:\n" + self.owner.query
        if self.owner.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dumps(_clipped(self.owner.view.get("opening"), 1500))
        brief_data = self.owner.relay.ask_json(prompt, [LIGHT_MODEL], 4000) or {}
        if not self.owner.relay.no_credit and (not _listify(brief_data.get("items")) or not isinstance(brief_data.get("roles"), dict) or not brief_data["roles"].get("buy")):
            brief_data = self.owner.relay.ask_json(prompt, [HEAVY_MODEL, LIGHT_MODEL], 2500) or brief_data
        self.parsed = brief_data
        roles = brief_data.get("roles") if isinstance(brief_data.get("roles"), dict) else {}
        self.roles = {k_: v_ for k_, v_ in roles.items() if isinstance(v_, str) and v_ in self.owner.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief_data.get(key), dict):
                brief_data[key] = {}
        self._learn_modes(brief_data)
        self._parse_items(brief_data)
        self._parse_lookups(brief_data)
        self.split_lookups()
        self._parse_limits(brief_data)
        self._parse_duties(brief_data)
        self._closest_by_default(brief_data)

    def _closest_by_default(self, brief_data: dict[str, Any]) -> None:
        """A one-item request silent on what to do when nothing fits still gets the closest listing ordered: a fit that
        was missed or misread is far likelier than a store holding nothing of the kind, and an order keeps a chance."""
        if len(self.wants) != 1 or brief_data.get("fallback_buy") is True or brief_data.get("fallback") or brief_data.get("quiet_decline") is True:
            return
        brief_data["fallback_buy"] = True
        brief_data["fallback"] = CLOSEST_DEFAULT

    def split_lookups(self) -> None:
        for item_ix, want in enumerate(self.wants, 1):
            pointer_musts = [need for i_, need in self.lookups if i_ == item_ix and need in want.musts]
            other_musts = [n_ for n_ in want.musts if n_ not in pointer_musts]
            if not pointer_musts or not other_musts:
                continue
            prompt = (LOOKUP_SPLITTER + "\n\nREQUEST:\n" + self.owner.query + f"\n\nPRODUCT: {want.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointer_musts) + "\nOTHER REQUIREMENTS:\n" + "\n".join(other_musts))
            reply_obj = self.owner.relay.ask_json(prompt, [HEAVY_MODEL, LIGHT_MODEL], 500) or {}
            moved_keys = [n_ for n_ in other_musts if n_ in [str(x_).strip() for x_ in _listify(reply_obj.get("describe_other"))]]
            if not moved_keys:
                continue
            want.musts = [n_ for n_ in want.musts if n_ not in moved_keys]
            joined_ids = pointer_musts[0] + " (that listing: " + "; ".join(moved_keys) + ")"
            want.musts = [joined_ids if n_ == pointer_musts[0] else n_ for n_ in want.musts]
            self.lookups = [(i_, joined_ids if (i_, need) == (item_ix, pointer_musts[0]) else need) for i_, need in self.lookups]
            self.owner.relay.trace.append({"kind": "unmix", "item": item_ix, "moved": moved_keys})

    def rules_text(self) -> str:
        rule_lines = [self._want_line(item_ix, want) for item_ix, want in enumerate(self.wants, 1)]
        rule_lines += self._rule_lines()
        attr_tool = self.owner.kit.attr_tool()
        if attr_tool and (self.owner.kit.shows_attrs.get(attr_tool) is True or self.roles.get("offer")):
            attrless = sorted(t_ for t_ in self.owner.kit.page_tools() | {self.roles.get("offer") or ""} if t_ and self.owner.kit.shows_attrs.get(t_) is False)
            more_pages = sorted(t_ for t_ in self.owner.kit.page_tools() if t_ != attr_tool)
            rule_lines.append(f"listing attributes come from {attr_tool}" + (f"; {', '.join(attrless)} show none" if attrless else "")
                         + (f"; a listing's page (what a chain of pages or a page check refers to) comes from {' or '.join(more_pages)}" if more_pages else ""))
        return "\n".join("- " + s_ for s_ in rule_lines)

    def status(self) -> str:
        remaining = self.owner.relay.left()
        text = f"[Kyo] step {self.owner.steps_taken} of {self.owner.max_steps} used, {remaining} left." + self._step_alert(remaining) + self._status_tail()
        return text + "\n[ledger]\n" + self.rules_text()

    def active_needs(self) -> list[tuple[int, str]]:
        result_rows = []
        for item_ix, want in enumerate(self.wants, 1):
            for need in want.musts + want.heard_needs:
                if need not in want.dropped:
                    result_rows.append((item_ix, need))
        return result_rows

    def due_items(self, stage: str) -> list[dict[str, Any]]:
        """The steps the gate carries out at this stage: its own, and at the order any step before the choice that needs the shopper."""
        return [s_ for s_ in self.duty_items if s_["when"] == stage or (stage == "order" and s_["when"] == "choose" and s_["messages"])]

    def duties_at(self, stage: str) -> bool:
        return bool(self.due_items(stage)) and (stage, self.owner.relay.cart_key() + f"#{self.owner.epoch}") not in self.owner.clerk.gates_passed

    def _tools_text(self, skip_tools: set[Any] | None = None) -> list[str]:
        return [f"- {t_['function']['name']}: {t_['function'].get('description', '')} | parameters: {_dumps(t_['function'].get('parameters') or {})}"
                for t_ in self.owner.tool_defs if not skip_tools or t_["function"]["name"] not in skip_tools]

    def _learn_modes(self, brief_data: dict[str, Any]) -> None:
        """One tool both checks out and ends without a purchase: the arguments that choose each mode, asked once."""
        checkout_tool, decline_tool = self.roles.get("buy"), self.roles.get("refuse")
        if not checkout_tool or checkout_tool != decline_tool or (brief_data["buy_args"] and brief_data["refuse_args"] and brief_data["buy_args"] != brief_data["refuse_args"]):
            return
        ask_tool = (f"The store tool {checkout_tool} serves two purposes, checking out and ending without a purchase, and its arguments select which one. Its schema: "
               + _dumps(self.owner.schemas.get(checkout_tool)) + "\nReply with ONE JSON object and nothing more: {\"buy_args\": the fixed arguments that make it check out, \"refuse_args\": the fixed arguments that make it end without a purchase} (only the arguments that select the mode).")
        mode_sets = self.owner.relay.ask_json(ask_tool, [LIGHT_MODEL, HEAVY_MODEL], 600) or {}
        if isinstance(mode_sets.get("buy_args"), dict) and isinstance(mode_sets.get("refuse_args"), dict) and mode_sets["buy_args"] != mode_sets["refuse_args"]:
            brief_data["buy_args"], brief_data["refuse_args"] = mode_sets["buy_args"], mode_sets["refuse_args"]

    def _parse_items(self, brief_data: dict[str, Any]) -> None:
        for entry in _listify(brief_data.get("items")):
            if not isinstance(entry, dict):
                continue
            units_said = _as_int(entry.get("quantity"))
            firm_needs = [str(n_) for n_ in _listify(entry.get("musts")) if n_]
            self.wants.append(Want(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), units_said if units_said and units_said > 0 else 1, firm_needs))
        if not self.wants:
            self.wants = [Want("the requested product", "", 1, [])]

    def _parse_lookups(self, brief_data: dict[str, Any]) -> None:
        for lookup_text in _listify(brief_data.get("lookups")):
            text = str(lookup_text).strip()
            if not any(text in want.musts for want in self.wants):
                key = _plain_words(text.split(":")[0])
                text = next((n_ for want in self.wants for n_ in want.musts if _plain_words(n_.split(":")[0]) == key), text)
            item_ix = next((i_ for i_, want in enumerate(self.wants, 1) if text in want.musts), 1)
            if text and (item_ix, text) not in self.lookups:
                self.lookups.append((item_ix, text))

    def _parse_limits(self, brief_data: dict[str, Any]) -> None:
        budget_val = brief_data.get("budget")
        if isinstance(budget_val, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget_val):
            budget_val = float(budget_val)
        if isinstance(budget_val, (int, float)) and not isinstance(budget_val, bool) and budget_val > 0:
            self.budget = budget_val
        max_len = _as_int(brief_data.get("messages"))
        if max_len is not None and max_len >= 0:
            self.max_messages = max_len
        elif self.roles.get("ask"):
            self.max_messages = 3
        step_items = _as_int(brief_data.get("steps"))
        if step_items is not None and 10 <= step_items < self.owner.max_steps:
            self.owner.max_steps = step_items

    def _parse_duties(self, brief_data: dict[str, Any]) -> None:
        for entry in _listify(brief_data.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when_word = str(entry.get("when") or "").strip().lower()
                self.duty_items.append({"step": str(entry["step"]).strip(), "when": when_word if when_word in ("choose", "order", "end") else "order",
                                     "messages": max(0, _as_int(entry.get("messages")) or 0)})
        self._choose_steps_to_interview(brief_data)
        if self.roles.get("ask"):
            needs_talk = [s_["messages"] for s_ in self.duty_items if s_["messages"]]
            self.reserve_messages = min(max(needs_talk), self.max_messages) if needs_talk else 0

    def _choose_steps_to_interview(self, brief_data: dict[str, Any]) -> None:
        """A step before the choice that needs the shopper is a question the interview asks, not a message held back for
        the checkout check: it joins the things to check first, and no message is reserved for it."""
        talk_first = [s_ for s_ in self.duty_items if s_["when"] == "choose" and s_["messages"]]
        if not talk_first:
            return
        check_first = [str(x_) for x_ in _listify(brief_data.get("ask_first")) if str(x_).strip()]
        brief_data["ask_first"] = check_first + [s_["step"] for s_ in talk_first if s_["step"] not in check_first]
        self.duty_items = [s_ for s_ in self.duty_items if s_ not in talk_first]

    def _want_line(self, item_ix: int, want: Want) -> str:
        need_list = [n_ + (f" (the page it points at gives: {self.lookup_values[n_]})" if n_ in self.lookup_values else "") for n_ in want.musts + want.heard_needs if n_ not in want.dropped]
        opening_asks = f"item {item_ix}: {want.name}" if len(self.wants) > 1 else f"item: {want.name}"
        if want.shelf:
            opening_asks += f" (store label: {want.shelf})"
        opening_asks += f", quantity {want.qty}"
        text = opening_asks + "; requirements: " + ("; ".join(need_list) if need_list else 'nothing beyond the product kind')
        if want.dont_care:
            text += '; the shopper does not mind about: ' + ", ".join(want.dont_care)
        return text

    def _rule_lines(self) -> list[str]:
        brief_data = self.parsed
        rule_lines = self._budget_line() + [self._pick_line()]
        if brief_data.get("in_stock_only", True) is not False:
            rule_lines.append('only listings in stock')
        if self.owner.till.sets_learned:
            rule_lines.append("the cart call sets a line's count to the number sent (it does not add): send each line's full wanted count; 0 takes a line off")
        rule_lines += [f"wish (tie-breaker only): {wish_text}" for wish_text in _listify(brief_data.get("wishes"))]
        rule_lines += [f"rule from the request: {rule_text}" for rule_text in _listify(brief_data.get("rules"))]
        if brief_data.get("quiet_decline") is True:
            rule_lines.append('when nothing fits: end without naming or suggesting any listing')
        moment_words = {"choose": "before the choice", "order": 'at the order (Kyo handles it when you check out)', "end": 'at an ending without buying (Kyo handles it then)'}
        return rule_lines + [f"step the request asks for, {moment_words[entry['when']]}: {entry['step']}" for entry in self.duty_items]

    def _step_alert(self, remaining: int) -> str:
        till = self.owner.till
        if remaining <= self.owner.relay.kept_steps() + 1:
            return ' LAST STEPS: check out now if the cart holds a line for every item; end without buying only if your searches showed that some item has no fitting listing (running short of steps shows nothing like that).'
        if remaining <= 6 or (remaining <= len(self.wants) + 5 and till.checkout_count == 0):
            return ' Few steps are left: get the chosen lines into the cart and check out now.'
        if till.checkout_count == 0 and remaining <= 12:
            return ' Steps are running low: settle your picks, fill the cart and check out within the next few steps.'
        if till.checkout_count == 0 and not till.cart_touched and self.owner.steps_taken >= self.owner.max_steps // 2:
            return " Half the steps are used and the cart is still empty: add each item's fitting listing now (the store may still sell one out)."
        return ""

    def _status_tail(self) -> str:
        """Pointers still to settle, the review's listings and the changes noticed so far."""
        added = ""
        live_needs = {need for _, need in self.active_needs()}
        lookup_texts = [need for _, need in self.lookups if need in live_needs and need not in self.lookup_values]
        if lookup_texts and self.owner.till.checkout_count == 0:
            added += '\n[settle these first: open the listing each one points to, then every listing its page leads to; the last page holds the value] ' + " | ".join(lookup_texts[:3])
        if self.owner.reviewer.suggested:
            added += '\n[listings the review suggests] ' + ", ".join(self.owner.reviewer.suggested)
        if self.owner.bulletin.updates:
            added += '\n[changes up to now] ' + " | ".join(self.owner.bulletin.updates[-4:])
        return added

    def as_written(self, figure: Any, text: str) -> Any:
        """A budget a reader took from the shopper's words, kept only as those words write it: a converted figure gives way to the number stated in the request's own unit."""
        if not isinstance(figure, (int, float)) or isinstance(figure, bool):
            return figure
        stated = [(float(n_.replace(",", "")), mark_) for n_, mark_ in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(\S?)", text)]
        if any(abs(n_ - figure) < 1e-9 for n_, _ in stated):
            return figure
        if len({n_ for n_, _ in stated}) == 1:
            return stated[0][0]
        own = re.search(re.escape(f"{self.budget:g}") + r"\s*(\S?)", self.owner.query.replace(",", "")) if isinstance(self.budget, (int, float)) else None
        marked = {n_ for n_, mark_ in stated if own and own.group(1) and not own.group(1).isalnum() and mark_ == own.group(1)}
        if len(marked) == 1:
            return marked.pop()
        self.owner.relay.trace.append({"kind": "budget_unread", "got": figure})
        return None

    def _budget_line(self) -> list[str]:
        if self.budget is None:
            return []
        scope_words = "for each item" if self.parsed.get("budget_scope") == "each" else "in total"
        return [f"budget: at most {self.budget} {self.parsed.get('budget_unit') or ''} {scope_words}".replace("  ", " ")]

    def _pick_line(self) -> str:
        pick = self.parsed.get("pick")
        if pick == "cheapest":
            return 'pick: the lowest-priced fitting listing'
        if pick in ("most", "least"):
            return f"pick: the {pick} {self.parsed.get('pick_measure') or ''} among fitting listings".replace("  ", " ")
        return 'pick: any listing meeting everything'


class Ledger:
    """Listing records gathered from every result."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.rows: dict[str, dict[str, Any]] = {}
        self.opened: dict[str, dict[str, Any]] = {}
        self.seen: list[str] = []
        self.id_key: str | None = None
        self.pages_read: set[str] = set()
        self.offer_step: dict[str, int] = {}
        self.sold_step: dict[str, int] = {}
        self.blocked: set[str] = set()
        self.id_args: set[tuple[str, str]] = set()

    def record_of(self, node: Any, in_rows_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_shaped(node.get(self.id_key), True):
            return node[self.id_key].strip()
        has_nested = any(isinstance(v_, dict) for v_ in node.values())
        if not (in_rows_list or depth == 0 or has_nested):
            return None
        found = _guess_id_key([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [i_ for i_ in node if isinstance(i_, dict) and len(i_) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(_id_shaped(r_.get(self.id_key), True) for r_ in rows):
            return True
        found = _guess_id_key(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observed: Any) -> None:
        self._absorb_node(observed, 0)

    def label_enum(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for enum_value in schema.get("enum") or []:
                if isinstance(enum_value, str):
                    found.add(enum_value.strip().lower())
            for datum in schema.values():
                if isinstance(datum, dict):
                    walk(datum, depth + 1)

        for schema in self.owner.schemas.values():
            walk(schema, 0)
        return found

    def label_key(self) -> str | None:
        unit_values: dict[str, set[str]] = {}
        presence: dict[str, int] = {}
        for row_rec in self.rows.values():
            for key, datum in row_rec.items():
                if isinstance(datum, str) and 0 < len(datum) <= 40 and not re.search(r"\d{3}", datum):
                    unit_values.setdefault(key, set()).add(datum)
                    presence[key] = presence.get(key, 0) + 1
        row_total = max(1, len(self.rows))
        listed_labels = self.label_enum()
        best = None
        for key, datums in unit_values.items():
            if presence[key] < 0.8 * row_total or len(datums) > 40:
                continue
            listed_share = sum(1 for v_ in datums if v_.strip().lower() in listed_labels) / len(datums)
            mean_len = sum(len(v_) for v_ in datums) / len(datums)
            rank_score = (-listed_share, len(datums), -presence[key], -mean_len)
            if best is None or rank_score < best[0]:
                best = (rank_score, key)
        return best[1] if best else None

    def keys_text(self) -> list[tuple[str, str, int, list[str], int]]:
        groups = self._keys_per_label(self.label_key())
        wanted_sets = self._label_needs()
        matched_labels = {prop for prop in groups if self._label_fits(prop, wanted_sets)}
        if matched_labels:
            groups = {prop: group for prop, group in groups.items() if prop in matched_labels}
        key_rows = []
        for prop, group in sorted(groups.items(), key=lambda pair_: -pair_[1]["n"]):
            for tool_name, slot in self._ask_keys(group)[:14]:
                key_rows.append((prop, tool_name, slot["n"], [v_[:40] for v_ in slot["values"][:4]], len({str(v_).strip().lower() for v_ in slot["per"].values()})))
        return key_rows

    def label_groups(self) -> list[list[str]]:
        cat_key = self.label_key()
        groups: dict[str, list[str]] = {}
        for lid in self.seen:
            prop = str(self.rows[lid].get(cat_key) or "") if cat_key else ""
            groups.setdefault(prop, []).append(lid)
        wanted_sets = self._label_needs()
        matched_labels = [lids for prop, lids in groups.items() if self._label_fits(prop, wanted_sets)]
        pool_ids = matched_labels or list(groups.values())
        return sorted(pool_ids, key=lambda lids: -len(lids))[: max(1, len(self.owner.brief.wants))]

    def price_of(self, lid: str) -> float | None:
        datum = (self.rows.get(lid) or {}).get(self.owner.scout.price_key) if self.owner.scout.price_key else None
        if isinstance(datum, dict):
            datum = next((v_ for v_ in datum.values() if isinstance(v_, (int, float)) and not isinstance(v_, bool)), None)
        if isinstance(datum, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", datum):
            datum = float(datum)
        return float(datum) if isinstance(datum, (int, float)) and not isinstance(datum, bool) else None

    def buyable(self, lid: str) -> bool:
        if self.gone(lid):
            return False
        datum = (self.rows.get(lid) or {}).get(self.owner.scout.stock_key) if self.owner.scout.stock_key else None
        return not (datum is False or (isinstance(datum, (int, float)) and not isinstance(datum, bool) and datum <= 0))

    def rows_in(self, observed: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observed, dict):
            lid = self.record_of(observed)
            if lid:
                return [(lid, observed)]
            found: list[tuple[str, dict[str, Any]]] = []
            for datum in observed.values():
                if isinstance(datum, list):
                    found += [(lid_, r_) for r_ in datum if isinstance(r_, dict) for lid_ in [self.record_of(r_, True)] if lid_]
                elif isinstance(datum, dict):
                    found += [(lid_, r_) for v_ in datum.values() if isinstance(v_, list) for r_ in v_ if isinstance(r_, dict) for lid_ in [self.record_of(r_, True)] if lid_]
            return found
        if isinstance(observed, list):
            return [(lid_, r_) for r_ in observed if isinstance(r_, dict) for lid_ in [self.record_of(r_, True)] if lid_]
        return []

    def note_view(self, tool_name: str, observed: Any) -> None:
        for lid, record in self.rows_in(observed):
            field_set = set(record)
            self.owner.kit.field_sets.setdefault(tool_name, set()).update(field_set)
            view_version, units_held = self.owner.kit.seen_fields.get(lid, (-1, set()))
            self.owner.kit.seen_fields[lid] = (self.owner.epoch, (units_held if view_version == self.owner.epoch else set()) | field_set)

    def all_seen(self, tool_name: str, lid: str) -> bool:
        view_version, units_held = self.owner.kit.seen_fields.get(lid, (-1, set()))
        field_set = self.owner.kit.field_sets.get(tool_name)
        return bool(field_set) and view_version == self.owner.epoch and field_set <= units_held and lid not in self.sold_step

    def gone(self, lid: str) -> bool:
        return lid in self.blocked or lid in self.sold_step

    def _label_needs(self) -> list[tuple[str, str, set[str]]]:
        return [(_plain_words(want.shelf or ""), _plain_words(want.name), {word_ for word_ in _token_set(want.shelf or "") | _token_set(want.name) if len(word_) >= 4}) for want in self.owner.brief.wants]

    def _label_fits(self, prop: str, wanted_sets: list[tuple[str, str, set[str]]]) -> bool:
        return any(_plain_words(prop) in (label_text, tool_name) or {word_ for word_ in _token_set(prop) if len(word_) >= 4} & long_words for label_text, tool_name, long_words in wanted_sets)

    def _keys_per_label(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for lid in self.seen:
            record = self.opened.get(lid)
            if not record:
                continue
            prop = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(prop, {"n": 0, "keys": {}})
            group["n"] += 1
            self._tally_keys(group["keys"], lid, _attrs_of(record) or {})
        return groups

    @staticmethod
    def _ask_keys(group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """A category's keys worth asking about: no long texts, variant pickers, per-listing names or twins."""
        kept_keys: list[tuple[str, dict[str, Any]]] = []
        for tool_name, slot in sorted(group["keys"].items(), key=lambda pair_: -pair_[1]["n"]):
            datums = list(slot["per"].values())
            if not datums:
                continue
            mean_len = sum(len(v_) for v_ in datums) / len(datums)
            if mean_len > 40 or sum(1 for v_ in datums if " / " in v_) * 2 >= len(datums):
                continue
            if len(datums) >= 4 and len(set(datums)) == len(datums) and mean_len > 12:
                continue
            if any(tool_name.startswith(other_tool + " ") and slot["per"] == param_specs["per"] for other_tool, param_specs in kept_keys):
                continue
            kept_keys.append((tool_name, slot))
        return kept_keys

    def _absorb_node(self, node: Any, depth: int, in_rows_list: bool = False) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            lid = self.record_of(node, in_rows_list, depth)
            if lid:
                self._merge_record(lid, node)
                return
            for datum in node.values():
                self._absorb_node(datum, depth + 1)
        elif isinstance(node, list):
            rows = self.learn_rows(node)
            for datum in node:
                self._absorb_node(datum, depth + 1, rows)

    def _merge_record(self, lid: str, node: dict[str, Any]) -> None:
        """A listing's record merged in; a flag of its own turning from true to false marks it unbuyable (and back)."""
        if lid not in self.rows:
            self.seen.append(lid)
            self.rows[lid] = {}
        old_row = self.rows[lid]
        if any(old_row.get(k_) is True and v_ is False for k_, v_ in node.items()):
            self.blocked.add(lid)
        elif any(old_row.get(k_) is False and v_ is True for k_, v_ in node.items()):
            self.blocked.discard(lid)
        self.rows[lid].update(node)
        if _attrs_of(node):
            self.opened[lid] = dict(self.rows[lid])
        elif lid in self.opened:
            self.opened[lid].update(node)

    @staticmethod
    def _tally_keys(key_slots: dict[str, Any], lid: str, attr_table: dict[str, Any]) -> None:
        """One listing's attribute names counted once each, with its value kept per listing and in the value list."""
        done: set[str] = set()
        for key, datum in attr_table.items():
            tool_name = _plain_words(key)
            if not tool_name or tool_name in done:
                continue
            done.add(tool_name)
            slot = key_slots.setdefault(tool_name, {"n": 0, "values": [], "per": {}})
            slot["n"] += 1
            text = str(datum).strip().lower()
            slot["per"][lid] = text
            if text not in slot["values"]:
                slot["values"].append(text)


class Scout:
    """Result pages, paging and reading more listings."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.auto_page_count = 0
        self.auto_pages: list[Any] = []
        self.cheaper_marks: set[str] = set()
        self.swept: set[tuple[str, Any, int]] = set()
        self.widened = False

    def shape_of(self, tool_name: str, observed: Any) -> dict[str, Any] | None:
        if tool_name in self.shapes:
            return self.shapes[tool_name]
        if not isinstance(observed, dict):
            return None
        list_keys = [k_ for k_, v_ in observed.items() if isinstance(v_, list) and len(v_) >= 2 and all(isinstance(r_, dict) for r_ in v_)]
        if not list_keys:
            return None
        self.shapes[tool_name] = None
        reply_obj = self._shape_prompt(tool_name, observed, list_keys)
        row_key = reply_obj.get("rows") if reply_obj.get("rows") in list_keys else None
        if not row_key:
            return None
        pieces = self._valid_shape(tool_name, observed, row_key, reply_obj)
        self.shapes[tool_name] = pieces
        self.price_key = self.price_key or pieces["price"]
        self.stock_key = self.stock_key or pieces["stock"]
        self.owner.relay.trace.append({"kind": "shape", "tool": tool_name, "shape": pieces})
        return pieces

    def next_page(self, tool_name: str, observed: Any) -> tuple[int, int, str, Any] | None:
        pieces = self.shape_of(tool_name, observed)
        if not pieces or not pieces["count"] or not pieces["next"] or not pieces["next_arg"] or not isinstance(observed, dict):
            return None
        rows, qty, datum = observed.get(pieces["rows"]), observed.get(pieces["count"]), observed.get(pieces["next"])
        if not isinstance(rows, list) or not rows or isinstance(qty, bool) or not isinstance(qty, int):
            return None
        if datum is not None and (isinstance(datum, bool) or not isinstance(datum, (str, int))):
            return None
        return qty, len(rows), pieces["next_arg"], datum

    def price_sorted(self, observed: Any) -> bool:
        price_list = [self.owner.ledger.price_of(lid_) for lid_, _ in self.owner.ledger.rows_in(observed)]
        return len(price_list) >= 3 and None not in price_list and all(lo_ <= hi_ for lo_, hi_ in zip(price_list, price_list[1:]))

    def more_pages(self, tool_name: str, call_args: dict[str, Any], observed: Any) -> str:
        self.auto_pages = []
        pick = self.owner.brief.parsed.get("pick")
        plan = self.next_page(tool_name, observed) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        qty, shown_n, page_param, datum = plan
        given_value = call_args.get(page_param)
        paging_on = given_value not in (None, 0, 1) if isinstance(given_value, int) and not isinstance(given_value, bool) else bool(str(given_value or "").strip())
        if datum is None or paging_on or qty <= shown_n:
            return ""
        if pick == "cheapest" and self.price_sorted(observed):
            return ""
        more_pages = -(-(qty - shown_n) // shown_n)
        if more_pages > (6 if pick == "cheapest" else 2) or self.auto_page_count + more_pages > AUTO_PAGE_LIMIT or self.owner.relay.left() - self.owner.relay.kept_steps() - more_pages < 12:
            return (f"\n[Kyo] This search matched {qty} listings and showed {shown_n}, in no price order. The pick compares every fitting listing: "
                    f"search with narrower words (the requirement's value) or read the next page with {page_param}={_dumps(datum)}.")
        chunks = self._follow_pages(tool_name, call_args, page_param, datum, more_pages)
        self.owner.relay.trace.append({"kind": "pages", "tool": tool_name, "read": len(chunks)})
        return ('\n[Kyo read the other pages of this search itself, one step per page]\n' + "\n".join(chunks)) if chunks else ""

    def cheapest_rows(self, obs_list: list[Any]) -> str:
        if self.owner.brief.parsed.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {lid_ for view in obs_list for lid_, _ in self.owner.ledger.rows_in(view)}
        rows = [(self.owner.ledger.price_of(lid_), lid_) for lid_ in found if self.owner.ledger.buyable(lid_) and self.owner.ledger.price_of(lid_) is not None]
        if len(rows) < 2:
            return ""
        return '\n[Kyo] buyable rows here, cheapest first: ' + ", ".join(f"{lid_} ({s_:g})" for s_, lid_ in sorted(rows)[:6])

    def widen_search(self) -> str:
        if self.widened or self.owner.brief.parsed.get("pick") != "cheapest" or self.owner.relay.left() < 14 + 2 * len(self.owner.brief.wants):
            return ""
        self.widened = True
        lines = [f"{lid_} at {self.owner.ledger.price_of(lid_):g}" for lid_ in self.owner.reviewer.cart_ids() if self.owner.ledger.price_of(lid_) is not None]
        return (f"Not run yet (no step used). {self.owner.relay.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often found only in the attributes) brings up few of the fitting listings, and rows arrive in no price order."
                " Before checking out: when a search tool filters by category or by a price limit, read the item's whole category (its label exactly as the store lists it) capped just under the cart line's price, every page of it;"
                " also run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and consider only buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + '; read the attributes of the cheapest of those rows, swap in any that fits, then check out.')

    def open_attrs(self, lids: list[str], reserve: int, by_steps: bool = False) -> list[str]:
        """Reads the pages of listings not opened yet; for the order's own lines (by_steps) only the step budget limits it, not the counted clock."""
        tool = self.owner.kit.attr_tool()
        id_param = self.owner.kit.id_param_of(tool)
        if not tool or not id_param:
            return []
        ledger = self.owner.ledger
        chunks = []
        for lid in [lid_ for lid_ in lids if lid_ in ledger.rows and lid_ not in ledger.opened and not ledger.gone(lid_)]:
            room = self.owner.relay.steps_left() if by_steps else self.owner.relay.left()
            if room <= reserve or self.owner.done or self.owner.kit.shows_attrs.get(tool) is False:
                break
            shown_text = self._open_one(tool, id_param, lid)
            if shown_text is None:
                break
            chunks.append(f"{lid}: " + shown_text)
        if chunks:
            self.owner.relay.trace.append({"kind": "fill_facts", "n": len(chunks)})
        return chunks

    def _arg_spec(self, key: str) -> tuple[str, str] | None:
        spec = self.owner.brief.parsed.get(key)
        if not isinstance(spec, dict):
            return None
        tool, arg = str(spec.get("tool") or ""), str(spec.get("arg") or "")
        props = (self.owner.schemas.get(tool) or {}).get("properties") or {}
        return (tool, arg) if arg in props else None

    def cap_sweep(self) -> list[Any]:
        """Under a cheapest pick, list each cart line's category below that line's price once per cart (one step per line),
        so the cheaper rows reach the ledger and the cheaper reads that follow can open them."""
        seen_views: list[Any] = []
        if self.owner.brief.parsed.get("pick") != "cheapest":
            return seen_views
        cap = self._arg_spec("price_cap")
        cat = self._arg_spec("category_arg")
        cat_key = self.owner.ledger.label_key()
        call_mark = "sweep" + self.owner.relay.cart_key() + f"#{self.owner.epoch}"
        if not cap or call_mark in self.cheaper_marks:
            return seen_views
        self.cheaper_marks.add(call_mark)
        ledger, relay = self.owner.ledger, self.owner.relay
        for lid in self.owner.reviewer.cart_ids():
            price = ledger.price_of(lid)
            if (lid, price, self.owner.epoch) in self.swept:
                continue
            if price is None or price <= 0 or relay.left() < relay.kept_steps() + 6 or self.owner.done:
                return seen_views
            self.swept.add((lid, price, self.owner.epoch))
            call_args: dict[str, Any] = {cap[1]: price - 1 if float(price).is_integer() else round(price * 0.999, 2)}
            label = str((ledger.rows.get(lid) or {}).get(cat_key) or "") if cat_key else ""
            if cat and cat[0] == cap[0] and label:
                call_args[cat[1]] = label
            faults: list[str] = []
            call_args = _fit_schema(call_args, self.owner.schemas.get(cap[0]), "args", faults)
            if faults:
                return seen_views
            env_reply = relay.env_call(cap[0], call_args)
            if env_reply is None:
                return seen_views
            self.absorb_result(cap[0], env_reply.get("observation"))
            seen_views.append(env_reply.get("observation"))
            relay.trace.append({"kind": "cap_sweep", "line": lid, "args": call_args})
        return seen_views

    def cheaper_reads(self) -> list[str]:
        if self.owner.brief.parsed.get("pick") != "cheapest":
            return []
        call_mark = self.owner.relay.cart_key() + f"#{self.owner.epoch}"
        cat_key = self.owner.ledger.label_key()
        if call_mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(call_mark)
        cart_ids = self.owner.reviewer.cart_ids()
        chosen: list[str] = []
        for cart_id in cart_ids:
            chosen += [lid_ for lid_ in self._cheaper_than(cart_id, cat_key, cart_ids)[:2] if lid_ not in chosen]
        return self.open_attrs(chosen[:CHEAPER_READS], self.owner.relay.kept_steps() + 4 + SWAP_STEPS)

    def absorb_result(self, tool_name: str, observed: Any) -> str:
        self.owner.ledger.absorb(observed)
        lines = self.owner.bulletin.new_notes(observed)
        if not lines:
            return ""
        notice_meaning = self.owner.bulletin.parse_note(lines)
        if notice_meaning:
            self.owner.bulletin.queued_notes.append(f"(seen on {tool_name}) {notice_meaning}")
        return notice_meaning

    def read_side(self) -> None:
        tool = self.owner.brief.roles.get("compare")
        schema = self.owner.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        id_param = next((k_ for k_, v_ in props.items() if isinstance(v_, dict) and "array" in _schema_types(v_)), None)
        if not tool or not id_param or any(k_ != id_param for k_ in schema.get("required") or []) or self.owner.kit.side_bare():
            self.read_one_by_one()
            return
        kit, ledger = self.owner.kit, self.owner.ledger
        for _ in range(4):
            if self.owner.relay.left() <= 12 or len(ledger.opened) >= 12 or self.owner.done:
                return
            chosen = self._side_choice(kit.safe_side_cap())
            if len(chosen) < 2:
                return
            faults: list[str] = []
            call_args = _fit_schema({id_param: chosen}, schema, "args", faults)
            if faults:
                return
            opened_before = len(ledger.opened)
            self._side_call(tool, schema, id_param, chosen, call_args)
            if len(ledger.opened) > opened_before:
                kit.side_good_n = max(kit.side_good_n, len(chosen))
            elif kit.side_bad_n and kit.side_bad_n <= len(chosen) <= 2:
                return
            else:
                kit.side_bad_n = min(kit.side_bad_n or 999, len(chosen))

    def read_one_by_one(self) -> None:
        tool = self.owner.kit.attr_tool()
        id_param = self.owner.kit.id_param_of(tool)
        if not tool or not id_param:
            return
        lane_lists: list[list[str]] = []
        for lids in self.owner.ledger.label_groups():
            opened_count = sum(1 for lid_ in lids if lid_ in self.owner.ledger.opened)
            pool_ids = [lid_ for lid_ in lids if lid_ not in self.owner.ledger.opened and self.owner.ledger.buyable(lid_)]
            if self.owner.brief.parsed.get("pick") == "cheapest":
                pool_ids.sort(key=lambda lid_: self.owner.ledger.price_of(lid_) if self.owner.ledger.price_of(lid_) is not None else float("inf"))
            lane_lists.append(pool_ids[: max(0, SAMPLE_PER_KIND - opened_count)])
        depth = max((len(lane_ids) for lane_ids in lane_lists), default=0)
        read_queue = [lane_ids[k_] for k_ in range(depth) for lane_ids in lane_lists if k_ < len(lane_ids)]
        # pages read by Kyo itself cost a step each but no model turn, so only the step budget limits this sample
        self.open_attrs(read_queue, max(12, 2 * self.owner.max_steps // 3), by_steps=True)

    def _follow_pages(self, tool_name: str, call_args: dict[str, Any], page_param: str, datum: Any, more_pages: int) -> list[str]:
        chunks = []
        for _ in range(more_pages):
            page_text, datum = self._next_page_read(tool_name, call_args, page_param, datum)
            if page_text is None:
                break
            chunks.append(page_text)
            if datum is None:
                break
        return chunks

    def _side_call(self, tool: str, schema: dict[str, Any], id_param: str, chosen: list[str], call_args: dict[str, Any]) -> None:
        ledger, kit = self.owner.ledger, self.owner.kit
        opened_before = len(ledger.opened)
        env_reply = self.owner.relay.env_call(tool, call_args)
        if env_reply is None:
            return
        view = env_reply.get("observation")
        self.absorb_result(tool, view)
        ledger.note_view(tool, view)
        rows = ledger.rows_in(view)
        if rows:
            kit.shows_attrs[tool] = kit.shows_attrs.get(tool, False) or any(_attrs_of(r_) for _, r_ in rows)
        self.owner.relay.trace.append({"kind": "widen", "ids": chosen, "opened": len(ledger.opened) - opened_before})
        if rows and kit.shows_attrs.get(tool) is False:
            self.read_one_by_one()
        elif len(ledger.opened) == opened_before and len(chosen) > 2 and self.owner.relay.left() > 12:
            kit.side_bad_n = min(kit.side_bad_n or 999, len(chosen))
            self._side_again(tool, schema, {id_param: chosen[:2]})
            if len(ledger.opened) > opened_before:
                kit.side_good_n = max(kit.side_good_n, 2)

    def _shape_prompt(self, tool_name: str, observed: dict[str, Any], list_keys: list[str]) -> dict[str, Any]:
        result_sample = {k_: (v_[:3] if k_ in list_keys else v_) for k_, v_ in observed.items()}
        prompt = (RESULT_SHAPE + "\n\nTOOL: " + tool_name + " | parameters: " + _dumps(self.owner.schemas.get(tool_name) or {})
                       + "\n\nRESULT (first rows only):\n" + _dumps(_clipped(result_sample, 120))[:4000])
        return self.owner.relay.ask_json(prompt, [LIGHT_MODEL, HEAVY_MODEL], 1500) or {}

    def _valid_shape(self, tool_name: str, observed: dict[str, Any], row_key: str, reply_obj: dict[str, Any]) -> dict[str, Any]:
        """The named parts kept only where the result (or the tool's parameters, or a row) really has that key."""
        first_row = observed[row_key][0]
        props = (self.owner.schemas.get(tool_name) or {}).get("properties") or {}

        def present(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        pieces = {"rows": row_key, "count": present(reply_obj.get("count"), observed), "next": present(reply_obj.get("next"), observed),
                 "next_arg": present(reply_obj.get("next_arg"), props), "price": present(reply_obj.get("price"), first_row),
                 "stock": present(reply_obj.get("stock"), first_row)}
        if pieces["count"] and (isinstance(observed[pieces["count"]], bool) or not isinstance(observed[pieces["count"]], int)):
            pieces["count"] = None
        return pieces

    def _open_one(self, tool: str, id_param: str, lid: str) -> str | None:
        faults: list[str] = []
        call_args = _fit_schema({id_param: lid}, self.owner.schemas.get(tool), "args", faults)
        if faults:
            return None
        env_reply = self.owner.relay.env_call(tool, call_args)
        if env_reply is None:
            return None
        view = env_reply.get("observation")
        self.absorb_result(tool, view)
        self.owner.ledger.note_view(tool, view)
        if isinstance(view, dict) and self.owner.ledger.record_of(view):
            views = self.owner.kit.shows_attrs
            views[tool] = views.get(tool, False) or bool(_attrs_of(view))
        return self.owner.relay.render(view)[:1500]

    def _side_choice(self, room_cap: int) -> list[str]:
        """Unopened listings shared evenly across the label groups, up to the tool's room."""
        ranked_ids = self.owner.ledger.label_groups()
        per_group = max(1, room_cap // max(1, len(ranked_ids)))
        chosen: list[str] = []
        for lids in ranked_ids:
            chosen += [lid_ for lid_ in lids if lid_ not in self.owner.ledger.opened][:per_group]
        return chosen[:room_cap]

    def _cheaper_than(self, cart_id: str, cat_key: str, cart_ids: list[str]) -> list[str]:
        """Unopened, buyable listings of the cart line's own kind seen at a lower price, cheapest first."""
        ledger = self.owner.ledger
        price = ledger.price_of(cart_id)
        kind_name = str((ledger.rows.get(cart_id) or {}).get(cat_key) or "").strip().lower()
        if price is None or not kind_name:
            return []
        pool_ids = [lid_ for lid_ in ledger.seen if lid_ not in cart_ids and lid_ not in ledger.opened and ledger.buyable(lid_)
                    and str((ledger.rows.get(lid_) or {}).get(cat_key) or "").strip().lower() == kind_name
                    and (ledger.price_of(lid_) if ledger.price_of(lid_) is not None else float("inf")) < price]
        pool_ids.sort(key=lambda lid_: ledger.price_of(lid_) or 0.0)
        return pool_ids

    def _next_page_read(self, tool_name: str, call_args: dict[str, Any], page_param: str, datum: Any) -> tuple[str | None, Any]:
        """One more page read: its text and the next page's value, or (None, None) when it could not be read."""
        faults: list[str] = []
        call_spec = _fit_schema({**call_args, page_param: datum}, self.owner.schemas.get(tool_name), "args", faults)
        if faults:
            return None, None
        env_reply = self.owner.relay.env_call(tool_name, call_spec)
        if env_reply is None or self.owner.done:
            return None, None
        self.auto_page_count += 1
        view = env_reply.get("observation")
        self.absorb_result(tool_name, view)
        self.auto_pages.append(view)
        page_text = self.owner.relay.render(view)
        self.owner.dispatch.memo[tool_name + _dumps(call_spec)] = (self.owner.epoch, self.owner.steps_taken, page_text)
        next_plan = self.next_page(tool_name, view)
        return f"{page_param}={_dumps(datum)}: {page_text}", (next_plan[3] if next_plan else None)

    def _side_again(self, tool: str, schema: dict[str, Any], call_args: dict[str, Any]) -> None:
        """A side-by-side read that opened nothing, tried once more with the first two listings only."""
        faults: list[str] = []
        call_args = _fit_schema(call_args, schema, "args", faults)
        if faults:
            return
        env_reply = self.owner.relay.env_call(tool, call_args)
        if env_reply is not None:
            self.absorb_result(tool, env_reply.get("observation"))


class Interviewer:
    """Questions to the shopper and their answers."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.sent_count = 0
        self.asked = False

    def should_ask(self) -> bool:
        if self.asked or not self.owner.brief.roles.get("ask") or self.owner.relay.messages_left() <= 0:
            return False
        if self.owner.relay.left() <= 8:
            return False
        opened_count = len(self.owner.ledger.opened)
        if opened_count >= 3 * len(self.owner.brief.wants) + 1:
            return True
        return self.owner.steps_taken >= max(8, self.owner.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        tool_name = self.owner.brief.roles.get("ask")
        schema = self.owner.schemas.get(tool_name or "") or {}
        props = schema.get("properties") or {}
        chunks = [k_ for k_, v_ in props.items() if isinstance(v_, dict) and "string" in _schema_types(v_) and "enum" not in v_]
        id_param = next((k_ for k_ in schema.get("required") or [] if k_ in chunks), None) or (chunks[0] if chunks else None)
        if not id_param:
            return None
        call_args: dict[str, Any] = {id_param: text}
        for key in schema.get("required") or []:
            if key not in call_args:
                param_specs = props.get(key) or {}
                if isinstance(param_specs.get("enum"), list) and param_specs["enum"]:
                    call_args[key] = param_specs["enum"][0]
                else:
                    return None
        faults: list[str] = []
        call_args = _fit_schema(call_args, schema, "args", faults)
        return None if faults else call_args

    def interview(self) -> str:
        self.asked = True
        qty = self.owner.relay.messages_left()
        if qty <= 0:
            return ""
        self.owner.scout.read_side()
        ranked_asks = self._turns_by_item(self._ranked(qty))
        if len(self.owner.brief.wants) >= 2 and qty >= 2:
            return self._interview_by_item(qty, ranked_asks)
        learned: list[str] = []
        opening_asks = ranked_asks[: qty - 1] if qty >= 2 else ranked_asks[:qty]
        just_asked = self.send_questions([(item_ix, prop, self.question_for(item_ix, prop)) for item_ix, prop in opening_asks])
        if not just_asked:
            return ""
        sent = list(just_asked)
        heard = self.parse_answers(just_asked)
        self.use_answers(heard, learned)
        retries = self._word_retries(heard, opening_asks[: len(just_asked)], opening_asks)
        if retries and self.owner.relay.messages_left() > 0:
            asked = self.send_questions([(retries[0][0], retries[0][1], self.question_for(*retries[0]))])
            if asked:
                sent += asked
                self.use_answers(self.parse_answers(asked), learned)
        elif qty >= 2 and self.owner.relay.messages_left() > 0:
            sent += self._send_final(heard, just_asked, opening_asks, ranked_asks, learned)
        if learned:
            self.owner.epoch += 1
        return self._interview_report(sent, learned)

    def _interview_by_item(self, qty: int, ranked_asks: list[tuple[int, str]]) -> str:
        """Several products: each product's best question goes out first, then every further message goes to a product whose
        unwritten preference is still unheard, one answer read at a time."""
        firsts: list[tuple[int, str]] = []
        for pick in ranked_asks:
            if all(pick[0] != f_[0] for f_ in firsts):
                firsts.append(pick)
        firsts = firsts[:qty]
        rest = [pick for pick in ranked_asks if pick not in firsts]
        learned: list[str] = []
        sent = self.send_questions([(item_ix, prop, self.question_for(item_ix, prop)) for item_ix, prop in firsts])
        if not sent:
            return ""
        heard = self.parse_answers(sent)
        heard_items = self._heard_items(heard, learned)
        asked_picks = list(firsts[: len(sent)])
        retries = self._word_retries(heard, asked_picks, asked_picks)
        while (rest or retries) and self.owner.relay.messages_left() > 0:
            if retries:
                pick = retries.pop(0)
            else:
                pick = next((p_ for p_ in rest if p_[0] not in heard_items), rest[0])
                rest.remove(pick)
            asked = self.send_questions([(pick[0], pick[1], self.question_for(*pick))])
            if not asked:
                break
            sent += asked
            asked_picks.append(pick)
            heard = self.parse_answers(asked)
            heard_items |= self._heard_items(heard, learned)
            retries += self._word_retries(heard, [pick], asked_picks)
        if learned:
            self.owner.epoch += 1
        return self._interview_report(sent, learned)

    def _word_retries(self, heard: dict[str, Any], batch: list[tuple[int, str]], asked_picks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Questions the shopper sent back as asking too much at once (and answered nothing for): each is asked again with
        only the last word of its property name, when that name has several words and that word was not asked yet."""
        told = {self._item_of(entry) for entry in _listify(heard.get("needs")) if isinstance(entry, dict)}
        retries: list[tuple[int, str]] = []
        for number in _listify(heard.get("crowded")):
            pos = _as_int(number)
            if not pos or not 1 <= pos <= len(batch):
                continue
            item_ix, prop = batch[pos - 1]
            prop_words = prop.split()
            if item_ix in told or len(prop_words) < 2:
                continue
            head = prop_words[-1]
            if any(i_ == item_ix and _plain_words(p_) == _plain_words(head) for i_, p_ in asked_picks + retries):
                continue
            retries.append((item_ix, head))
        if retries:
            self.owner.relay.trace.append({"kind": "word_retry", "ask": retries})
        return retries

    def _heard_items(self, heard: dict[str, Any], learned: list[str]) -> set[int]:
        self.use_answers(heard, learned)
        return {self._item_of(entry) for entry in _listify(heard.get("needs")) if isinstance(entry, dict) and str(entry.get("need") or "").strip()}

    def question_for(self, item_ix: int, prop: str, sample_values: list[str] | None = None) -> str:
        text = f"What {prop} would you like for the {self.owner.brief.wants[item_ix - 1].name}"
        return text + (f", for example {sample_values[0]} or {sample_values[1]}?" if sample_values else "?")

    def value_pair(self, item_ix: int, prop: str) -> list[str]:
        want_units = _plain_words(prop)
        settled_lines = " ".join(self.owner.brief.wants[item_ix - 1].musts).lower()
        for _, tool_name, _, datums, _ in self.owner.ledger.keys_text():
            if tool_name == want_units:
                return [v_ for v_ in datums if v_ and v_ not in settled_lines and len(v_) <= 30][:2]
        return []

    def send_questions(self, question_list: list[tuple[int, str, str]], steps_floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.owner.brief.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for item_ix, _, text in question_list:
            if self.owner.done or self.owner.relay.left() <= steps_floor or self.sent_count >= self.owner.brief.max_messages:
                break
            sent_ask, broke_off = self._send_one(tool, item_ix, text)
            if sent_ask is None:
                break
            sent.append(sent_ask)
            if broke_off:
                break
        return sent

    def use_answers(self, heard: dict[str, Any], learned: list[str]) -> None:
        for entry in _listify(heard.get("needs")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                self._add_need(entry, learned)
        for entry in _listify(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                self._drop_need(entry, learned)
        for entry in _listify(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                self.owner.brief.wants[self._item_of(entry) - 1].dont_care.append(str(entry["property"]))
        budget = heard.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.owner.brief.budget = budget

    def parse_answers(self, sent: list[tuple[int, str, str]], seats: tuple[str, ...] = (HEAVY_MODEL, LIGHT_MODEL)) -> dict[str, Any]:
        prompt = REPLY_READER + "\n\nITEMS:\n" + "\n".join(f"item {i_}: {wanted_.name}" for i_, wanted_ in enumerate(self.owner.brief.wants, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{lid_}. (about item {i_}) Q: {qtext_}\nA: {lo_}" for lid_, (i_, qtext_, lo_) in enumerate(sent, 1))
        reply_obj = self.owner.relay.ask_json(prompt, list(seats), 1500) or {}
        if "budget" in reply_obj:
            reply_obj["budget"] = self.owner.brief.as_written(reply_obj["budget"], " ".join(lo_ for _, _, lo_ in sent))
        return reply_obj

    def reply_text(self, observed: Any) -> str:
        if isinstance(observed, str):
            return observed
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for datum in node.values():
                    walk(datum, depth + 1)
            elif isinstance(node, list):
                for datum in node:
                    walk(datum, depth + 1)

        walk(observed, 0)
        return " ".join(found)[:800] if found else _dumps(observed)[:400]

    def _turns_by_item(self, ranked_asks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """With several products, each one gets its best question before any gets a second (planner order kept within an item)."""
        if len(self.owner.brief.wants) < 2:
            return ranked_asks
        lanes: dict[int, list[tuple[int, str]]] = {}
        for pick in ranked_asks:
            lanes.setdefault(pick[0], []).append(pick)
        ordered: list[tuple[int, str]] = []
        while any(lanes.values()):
            for item_ix in list(lanes):
                if lanes[item_ix]:
                    ordered.append(lanes[item_ix].pop(0))
        if ordered != ranked_asks:
            self.owner.relay.trace.append({"kind": "plan_turns", "ask": ordered})
        return ordered

    def _ranked(self, qty: int) -> list[tuple[int, str]]:
        plan = self.owner.relay.ask_json(self._planner_prompt(qty), [HEAVY_MODEL, LIGHT_MODEL], 1800) or {}
        count_items = len(self.owner.brief.wants)
        ranked_asks: list[tuple[int, str]] = []
        for pick in _listify(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            item_ix = min(max(1, _as_int(pick.get("item") or 1) or 1), count_items)
            prop = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not prop or len(prop.split()) > 5:
                continue
            if all(_plain_words(prop) != _plain_words(other_tool) for i_, other_tool in ranked_asks if i_ == item_ix):
                ranked_asks.append((item_ix, prop))
        self.owner.relay.trace.append({"kind": "plan", "ask": _clipped(plan.get("ask"), 120)})
        return ranked_asks

    def _final_ask(self, heard: dict[str, Any], just_asked: list[tuple[int, str, str]], opening_asks: list[tuple[int, str]],
                    ranked_asks: list[tuple[int, str]]) -> tuple[int, str, str] | None:
        """The last message: the next pick (a question already asks for one property only, so asking it again with examples gains nothing)."""
        if len(ranked_asks) > len(opening_asks):
            item_ix, prop = ranked_asks[len(opening_asks)]
            return item_ix, prop, self.question_for(item_ix, prop)
        return None

    def _interview_report(self, sent: list[tuple[int, str, str]], learned: list[str]) -> str:
        lines = [f"Q: {qtext_}\nA: {lo_}" for _, qtext_, lo_ in sent]
        summary = "Kyo's interviewer asked the shopper these questions (one step per question):\n" + "\n".join(lines)
        summary += '\nFirm requirements these answers add: ' + ("; ".join(learned) if learned else "none")
        summary += '. Check every pick against them (search again with the new values where needed).'
        pointer_needs = [need for _, need in self.owner.brief.lookups if any(need in t_ for t_ in learned)]
        if pointer_needs:
            summary += (' These take their value from another listing: ' + " | ".join(pointer_needs) + '. Find that listing, open its page in full, then open every listing that page leads to; the last page holds the value, and the order counts only after all of them are open.')
        if self.owner.bulletin.queued_notes:
            summary += '\n[notes that came up meanwhile] ' + " | ".join(self.owner.bulletin.queued_notes)
            self.owner.bulletin.queued_notes = []
        return summary

    def _planner_prompt(self, qty: int) -> str:
        brief = self.owner.brief
        settled_lines = [f"item {item_ix}: {want.name} - settled by the request: " + ("; ".join(want.musts) or "nothing")
                         for item_ix, want in enumerate(brief.wants, 1)]
        listing_seen = []
        last_label = None
        for pos, (prop, tool_name, carriers, datums, kinds) in enumerate(self.owner.ledger.keys_text(), 1):
            if prop != last_label:
                listing_seen.append(f"category {prop!r}:")
                last_label = prop
            listing_seen.append(f"  {pos}. {tool_name} (on {carriers}, {kinds} different value{'s' if kinds != 1 else ''}): " + " | ".join(datums))
        spare = len(brief.wants) if len(brief.wants) >= 2 else 1
        prompt = ASK_PLANNER.replace("COUNT", str(qty + spare)) + "\n\nREQUEST:\n" + self.owner.query + "\n\nITEMS:\n" + "\n".join(settled_lines)
        top_pick = [str(x_) for x_ in _listify(brief.parsed.get("ask_first")) if str(x_).strip()]
        if top_pick:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(top_pick)
        return prompt + "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing_seen) or "(no full listing records yet)")

    def _send_final(self, heard: dict[str, Any], just_asked: list[tuple[int, str, str]], opening_asks: list[tuple[int, str]],
                  ranked_asks: list[tuple[int, str]], learned: list[str]) -> list[tuple[int, str, str]]:
        retry_ask = self._final_ask(heard, just_asked, opening_asks, ranked_asks)
        if not retry_ask:
            return []
        asked = self.send_questions([retry_ask])
        if asked:
            self.use_answers(self.parse_answers(asked), learned)
        return asked

    def _item_of(self, entry: dict[str, Any]) -> int:
        return min(max(1, _as_int(entry.get("item")) or 1), len(self.owner.brief.wants))

    def _add_need(self, entry: dict[str, Any], learned: list[str]) -> None:
        brief = self.owner.brief
        item_ix = self._item_of(entry)
        need = str(entry["need"]).strip()
        want = brief.wants[item_ix - 1]
        if need not in want.heard_needs:
            want.heard_needs.append(need)
            learned.append(f"item {item_ix}: {need}" if len(brief.wants) > 1 else need)
        if entry.get("points") is True and (item_ix, need) not in brief.lookups:
            brief.lookups.append((item_ix, need))

    def _drop_need(self, entry: dict[str, Any], learned: list[str]) -> None:
        want = self.owner.brief.wants[self._item_of(entry) - 1]
        prop_gone = _plain_words(str(entry["property"]).split(":")[0])
        matches = [n_ for n_ in want.musts + want.heard_needs if n_ not in want.dropped and _plain_words(n_.split(":")[0]) == prop_gone]
        want.dropped.extend(matches)
        learned.extend(f"dropped {n_}" for n_ in matches)

    def _send_one(self, tool: str, item_ix: int, text: str) -> tuple[tuple[int, str, str] | None, bool]:
        """One message sent and logged: (item, question, reply) and whether the reply was an error; None when not sent."""
        call_args = self.message_args(text)
        if not call_args:
            return None, True
        env_reply = self.owner.relay.env_call(tool, call_args)
        if env_reply is None:
            return None, True
        self.sent_count += 1
        answer_obs = env_reply.get("observation")
        spoken_words = self.reply_text(answer_obs)
        clerk = self.owner.clerk
        clerk.record.append(f"step {self.owner.steps_taken}: message to the shopper {text!r} -> reply {spoken_words[:300]!r} (cart then: {self.owner.relay.cart_key()})")
        clerk.record_steps.append(self.owner.steps_taken)
        self.owner.relay.trace.append({"kind": "ask", "q": text, "a": spoken_words[:300]})
        return (item_ix, text, spoken_words), isinstance(answer_obs, dict) and bool(answer_obs.get("error"))


class Clerk:
    """Steps the request asks for at the order or the ending."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.gate_messages = 0
        self.gates_passed: set[tuple[str, str]] = set()
        self.record: list[str] = []
        self.record_steps: list[int] = []

    def run_gate(self, stage: str) -> tuple[bool, str]:
        brief = self.owner.brief
        step_items = brief.due_items(stage)
        if not step_items or self.owner.done or not brief.duties_at(stage):
            return True, ""
        end_tools = {brief.roles.get("buy"), brief.roles.get("refuse")} - {None}
        tool_defs = brief._tools_text(end_tools)
        spoken: list[str] = []
        pushed_once = [False]
        for _ in range(GATE_ROUNDS):
            seats = [HEAVY_MODEL, LIGHT_MODEL] if pushed_once[0] else [LIGHT_MODEL, HEAVY_MODEL]
            reply_obj = self.owner.relay.ask_json(self._gate_prompt(stage, step_items, tool_defs), seats, 2000)
            if reply_obj is None:
                break
            outcome = self._gate_round(stage, step_items, end_tools, reply_obj, spoken, pushed_once)
            if isinstance(outcome, tuple):
                return outcome
            if outcome == "stop":
                break
        return True, self._gate_report(spoken)

    def _gate_prompt(self, stage: str, step_items: list[dict[str, Any]], tool_defs: list[str]) -> str:
        record = "\n".join(f"{lid_}. {cart_id}" for lid_, cart_id in enumerate(self.record, 1)) or "(nothing yet)"
        return (STEP_GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                + "\n\nREQUEST:\n" + self.owner.query + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {s_['step']}" for s_ in step_items)
                + "\n\nLEDGER:\n" + self.owner.brief.rules_text() + "\n\nCART NOW: " + self.owner.reviewer.cart_text()
                + "\n\nRECORD (messages and step calls so far):\n" + record
                + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.owner.bulletin.updates[-4:]) if self.owner.bulletin.updates else "")
                + f"\n\nMESSAGES LEFT: {max(0, self.owner.brief.max_messages - self.owner.interviewer.sent_count)}   STEPS LEFT: {self.owner.relay.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                + "\n\nTOOLS:\n" + "\n".join(tool_defs))

    def _unproven(self, stage: str, step_items: list[dict[str, Any]], reply_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Steps the gate's answer does not prove done: no record line, or (a message step at the order) none since the cart changed."""
        shown_n = [e_ for e_ in _listify(reply_obj.get("steps")) if isinstance(e_, dict)]
        needs_talk = stage == "order" and any(s_["messages"] > 0 for s_ in step_items)
        proof_text = {lid_ for lid_ in range(1, len(self.record) + 1) if not needs_talk or (lid_ <= len(self.record_steps) and self.record_steps[lid_ - 1] > self.owner.till.cart_step)}
        return [e_ for e_ in shown_n if e_.get("done") is not True or not any((_as_int(i_) or 0) in proof_text for i_ in _listify(e_.get("lines")))]

    def _nudge(self, unproven_steps: list[dict[str, Any]]) -> None:
        self.owner.relay.trace.append({"kind": "gate_unproven", "steps": _clipped(unproven_steps, 120)})
        self.record.append(f"(Kyo: no line made since the cart last changed at step {self.owner.till.cart_step} shows " + "; ".join(str(e_.get("step") or "") for e_ in unproven_steps)[:200]
                             + f" done, and {self.owner.brief.max_messages - self.owner.interviewer.sent_count} message(s) and {self.owner.relay.left()} steps are left: give the call that does it)")
        self.record_steps.append(-1)

    def _gate_call(self, tool_name: str, call_spec: dict[str, Any], spoken: list[str]) -> bool:
        """Run the gate's store call; False when the gate stops here."""
        faults: list[str] = []
        call_args = _fit_schema(call_spec.get("args") if isinstance(call_spec.get("args"), dict) else {}, self.owner.schemas[tool_name], "args", faults)
        if faults:
            return False
        if tool_name == self.owner.brief.roles.get("ask"):
            return self._gate_ask(call_args, spoken)
        return self._gate_store(tool_name, call_args, spoken)

    @staticmethod
    def _gate_report(spoken: list[str]) -> str:
        return ('Before going on, Kyo did the steps the request asks for:\n' + "\n".join(spoken)) if spoken else ""

    def _gate_round(self, stage: str, step_items: list[dict[str, Any]], end_tools: set[Any], reply_obj: dict[str, Any],
                    spoken: list[str], pushed_once: list[bool]) -> Any:
        """One answer of the duty check: (False, text) when a reply changes the task; "again" after a push to prove a
        step; "stop" when the check is over; "next" after a store call."""
        call_spec = reply_obj.get("call") if isinstance(reply_obj.get("call"), dict) else None
        changed_text = self._reply_changes(stage, reply_obj, call_spec, spoken)
        if changed_text is not None:
            return False, changed_text
        return self._gate_next(stage, step_items, end_tools, reply_obj, call_spec, spoken, pushed_once)

    def _reply_changes(self, stage: str, reply_obj: dict[str, Any], call_spec: dict[str, Any] | None, spoken: list[str]) -> str | None:
        """The text for a shopper's reply that changes the task, or None when it does not."""
        learned: list[str] = []
        hold = ""
        if spoken:
            self.owner.interviewer.use_answers({"needs": reply_obj.get("needs")}, learned)
            hold = str(reply_obj.get("hold") or "").strip()
        self.owner.relay.trace.append({"kind": "gate", "stage": stage, "call": _clipped(call_spec, 200), "steps": _clipped(reply_obj.get("steps"), 120), "hold": hold[:200], "told": learned})
        if not (learned or hold):
            return None
        self.owner.epoch += 1
        return (self._gate_report(spoken) + "\n" if spoken else "") + (
            "The shopper's answer changes the task: " + "; ".join(learned + ([hold] if hold else [])) + '. Bring the cart in line with it, then check out again.')

    def _gate_next(self, stage: str, step_items: list[dict[str, Any]], end_tools: set[Any], reply_obj: dict[str, Any],
                   call_spec: dict[str, Any] | None, spoken: list[str], pushed_once: list[bool]) -> str:
        tool_name = str((call_spec or {}).get("tool") or "")
        unproven_steps = self._unproven(stage, step_items, reply_obj)
        if not call_spec and unproven_steps and not spoken and not pushed_once[0] and self._can_message(2):
            pushed_once[0] = True
            self._nudge(unproven_steps)
            return "again"
        if not call_spec or tool_name not in self.owner.schemas or tool_name in end_tools:
            if not call_spec:
                self.gates_passed.add((stage, self.owner.relay.cart_key() + f"#{self.owner.epoch}"))
            return "stop"
        brief = self.owner.brief
        if self.owner.relay.left() < 2 or (tool_name == brief.roles.get("ask") and self.owner.interviewer.sent_count >= brief.max_messages):
            return "stop"
        return "next" if self._gate_call(tool_name, call_spec, spoken) else "stop"

    def _can_message(self, steps_needed: int) -> bool:
        brief = self.owner.brief
        return bool(brief.roles.get("ask")) and self.owner.interviewer.sent_count < brief.max_messages and self.owner.relay.left() >= steps_needed

    def _gate_ask(self, call_args: dict[str, Any], spoken: list[str]) -> bool:
        text = next((v_ for v_ in call_args.values() if isinstance(v_, str)), "")
        just_asked = self.owner.interviewer.send_questions([(1, "", text)], 1)
        if not just_asked:
            return False
        self.gate_messages += 1
        spoken.append(f"Q: {text}\nA: {just_asked[0][2]}")
        return True

    def _gate_store(self, tool_name: str, call_args: dict[str, Any], spoken: list[str]) -> bool:
        relay = self.owner.relay
        env_reply = relay.env_call(tool_name, call_args)
        if env_reply is None:
            return False
        view = env_reply.get("observation")
        self.owner.ledger.absorb(view)
        self.record.append(f"step {self.owner.steps_taken}: {tool_name} {_dumps(call_args)} -> {relay.render(view)[:300]} (cart then: {relay.cart_key()})")
        self.record_steps.append(self.owner.steps_taken)
        spoken.append(f"{tool_name} {_dumps(call_args)} -> {relay.render(view)[:400]}")
        return True


class Bulletin:
    """Notes riding on tool results, and evidence lines."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.queued_notes: list[str] = []
        self.read_notes: set[str] = set()
        self.updates: list[str] = []
        self.sold_texts: set[str] = set()
        self.sold_flag = False

    def new_notes(self, observed: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observed, dict) and not self.owner.ledger.record_of(observed):
            for key, datum in observed.items():
                if isinstance(datum, str) and len(datum.split()) >= 5:
                    lines.append(datum)
                elif isinstance(datum, list) and datum and all(isinstance(v_, str) for v_ in datum):
                    lines.extend(v_ for v_ in datum if len(v_.split()) >= 5)
                elif isinstance(datum, dict) and not self.owner.ledger.record_of(datum, False, 1):
                    for line_props in datum.values():
                        if isinstance(line_props, str) and len(line_props.split()) >= 6:
                            lines.append(line_props)
        return [cart_id for cart_id in lines if cart_id not in self.read_notes]

    def parse_note(self, lines: list[str]) -> str:
        self.read_notes.update(lines)
        prompt = NOTE_READER + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.owner.brief.rules_text() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        reply_obj = self.owner.relay.ask_json(prompt, [LIGHT_MODEL, SPARE_MODEL], 1200) or {}
        if "budget" in reply_obj:
            reply_obj["budget"] = self.owner.brief.as_written(reply_obj["budget"], " ".join(lines))
        gone_ids = [str(gone_) for gone_ in _listify(reply_obj.get("gone")) if isinstance(gone_, (str, int))]
        self.owner.ledger.blocked.update(gone_ for gone_ in gone_ids if gone_ in self.owner.ledger.rows)
        self.sold_flag = bool(gone_ids) or reply_obj.get("sold_out") is True
        if self.sold_flag and len(lines) == 1:
            self.sold_texts.update(lines)
        if reply_obj.get("kind") not in (None, "change"):
            return ""
        altered = self._note_effects(reply_obj, gone_ids)
        summary = str(reply_obj.get("summary") or "").strip()
        if altered or summary:
            self.owner.epoch += 1
            self.owner.dispatch.recent_marks = []
            self.updates.append(summary or "; ".join(altered))
        return (summary + (" (" + "; ".join(altered) + ")" if altered else "")).strip()

    def evidence(self, observed: Any) -> str:
        lids = self._attr_rows(observed)
        need_list = self.owner.brief.active_needs()
        if not lids or not need_list:
            return ""
        tagged = len(self.owner.brief.wants) > 1
        lines = [text_ for text_ in (self._evidence_of(lid, need_list, tagged) for lid in lids[:12]) if text_]
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; a key naming a different property (another part or aspect) decides nothing, and a key under another name may still carry the same property] " + " | ".join(lines)

    def _want_at(self, entry: dict[str, Any]) -> Want:
        return self.owner.brief.wants[min(max(1, _as_int(entry.get("item")) or 1), len(self.owner.brief.wants)) - 1]

    def _drop_from_note(self, entry: dict[str, Any]) -> str:
        want = self._want_at(entry)
        id_param = str(entry["need"]).strip()
        live_needs = [n_ for n_ in want.musts + want.heard_needs if n_ not in want.dropped]
        matches = [n_ for n_ in live_needs if n_ == id_param] or [n_ for n_ in live_needs if _plain_words(n_.split(":")[0]) == _plain_words(id_param.split(":")[0])][:1]
        want.dropped.extend(matches or [id_param])
        return f"dropped {id_param}"

    def _add_from_note(self, entry: dict[str, Any]) -> str:
        want = self._want_at(entry)
        need = str(entry["need"]).strip()
        if need in want.dropped:
            want.dropped.remove(need)
        if need not in want.musts + want.heard_needs:
            want.heard_needs.append(need)
        return f"added {need}"

    def _attr_rows(self, observed: Any) -> list[str]:
        lids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                lid = self.owner.ledger.record_of(node, False, depth)
                if lid and _attrs_of(node) and lid not in lids:
                    lids.append(lid)
                    return
                for datum in node.values():
                    walk(datum, depth + 1)
            elif isinstance(node, list):
                for datum in node:
                    walk(datum, depth + 1)

        walk(observed, 0)
        return lids

    def _evidence_of(self, lid: str, need_list: list[tuple[int, str]], tagged: bool) -> str:
        attr_table = _attrs_of(self.owner.ledger.opened.get(lid) or {}) or {}
        rule_lines = []
        for item_ix, need in need_list[:8]:
            prop_name = need.split(":", 1)[0].strip()
            key_hits = [f"{k_}={str(v_)[:50]}" for k_, v_ in attr_table.items() if _key_matches(_token_set(prop_name), _token_set(k_))]
            if key_hits:
                rule_lines.append((f"(item {item_ix}) " if tagged else "") + f"{prop_name} -> " + ", ".join(key_hits[:3]))
        return f"{lid}: " + "; ".join(rule_lines) if rule_lines else ""

    def _note_effects(self, reply_obj: dict[str, Any], gone_ids: list[str]) -> list[str]:
        altered = []
        budget_val = reply_obj.get("budget")
        if isinstance(budget_val, (int, float)) and not isinstance(budget_val, bool) and budget_val > 0 and budget_val != self.owner.brief.budget:
            self.owner.brief.budget = budget_val
            altered.append(f"budget now {budget_val}")
        removals = [e_ for e_ in _listify(reply_obj.get("drop")) if isinstance(e_, dict) and str(e_.get("need") or "").strip()]
        additions = [e_ for e_ in _listify(reply_obj.get("add")) if isinstance(e_, dict) and str(e_.get("need") or "").strip()]
        altered += [self._drop_from_note(entry) for entry in removals]
        altered += [self._add_from_note(entry) for entry in additions]
        if gone_ids:
            altered.append("unavailable: " + ", ".join(gone_ids))
        if reply_obj.get("price_moved") is True:
            altered.append("a price changed")
        return altered


class Reviewer:
    """Checks before a checkout or an ending."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.checked: set[str] = set()
        self.review_done = False
        self.suggested: list[str] = []
        self.review_choice: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.fallback_ids: set[str] = set()
        self.fallback_done = False
        self.fallback_seen = 0
        self.auto_bought = False
        self.link_rounds = 0
        self.link_holds = 0
        self.walk_rounds = 0
        self.walk_steps = 0
        self.walk_searches: list[str] = []
        self.walked_for: tuple[str, ...] = ()
        self.walk_found: set[str] = set()

    def cart_text(self) -> str:
        if self.owner.till.cart_shown is not None and self.owner.till.cart_view_step >= self.owner.till.cart_step:
            return _dumps(_clipped(self.owner.till.cart_shown, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dumps(self.owner.till.lines)

    def cart_ids(self) -> list[str]:
        if self.owner.till.cart_shown is not None and self.owner.till.cart_view_step >= self.owner.till.cart_step:
            text = _dumps(self.owner.till.cart_shown)
            return [lid_ for lid_ in self.owner.ledger.rows if lid_ in text]
        return list(self.owner.till.lines)

    def record_line(self, lid: str) -> str:
        record = self.owner.ledger.opened.get(lid) or self.owner.ledger.rows.get(lid)
        return _dumps(_clipped(record, 400))[:3000] if record else "(no record seen)"

    def others_text(self, skip_tools: list[str], max_len: int = 30) -> str:
        lines = []
        id_order = list(reversed(self.owner.ledger.seen))
        if self.owner.brief.parsed.get("pick") == "cheapest":
            id_order.sort(key=lambda lid_: (lid_ not in self.owner.ledger.opened, self.owner.ledger.price_of(lid_) if self.owner.ledger.price_of(lid_) is not None else float("inf")))
        for lid in id_order:
            if lid in skip_tools or self.owner.ledger.gone(lid):
                continue
            record = self.owner.ledger.opened.get(lid) or self.owner.ledger.rows.get(lid)
            lines.append(_dumps(_clipped(record, 160))[:900])
            if len(lines) >= max_len:
                break
        return "\n".join(lines)

    def verdict_key(self, lids: list[str]) -> str:
        return "|".join(sorted(lids)) + f"#{self.owner.epoch}"

    def check_cart(self, lids: list[str] | None = None, spend_cap_now: float | None = None) -> str:
        spend_cap_now = self.owner.relay.soft_spend() if spend_cap_now is None else spend_cap_now
        plan_pair = lids is not None
        lids = lids if plan_pair else self.cart_ids()
        key = self.verdict_key(lids)
        if lids and any(lid_ in self.fallback_ids for lid_ in lids):
            self.verdicts[key] = True
            return ""
        if key in self.checked or self.owner.relay.cost > spend_cap_now or not lids:
            return ""
        self.checked.add(key)
        reply_obj = self.owner.relay.ask_json(self._check_prompt(lids, plan_pair), [HEAVY_MODEL, LIGHT_MODEL], 1500)
        self.owner.relay.trace.append({"kind": "check", "cart": lids, "got": reply_obj})
        self.verdicts[key] = bool(reply_obj) and reply_obj.get("ok") is not False
        if not reply_obj or reply_obj.get("ok") is not False:
            return ""
        return self._check_report(reply_obj, lids)

    def _unsettled_links(self) -> list[str]:
        live_needs = {need for _, need in self.owner.brief.active_needs()}
        return [need for _, need in self.owner.brief.lookups if need in live_needs and need not in self.owner.brief.lookup_values]

    def walk_due(self) -> bool:
        """A requirement points to a page not settled yet, and that set of pointers has not been walked."""
        pending = tuple(self._unsettled_links())
        return bool(pending) and pending != self.walked_for and self.walk_rounds < WALK_ROUNDS and self.owner.relay.steps_left() >= WALK_MIN_STEPS and self.owner.relay.left() >= WALK_MIN_STEPS

    def walk_links(self) -> str:
        """Kyo follows the pages requirements point to before the choice: each round either opens rows that may be the
        next link or runs one narrow search for it, then reads the new rows' pages."""
        relay, ledger, brief = self.owner.relay, self.owner.ledger, self.owner.brief
        tool = self.owner.kit.offer_tool()
        id_param = self.owner.kit.id_param_of(tool)
        search_tool = brief.roles.get("search")
        self.walked_for = tuple(self._unsettled_links())
        view: list[str] = []
        step_cap = max(WALK_MIN_STEPS // 2 + 1, self.owner.max_steps // 3)
        while self.walk_rounds < WALK_ROUNDS and not self.owner.done and tool and id_param:
            unsettled = self._unsettled_links()
            if not unsettled or relay.steps_left() < WALK_MIN_STEPS or relay.left() < WALK_MIN_STEPS or self.walk_steps >= step_cap:
                break
            steps_before = self.owner.steps_taken
            self.walk_rounds += 1
            reply_obj = relay.ask_json(self._walk_prompt(unsettled, search_tool), [LIGHT_MODEL, HEAVY_MODEL], 1800) or {}
            relay.trace.append({"kind": "walk", "pending": unsettled, "got": _clipped(reply_obj, 150)})
            if self._drop_non_pointers(reply_obj):
                break
            open_next, _ = self._parse_links(reply_obj, unsettled)
            open_next += [str(x_) for x_ in _listify(reply_obj.get("open")) if str(x_) in ledger.rows and str(x_) not in ledger.pages_read and str(x_) not in open_next]
            # a row is worth a step only when a search for the pointed-at shop brought it up: the page a pointer names is
            # known by its own fine print, not by a row that merely looks alike
            open_next = [lid_ for lid_ in open_next if lid_ in self.walk_found]
            if not open_next:
                found = self._walk_search(search_tool, reply_obj.get("search"))
                if found is None:
                    break
                open_next = [lid_ for lid_ in found if lid_ not in ledger.pages_read]
            if open_next and len(view) < WALK_PAGES:
                self._open_links(open_next, tool, id_param, view, WALK_OPENS)
            self.walk_steps += self.owner.steps_taken - steps_before
        self.walked_for = tuple(self._unsettled_links())
        text, _ = self._link_report(view, [])
        return text

    def _drop_non_pointers(self, reply_obj: dict[str, Any]) -> bool:
        """Requirements the walk judged to name no other listing (a fit with another item, a condition) leave the
        pointers; True when no pointer is left to walk."""
        brief = self.owner.brief
        not_links = {str(c_.get("need") or "").strip() for c_ in _listify(reply_obj.get("chains")) if isinstance(c_, dict) and c_.get("names_listing") is False}
        if not_links:
            brief.lookups = [(i_, need) for i_, need in brief.lookups if need not in not_links]
            self.owner.relay.trace.append({"kind": "walk_dropped", "needs": sorted(not_links)})
        return not self._unsettled_links()

    def _walk_prompt(self, unsettled: list[str], search_tool: str | None) -> str:
        prompt = LINK_WALKER + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(unsettled)
        prompt += "\n\nSEARCH TOOL: " + (f"{search_tool} | parameters: {_dumps(self.owner.schemas.get(search_tool) or {})}" if search_tool else "none")
        prompt += "\n\nSEARCHES DONE:\n" + ("\n".join(self.walk_searches[-6:]) or "(none)")
        return prompt + "\n\nRECORDS (newest first):\n" + "\n".join(self._link_records())

    def _walk_search(self, search_tool: str | None, args: Any) -> list[str] | None:
        """One narrow search for a pointed-at listing: the ids of the rows it returned, or None when no search runs."""
        if not search_tool or not isinstance(args, dict) or not args:
            return None
        faults: list[str] = []
        call_args = _fit_schema(args, self.owner.schemas.get(search_tool), "args", faults)
        mark = _dumps(call_args)
        if faults or any(done_.startswith(mark + " ->") for done_ in self.walk_searches):
            return None
        env_reply = self.owner.relay.env_call(search_tool, call_args)
        if env_reply is None or self.owner.done:
            return None
        observed = env_reply.get("observation")
        self.owner.scout.absorb_result(search_tool, observed)
        found = [lid_ for lid_, _ in self.owner.ledger.rows_in(observed)]
        self.walk_found.update(found)
        self.walk_searches.append(f"{mark} -> {len(found)} rows" + (": " + ", ".join(found[:8]) if found else ""))
        return found

    def trace_links(self) -> tuple[str, bool]:
        live_needs = {need for _, need in self.owner.brief.active_needs()}
        unsettled = [need for _, need in self.owner.brief.lookups if need in live_needs and need not in self.owner.brief.lookup_values]
        tool = self.owner.brief.roles.get("offer") or self.owner.brief.roles.get("open")
        id_param = self.owner.kit.id_param_of(tool)
        view: list[str] = []
        missing: list[str] = []
        while unsettled and self.link_rounds < 4 and self.owner.relay.cost < self.owner.relay.hard_spend():
            self.link_rounds += 1
            prompt = LINK_TRACER + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(unsettled)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(self._link_records())
            reply_obj = self.owner.relay.ask_json(prompt, [LIGHT_MODEL, HEAVY_MODEL], 1800) or {}
            self.owner.relay.trace.append({"kind": "trail", "pending": unsettled, "got": _clipped(reply_obj, 150)})
            open_next, missing = self._parse_links(reply_obj, unsettled)
            unsettled = [s_ for s_ in unsettled if s_ not in self.owner.brief.lookup_values]
            if not open_next or not tool or not id_param or self.owner.relay.left() < len(open_next) + 4 or self.owner.done:
                break
            if not self._open_links(open_next, tool, id_param, view):
                break
        return self._link_report(view, missing)

    def refresh_cart(self) -> str:
        tool = self.owner.kit.offer_tool()
        id_param = self.owner.kit.id_param_of(tool)
        if not tool or not id_param:
            return ""
        stale = self._stale_lines()
        if not stale or self.owner.relay.left() < len(stale) + 3:
            return ""
        lines = self._refresh_each(tool, id_param, stale)
        self.owner.relay.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.owner.epoch += 1
        return "\n".join(lines)

    def swap_note(self) -> str:
        missing = [lid_ for lid_ in self.cart_ids() if lid_ not in self.owner.ledger.sold_step and self.owner.ledger.gone(lid_)]
        if not missing:
            return ""
        for lid in missing:
            self.owner.ledger.sold_step[lid] = self.owner.steps_taken
        self.suggested = [lid_ for lid_ in self.suggested if lid_ not in missing]
        self.owner.epoch += 1
        chosen = self.swaps_for(missing)
        text = f"[Kyo] Sold out now: {', '.join(missing)} (the store shows it can no longer be bought). It will not come back, so do not open it again."
        if chosen and self.owner.relay.left() <= 6:
            return text + f" Few steps are left: in this turn add {chosen[0]} at the requested quantity and take out the sold-out line (two calls), then check out. Other listings already read that seem to fit: {', '.join(chosen[1:]) or 'none'}."
        if chosen and self.owner.brief.parsed.get("pick") == "cheapest":
            return text + f" Listings already read that seem to fit that item instead, cheapest first: {', '.join(chosen)}. The pick rule is the cheapest, so unless you have seen the whole fitting set already, run one search below {chosen[0]}'s price (the search's maximum price, if it has one) before you replace the sold-out line with the cheapest fitting listing."
        if chosen:
            return text + f" Listings already read that seem to fit that item instead: {', '.join(chosen)}. In this turn remove the sold-out line from the cart and add {chosen[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other fitting listing; search for one if none you have read fits.'

    def find_sold(self) -> str:
        lids = [lid_ for lid_ in self.cart_ids() if lid_ not in self.owner.ledger.sold_step]
        if any(self.owner.ledger.gone(lid_) for lid_ in lids):
            return ""
        if len(lids) == 1:
            self.owner.ledger.blocked.add(lids[0])
            return ""
        return self.refresh_cart() if lids else ""

    def swaps_for(self, missing: list[str]) -> list[str]:
        if self.owner.relay.cost > self.owner.relay.soft_spend():
            return []
        record_lines = self._swap_records(missing)
        if not record_lines:
            return []
        prompt = SWAP_FINDER + "\n\nREQUEST:\n" + self.owner.query + "\n\nREQUIREMENTS IN FORCE:\n" + self.owner.brief.rules_text()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{lid_}: {self.record_line(lid_)}" for lid_ in missing) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(record_lines)
        reply_obj = self.owner.relay.ask_json(prompt, [HEAVY_MODEL, LIGHT_MODEL], 1200)
        fits = self._swap_hits(reply_obj, missing)
        ranked_ids = [lid_ for _, lid_ in sorted(fits, key=lambda flag_: flag_[0])] if self.owner.brief.parsed.get("pick") == "cheapest" else [lid_ for _, lid_ in fits]
        self.owner.relay.trace.append({"kind": "swap", "lost": missing, "fits": fits})
        return ranked_ids[:4]

    def review_decline(self) -> str:
        if self.review_done or self.owner.relay.cost > self.owner.relay.soft_spend():
            return ""
        self.review_done = True
        reply_obj = self.review_prompt_run()
        keyed = self.review_hits(reply_obj)
        for _ in range(2):
            if keyed:
                break
            rows = [str(lid_) for lid_ in _listify((reply_obj or {}).get("open")) if str(lid_) in self.owner.ledger.rows and str(lid_) not in self.owner.ledger.pages_read and not self.owner.ledger.gone(str(lid_))]
            if not rows or not self.open_rows(rows):
                break
            reply_obj = self.review_prompt_run()
            keyed = self.review_hits(reply_obj)
        if not keyed:
            return ""
        self.review_choice = keyed
        self.suggested = [lid_ for lids in keyed.values() for lid_ in lids]
        all_read = all(lid_ in self.owner.ledger.opened for lid_ in self.suggested)
        next_move = ". Their pages are already open: cart the best of them for each item (keeping to the pick rule) and check out now; end without buying only if each of them fails." if all_read else '. Open them and buy what truly fits; end without buying only if each of them fails.'
        return 'An independent review thinks these listings may fit: ' + "; ".join(f"item {k_}: {', '.join(v_)}" for k_, v_ in keyed.items()) + (f" ({reply_obj.get('reason')})" if reply_obj.get("reason") else "") + next_move

    def closest_pick(self) -> bool:
        """The request's own rule for when nothing fits picks a listing to order anyway; True when Kyo now vouches for one."""
        brief, ledger = self.owner.brief, self.owner.ledger
        if brief.parsed.get("fallback_buy") is not True or (self.fallback_done and len(ledger.opened) <= self.fallback_seen):
            return False
        self.fallback_done = True
        self.fallback_seen = len(ledger.opened)
        opened_ids = [lid_ for lid_ in reversed(ledger.seen) if lid_ in ledger.opened][:24]
        if not opened_ids:
            return False
        prompt = CLOSEST_PICK.replace("FALLBACK", str(brief.parsed.get("fallback") or "order the closest listing")) + "\n\nREQUEST:\n" + self.owner.query
        prompt += "\n\nREQUIREMENTS IN FORCE:\n" + brief.rules_text() + "\n\nLISTING RECORDS SEEN (newest first):\n"
        prompt += "\n".join(("SOLD OUT " if ledger.gone(lid_) else "OPENED ") + _dumps(_clipped(ledger.opened.get(lid_), 200))[:1200] for lid_ in opened_ids)
        reply_obj = self._closest_reply(prompt)
        if not reply_obj.get("pick"):
            reply_obj = {"pick": self._closest_by_code(opened_ids), "by": "code"}
        self.owner.relay.trace.append({"kind": "closest", "got": reply_obj})
        picks = reply_obj.get("pick") if isinstance(reply_obj.get("pick"), dict) else {}
        choice = {str(k_): [str(v_)] for k_, v_ in picks.items() if str(v_) in ledger.opened and not ledger.gone(str(v_))}
        if len(brief.wants) == 1 and choice and "1" not in choice:
            choice = {"1": next(iter(choice.values()))}
        if not choice:
            return False
        self.review_choice = {**self.review_choice, **choice}
        self.fallback_ids = {lids[0] for lids in choice.values()}
        self.suggested = [lids[0] for lids in choice.values()]
        self.confirmed = self.suggested[0]
        return True

    def _closest_by_code(self, opened_ids: list[str]) -> dict[str, str]:
        """No model answer for the closest listing: per item, the in-stock opened listing of its kind within the budget
        whose record carries the most words of the item's requirements (the cheaper one on a tie)."""
        brief, ledger = self.owner.brief, self.owner.ledger
        cat_key = ledger.label_key()
        budget = brief.budget if isinstance(brief.budget, (int, float)) and brief.parsed.get("budget_scope") != "each" else None
        picks: dict[str, str] = {}
        for item_ix, want in enumerate(brief.wants, 1):
            needs = [n_ for i_, n_ in brief.active_needs() if i_ == item_ix]
            wanted = {w_ for n_ in needs for w_ in _token_set(n_.split(":", 1)[-1]) if len(w_) >= 3}
            kind_words = {w_ for w_ in _token_set(want.shelf or "") | _token_set(want.name) if len(w_) >= 4}
            best: tuple[int, float, str] | None = None
            for lid in opened_ids:
                record = ledger.opened.get(lid) or {}
                label = str(record.get(cat_key) or "") if cat_key else ""
                price = ledger.price_of(lid)
                if ledger.gone(lid) or not ledger.buyable(lid) or lid in picks.values():
                    continue
                if len(brief.wants) > 1 and not (kind_words & _token_set(label)):
                    continue
                if budget is not None and price is not None and price * want.qty > budget:
                    continue
                score = len(wanted & _token_set(_dumps(record)))
                rank = (score, -(price if price is not None else float("inf")), lid)
                if best is None or rank > best:
                    best = rank
            if best:
                picks[str(item_ix)] = best[2]
        return picks

    def _closest_reply(self, prompt: str) -> dict[str, Any]:
        """The closest-listing judgement; a reply cut short still yields its "pick" part when that part is complete."""
        for model_id in (HEAVY_MODEL, LIGHT_MODEL):
            message = self.owner.relay.ask_models([{"role": "user", "content": prompt}], [model_id], 3000, finishing=True)
            text = str((message or {}).get("content") or "")
            reply_obj = _first_object(text)
            if isinstance(reply_obj, dict) and reply_obj.get("pick"):
                return reply_obj
            found = re.search(r'"pick"\s*:\s*(\{[^{}]*\})', text)
            if found:
                try:
                    return {"pick": json.loads(found.group(1))}
                except ValueError:
                    pass
            if self.owner.relay.credit_out or self.owner.relay.no_credit:
                break
        return {}

    def review_prompt_run(self) -> dict[str, Any] | None:
        record_lines = []
        ledger = self.owner.ledger
        newest_first = [lid_ for lid_ in reversed(ledger.seen) if not ledger.gone(lid_)]
        opened_ids = [lid_ for lid_ in newest_first if lid_ in ledger.opened][:30]
        row_ids = [lid_ for lid_ in newest_first if lid_ not in ledger.opened and self._row_worth_review(lid_)][:40]
        record_lines += ["OPENED " + _dumps(_clipped(ledger.opened.get(lid_), 200))[:1200] for lid_ in opened_ids]
        record_lines += ["ROW " + _dumps(_clipped(ledger.rows.get(lid_), 120))[:500] for lid_ in row_ids]
        if not record_lines:
            return None
        prompt = NO_BUY_REVIEW + "\n\nREQUEST:\n" + self.owner.query + "\n\nREQUIREMENTS IN FORCE:\n" + self.owner.brief.rules_text()
        prompt += "\n\nLISTING RECORDS SEEN (opened pages first, each group newest first):\n" + "\n".join(record_lines)
        reply_obj = self.owner.relay.ask_json(prompt, [HEAVY_MODEL, LIGHT_MODEL], 1500)
        self.owner.relay.trace.append({"kind": "audit", "got": reply_obj})
        return reply_obj

    def _row_worth_review(self, lid: str) -> bool:
        """A search row the review may point at: buyable and, under a total budget, priced within it."""
        ledger, brief = self.owner.ledger, self.owner.brief
        price = ledger.price_of(lid)
        over_budget = brief.budget is not None and brief.parsed.get("budget_scope") != "each" and price is not None and isinstance(brief.budget, (int, float)) and price > brief.budget
        return ledger.buyable(lid) and not over_budget

    def review_hits(self, reply_obj: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (reply_obj or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        single = len(self.owner.brief.wants) == 1
        keyed = {("1" if single else str(k_)): [str(lid_) for lid_ in _listify(v_) if str(lid_) in self.owner.ledger.rows and not self.owner.ledger.gone(str(lid_))][:4] for k_, v_ in fits.items()}
        return {k_: v_ for k_, v_ in keyed.items() if v_}

    def open_rows(self, rows: list[str]) -> bool:
        plan = self._rows_plan(rows)
        plan = plan[: max(0, self.owner.relay.left() - 3)] if plan else plan
        if not plan:
            return False
        opened_before = len(self.owner.ledger.opened)
        if not self._run_rows_plan(plan):
            return False
        self.owner.relay.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.owner.ledger.opened) - opened_before})
        return len(self.owner.ledger.opened) > opened_before

    def _link_records(self) -> list[str]:
        record_lines = []
        for lid in reversed(self.owner.ledger.seen):
            record = (self.owner.ledger.opened.get(lid) or self.owner.ledger.rows.get(lid)) if lid in self.owner.ledger.pages_read else None
            if record:
                record_lines.append("OPENED " + _dumps(_clipped(record, 300))[:1400])
            else:
                record_lines.append("ROW " + _dumps(_clipped(self.owner.ledger.rows.get(lid), 80))[:300])
            if len(record_lines) >= 60:
                break
        return record_lines

    def _parse_links(self, reply_obj: dict[str, Any], unsettled: list[str]) -> tuple[list[str], list[str]]:
        """Chains the model traced: values taken when every link is open; links to open; links nobody has found."""
        open_next: list[str] = []
        missing: list[str] = []
        for chain in _listify(reply_obj.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = next((s_ for s_ in unsettled if s_ == str(chain.get("need") or "").strip()), None) or (unsettled[0] if len(unsettled) == 1 else None)
            if need:
                self._use_link_chain(chain, need, open_next, missing)
        return open_next, missing

    def _open_links(self, open_next: list[str], tool: str, id_param: str, view: list[str], per_round: int = 3) -> bool:
        for lid in open_next[: max(0, min(per_round, (WALK_PAGES if per_round != 3 else 6) - len(view)))]:
            faults: list[str] = []
            call_args = _fit_schema({id_param: lid}, self.owner.schemas.get(tool), "args", faults)
            if faults:
                return True
            env_reply = self.owner.relay.env_call(tool, call_args)
            if env_reply is None or self.owner.done:
                return True
            self.owner.scout.absorb_result(tool, env_reply.get("observation"))
            view.append(f"{lid}: " + self.owner.relay.render(env_reply.get("observation"))[:1500])
        return True

    def _link_report(self, view: list[str], missing: list[str]) -> tuple[str, bool]:
        text = ""
        if view:
            text += '[Kyo opened the pages requirements point at]\n' + "\n".join(view) + "\n"
        if self.owner.brief.lookup_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k_} -> {v_}" for k_, v_ in self.owner.brief.lookup_values.items()) + "\n"
        hold = bool(missing) and self.link_holds < 2 and self.owner.relay.steps_left() > 5 and self.owner.relay.left() >= 3
        if hold:
            self.link_holds += 1
            text += ("The order counts only once every page a requirement points to has been opened in full (a search row is not enough), following each page that refers on to a further listing. Still to find and open: "
                     + " | ".join(missing[:3]) + '. Search for its kind with the property values the description gives (shop names are seldom in titles), pick the row whose seller or brand field names that shop (when none does, open in full the rows matching those values: the shop may show only in the fine print of a page), open it, use the value its page holds, then check out.')
        return text, hold

    def _check_prompt(self, lids: list[str], plan_pair: bool) -> str:
        brief, ledger = self.owner.brief, self.owner.ledger
        prompt = CART_CHECK + "\n\nREQUEST:\n" + self.owner.query + "\n\nREQUIREMENTS IN FORCE:\n" + brief.rules_text()
        if plan_pair:
            units_said = "; ".join(f"item {i_}: quantity {wanted_.qty}" for i_, wanted_ in enumerate(brief.wants, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + units_said + "):\n" + ", ".join(lids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{lid_}: " + ("SOLD OUT (the store says it can no longer be bought). " if ledger.gone(lid_) else "") + self.record_line(lid_) for lid_ in lids)
        if brief.parsed.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others_text(lids, 15)
        return prompt

    def _check_report(self, reply_obj: dict[str, Any], lids: list[str]) -> str:
        ledger = self.owner.ledger
        self.suggested = [lid_ for lid_ in self.suggested if lid_ not in lids]
        issue_list = [str(s_) for s_ in _listify(reply_obj.get("problems"))][:5]
        better_ids = [str(hi_) for hi_ in _listify(reply_obj.get("better")) if str(hi_) in ledger.rows and not ledger.gone(str(hi_))][:4]
        text = 'An independent review finds this cart not yet right: ' + "; ".join(issue_list or ["see the requirements"])
        if better_ids:
            text += '. Listings that look like better picks: ' + ", ".join(better_ids)
        return text + '. Fix the cart (or show the review is wrong by reading the records again), then check out.'

    def _refresh_one(self, tool: str, id_param: str, lid: str) -> str | None:
        """One cart line read again: a line naming what changed, "" when nothing did, None when the re-reading stops."""
        ledger = self.owner.ledger
        faults: list[str] = []
        call_args = _fit_schema({id_param: lid}, self.owner.schemas.get(tool), "args", faults)
        if faults:
            return None
        opened_before = dict(ledger.rows.get(lid) or {})
        env_reply = self.owner.relay.env_call(tool, call_args)
        if env_reply is None or self.owner.done:
            return None
        observed = env_reply.get("observation")
        ledger.offer_step[lid] = self.owner.steps_taken
        self.owner.scout.absorb_result(tool, observed)
        after = ledger.rows.get(lid) or {}
        moved_keys = [k_ for k_, v_ in after.items() if k_ in opened_before and opened_before[k_] != v_ and not isinstance(v_, (dict, list))]
        if not moved_keys:
            return ""
        return f"{lid} changed ({', '.join(moved_keys[:4])}): {_dumps(_clipped(observed, 300))[:900]}"

    def _swap_records(self, missing: list[str]) -> list[str]:
        ledger = self.owner.ledger
        record_lines = []
        for lid in reversed(ledger.seen):
            if lid in missing or lid in ledger.sold_step or ledger.gone(lid):
                continue
            record_lines.append(_dumps(_clipped(ledger.opened.get(lid) or ledger.rows.get(lid), 200))[:1200])
            if len(record_lines) >= 25:
                break
        return record_lines

    def _swap_hits(self, reply_obj: Any, missing: list[str]) -> list[tuple[float, str]]:
        ledger = self.owner.ledger
        fits: list[tuple[float, str]] = []
        for entry in _listify((reply_obj or {}).get("fits")):
            lid = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if lid in ledger.rows and lid not in missing and lid not in ledger.sold_step and lid not in self.cart_ids() and lid not in [lid_ for _, lid_ in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), lid))
        return fits

    def _rows_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        """One side-by-side read of the rows, or single reads of the first two; None when neither tool is there."""
        kit = self.owner.kit
        side_param, compare_tool = kit.side_param(), self.owner.brief.roles.get("compare")
        tool = kit.attr_tool()
        id_param = kit.id_param_of(tool)
        if side_param and compare_tool and not kit.side_bare():
            size = kit.safe_side_cap()
            batches = [rows[i_: i_ + size] for i_ in range(0, min(len(rows), 2 * size), size)]
            return [(compare_tool, {side_param: batch}) for batch in batches if batch]
        if tool and id_param:
            return [(tool, {id_param: lid_}) for lid_ in rows[:2]]
        return None

    def _run_rows_plan(self, plan: list[tuple[str, dict[str, Any]]]) -> bool:
        """The planned reads run in order; False when one does not fit its schema."""
        for tool_name, call_args in plan:
            faults: list[str] = []
            call_args = _fit_schema(call_args, self.owner.schemas.get(tool_name), "args", faults)
            if faults:
                return False
            env_reply = self.owner.relay.env_call(tool_name, call_args)
            if env_reply is None or self.owner.done:
                break
            self.owner.scout.absorb_result(tool_name, env_reply.get("observation"))
            self._retry_smaller(tool_name, call_args, env_reply.get("observation"))
        return True

    def _retry_smaller(self, tool_name: str, call_args: dict[str, Any], observed: Any) -> None:
        """A side-by-side read the store showed nothing for: learn the size and send the first part again once."""
        kit = self.owner.kit
        side_param = kit.side_param() if tool_name == self.owner.brief.roles.get("compare") else None
        sent = call_args.get(side_param) if side_param else None
        if not isinstance(sent, list) or len(sent) < 2:
            return
        if any(str(lid_) in _dumps(observed) for lid_ in sent):
            kit.side_good_n = max(kit.side_good_n, len(sent))
            return
        kit.side_bad_n = min(kit.side_bad_n or 999, len(sent))
        if self.owner.relay.left() < 4 or self.owner.done:
            return
        env_reply = self.owner.relay.env_call(tool_name, {**call_args, side_param: sent[: kit.side_cap()]})
        if env_reply is not None and not self.owner.done:
            self.owner.scout.absorb_result(tool_name, env_reply.get("observation"))

    def _stale_lines(self) -> list[str]:
        """Cart lines whose offer was last read before the latest refusal or more than two steps ago."""
        ledger = self.owner.ledger
        since_step = max(self.owner.steps_taken - 2, self.owner.till.refusal_step)
        return [lid_ for lid_ in self.cart_ids() if ledger.offer_step.get(lid_, -99) < since_step and lid_ not in ledger.sold_step]

    def _refresh_each(self, tool: str, id_param: str, stale: list[str]) -> list[str]:
        lines = []
        for lid in stale:
            changed = self._refresh_one(tool, id_param, lid)
            if changed is None:
                break
            if changed:
                lines.append(changed)
        return lines

    def _use_link_chain(self, chain: dict[str, Any], need: str, open_next: list[str], missing: list[str]) -> None:
        """One traced chain: its value kept when every link is open, else its unread links queued or its missing link noted."""
        ledger = self.owner.ledger
        chain_links = [link_ for link_ in _listify(chain.get("links")) if isinstance(link_, dict)]
        lids = [str(link_.get("id") or "").strip() for link_ in chain_links]
        if chain.get("done") is True and chain_links and all(lid_ in ledger.pages_read for lid_ in lids) and str(chain.get("value") or "").strip():
            self.owner.brief.lookup_values[need] = str(chain["value"]).strip()
            self.owner.epoch += 1
            return
        unread_ids = [lid_ for lid_ in lids if lid_ in ledger.rows and lid_ not in ledger.pages_read]
        if unread_ids:
            open_next += [lid_ for lid_ in unread_ids if lid_ not in open_next]
            return
        described_as = next((str(link_.get("about") or "") for link_, lid_ in zip(chain_links, lids) if lid_ not in ledger.rows), "") or "the listing it points at"
        missing.append(f"{need} -> find {described_as}")


class Till:
    """The cart, the checkout and the ending arguments."""

    def cart_total(self) -> float | None:
        """Price times count over the lines Kyo knows are held; None when a price is unknown or the cart is empty."""
        if not self.lines:
            return None
        total = 0.0
        for lid_, n_ in self.lines.items():
            price = self.owner.ledger.price_of(lid_)
            if price is None:
                return None
            total += price * n_
        return total

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.cart_touched = False
        self.cart_step = -1
        self.cart_view_step = -1
        self.cart_shown: Any = None
        self.lines: dict[str, int] = {}
        self.cart_obs = ""
        self.refusal_step = -1
        self.sold_stops = 0
        self.top_cart: list[str] = []
        self.dropped: set[str] = set()
        self.checkout_count = 0
        self.sets_learned = False
        self.last_sent: dict[str, int] = {}

    def cart_calls(self, lines: dict[str, int], to_drop: list[str]) -> list[dict[str, Any]] | None:
        tool_name = self.owner.brief.roles.get("cart") or ""
        schema = self.owner.schemas.get(tool_name) or {}
        props = schema.get("properties") or {}
        full_cart = self._full_cart_call(props, lines)
        if full_cart is not False:
            return full_cart
        id_param, qty, mode_param = self._line_call(tool_name, schema, props)
        if not id_param:
            return None
        removals = self._remove_calls(tool_name, props, to_drop, id_param, qty, mode_param)
        if removals is None:
            return None
        additions = self._add_calls(tool_name, props, lines, id_param, qty, mode_param)
        return None if additions is None else removals + additions

    def buy_confirmed(self) -> str:
        reviewer = self.owner.reviewer
        cart_tool, checkout_tool = self.owner.brief.roles.get("cart"), self.owner.brief.roles.get("buy")
        if reviewer.auto_bought or not reviewer.confirmed or self.owner.ledger.gone(reviewer.confirmed) or not cart_tool or not checkout_tool:
            return ""
        staged = self._confirmed_ready()
        if staged is None:
            return ""
        reviewer.auto_bought = True
        lines, to_drop, pendings = staged
        done = self._set_cart(cart_tool, pendings)
        if done is None:
            return ""
        call_args, refused = self._checkout_args(checkout_tool)
        replies = [] if refused else self._checkout_confirmed(checkout_tool, call_args, lines, to_drop)
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + reviewer.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def sets_count(self, tool_name: str) -> bool:
        if self.sets_learned:
            return True
        props = self.owner.schemas.get(tool_name, {}).get("properties") or {}
        if any(isinstance(sp_, dict) and isinstance(sp_.get("enum"), list) and len(sp_["enum"]) >= 2 for sp_ in props.values()):
            return False
        return any(isinstance(sp_, dict) and "integer" in _schema_types(sp_) and sp_.get("minimum") == 0 for sp_ in props.values())

    def note_cart(self, tool_name: str, call_args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_step = self.owner.steps_taken
        param_specs = self.owner.schemas.get(tool_name, {}).get("properties") or {}
        is_removal = any(self._is_removal_value(param_specs.get(key) or {}, datum) for key, datum in call_args.items())
        lines, full_cart = self._named_lines(call_args, param_specs)
        if full_cart:
            self.lines = {lid_: u_ for lid_, u_ in lines if u_ and u_ > 0}
            return
        sets_units = self.sets_count(tool_name)
        for lid, qty in lines:
            if qty is not None and not is_removal:
                self.last_sent[lid] = qty
            self._note_line(lid, qty, is_removal, sets_units)

    def learn_from_view(self, observed: Any) -> None:
        """The cart as the store shows it: Kyo's line counts follow it, and a count equal to the last number sent where adding would have summed shows the cart call sets counts."""
        shown = self._shown_counts(observed)
        if not shown:
            return
        for lid, count in shown.items():
            if not self.sets_learned and self.lines.get(lid, 0) > count == self.last_sent.get(lid):
                self.sets_learned = True
                self.owner.relay.trace.append({"kind": "cart_sets", "id": lid, "count": count})
        self.lines = {lid_: c_ for lid_, c_ in shown.items() if c_ > 0}

    def _shown_counts(self, observed: Any) -> dict[str, int]:
        """Counts per listing in a cart view: in each line naming a known listing, its one whole-number field that is not the price."""
        found: dict[str, int] = {}
        price_key = self.owner.scout.price_key

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                lid = next((str(v_) for v_ in node.values() if isinstance(v_, str) and v_ in self.owner.ledger.rows), None)
                counts = [v_ for k_, v_ in node.items() if k_ != price_key and isinstance(v_, int) and not isinstance(v_, bool) and v_ >= 0]
                if lid and len(counts) == 1:
                    found[lid] = counts[0]
                    return
                for v_ in node.values():
                    walk(v_, depth + 1)
            elif isinstance(node, list):
                for v_ in node:
                    walk(v_, depth + 1)
        walk(observed, 0)
        return found

    def is_checkout(self, tool_name: str, call_args: dict[str, Any]) -> bool:
        if tool_name != self.owner.brief.roles.get("buy"):
            return False
        mode_sets = self.owner.brief.parsed.get("buy_args") if isinstance(self.owner.brief.parsed.get("buy_args"), dict) else {}
        return all(call_args.get(k_) == v_ for k_, v_ in mode_sets.items())

    def is_decline(self, tool_name: str, call_args: dict[str, Any]) -> bool:
        if tool_name != self.owner.brief.roles.get("refuse"):
            return False
        mode_sets = self.owner.brief.parsed.get("refuse_args") if isinstance(self.owner.brief.parsed.get("refuse_args"), dict) else {}
        return all(call_args.get(k_) == v_ for k_, v_ in mode_sets.items())

    def bare_decline(self, tool_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
        schema = self.owner.schemas.get(tool_name) or {}
        props = schema.get("properties") or {}
        result_rows = dict(call_args)
        for key, param_specs in props.items():
            if not isinstance(param_specs, dict) or "string" not in _schema_types(param_specs) or "enum" in param_specs:
                continue
            datum = result_rows.get(key)
            if datum is None and key not in (schema.get("required") or []):
                continue
            text = str(datum or "")
            keyed = any(lid in text for lid in self.owner.ledger.rows) or self.owner.brief.parsed.get("quiet_decline") is not False
            if keyed or not text:
                result_rows[key] = 'Sorry, nothing in the store meets every one of your requirements, so I am not buying anything.'[: param_specs.get("maxLength", 200)]
        return result_rows

    def titles_to_ids(self, tool_name: str, call_args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        roles, ledger = self.owner.brief.roles, self.owner.ledger
        if tool_name not in {roles.get(r_) for r_ in ("cart", "open", "offer", "compare")} - {None} or not ledger.id_key:
            return call_args, ""
        props = (self.owner.schemas.get(tool_name) or {}).get("properties") or {}
        free_args = [k_ for k_, v_ in props.items() if isinstance(v_, dict) and ("string" in _schema_types(v_) or "array" in _schema_types(v_)) and "enum" not in v_]
        if len(free_args) != 1 or (tool_name != roles.get("cart") and (tool_name, free_args[0]) not in ledger.id_args):
            return call_args, ""
        result_rows = dict(call_args)
        for key in free_args:
            if key not in call_args:
                continue
            values = call_args[key] if isinstance(call_args[key], list) else [call_args[key]]
            mode_sets = [self._title_id(v_) for v_ in values]
            issue = next((s_ for _, s_ in mode_sets if s_), "")
            if issue:
                return call_args, issue
            result_rows[key] = [v_ for v_, _ in mode_sets] if isinstance(call_args[key], list) else mode_sets[0][0]
        return result_rows, ""

    def _kind_of(self, lid: str, cat_key: str | None) -> str:
        return str((self.owner.ledger.rows.get(lid) or {}).get(cat_key) or "").strip().lower() if cat_key else lid

    def _confirmed_lines(self, cat_key: str | None) -> dict[str, int]:
        plan: dict[str, int] = {}
        for item_ix, lids in self.owner.reviewer.review_choice.items():
            pos = _as_int(item_ix)
            pick = next((lid_ for lid_ in lids if not self.owner.ledger.gone(lid_)), None)
            if pick and pos and 1 <= pos <= len(self.owner.brief.wants) and self._kind_of(pick, cat_key) not in {self._kind_of(s_, cat_key) for s_ in plan}:
                plan[pick] = self.owner.brief.wants[pos - 1].qty
        return plan

    def _set_cart(self, cart_tool: str, pendings: list[dict[str, Any]]) -> list[str] | None:
        done = []
        for call_args in pendings:
            faults: list[str] = []
            call_args = _fit_schema(call_args, self.owner.schemas.get(cart_tool), "args", faults)
            if faults:
                return None
            env_reply = self.owner.relay.env_call(cart_tool, call_args)
            if env_reply is None or self.owner.done:
                return None
            self.note_cart(cart_tool, call_args)
            self.owner.scout.absorb_result(cart_tool, env_reply.get("observation"))
            done.append(_dumps(call_args))
        return done

    def _checkout_confirmed(self, checkout_tool: str, call_args: dict[str, Any], lines: dict[str, int], to_drop: list[str]) -> list[str]:
        """The checkout of the reviewed cart: a refusal gets one retry after a re-read and a fresh check."""
        replies: list[str] = []
        for pass_no in range(2):
            if pass_no and not self._still_confirmed(lines):
                break
            if not self._confirmed_try(checkout_tool, call_args, lines, to_drop, replies):
                break
        return replies

    def _named_lines(self, call_args: dict[str, Any], param_specs: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
        """The (listing, units) lines a cart call names, and whether it states the whole cart."""
        flat_units = next((_as_int(v_) for v_ in call_args.values() if _as_int(v_) is not None and not isinstance(v_, str)), None)
        lines: list[tuple[str, int | None]] = []
        full_cart = False
        for key, datum in call_args.items():
            if isinstance(datum, list):
                full_cart = self._lines_of_list(datum, (param_specs.get(key) or {}).get("items") or {}, flat_units, lines) or full_cart
            elif isinstance(datum, str) and datum in self.owner.ledger.rows:
                lines.append((datum, flat_units))
        return lines, full_cart

    def _full_cart_call(self, props: dict[str, Any], lines: dict[str, int]) -> list[dict[str, Any]] | None | bool:
        """A cart tool taking the whole cart as a list of line objects: the one call (None: no id key); False: no such list."""
        for key, param_specs in props.items():
            pieces = param_specs.get("items") if isinstance(param_specs, dict) and "array" in _schema_types(param_specs) else None
            if isinstance(pieces, dict) and isinstance(pieces.get("properties"), dict):
                line_props = pieces["properties"]
                id_field = next((k_ for k_, sp_ in line_props.items() if isinstance(sp_, dict) and "string" in _schema_types(sp_) and "enum" not in sp_), None)
                units_key = next((k_ for k_, sp_ in line_props.items() if isinstance(sp_, dict) and "integer" in _schema_types(sp_)), None)
                if not id_field:
                    return None
                return [{key: [{id_field: lid_, **({units_key: i_} if units_key else {})} for lid_, i_ in lines.items()]}]
        return False

    def _title_id(self, datum: Any) -> tuple[Any, str]:
        """A listing named by the start of a title: its id when one seen row carries that text; else what is wrong."""
        ledger = self.owner.ledger
        if isinstance(datum, str) and datum.strip() in ledger.rows:
            return datum.strip(), ""
        if not isinstance(datum, str) or _id_shaped(datum, True) or len(datum.strip()) < 8:
            return datum, ""
        low_val = datum.strip().lower()
        key_hits = [lid_ for lid_, row_rec in ledger.rows.items() if any(k_ != ledger.id_key and isinstance(v_, str) and v_.strip().lower().startswith(low_val) for k_, v_ in row_rec.items())]
        if len(key_hits) == 1:
            return key_hits[0], ""
        if key_hits:
            return datum, f"several listings carry the text {datum!r} ({', '.join(key_hits[:6])}); pass one listing id"
        return datum, f"{datum!r} is not a listing id; pass the id from the result rows (their {ledger.id_key} field)"

    def _line_call(self, tool_name: str, schema: dict[str, Any], props: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        """The cart tool's id, count and mode arguments."""
        chunks = [k_ for k_, sp_ in props.items() if isinstance(sp_, dict) and "string" in _schema_types(sp_) and "enum" not in sp_]
        id_param = (next((k_ for t_, k_ in self.owner.ledger.id_args if t_ == tool_name and k_ in chunks), None)
                  or next((k_ for k_ in schema.get("required") or [] if k_ in chunks), None) or (chunks[0] if len(chunks) == 1 else None))
        qty = next((k_ for k_, sp_ in props.items() if isinstance(sp_, dict) and "integer" in _schema_types(sp_)), None)
        mode_param = next((k_ for k_, sp_ in props.items() if isinstance(sp_, dict) and isinstance(sp_.get("enum"), list) and len(sp_["enum"]) >= 2), None)
        return id_param, qty, mode_param

    def _remove_calls(self, tool_name: str, props: dict[str, Any], to_drop: list[str], id_param: str, qty: str | None,
                       mode_param: str | None) -> list[dict[str, Any]] | None:
        pendings: list[dict[str, Any]] = []
        for lid in to_drop:
            units_held = self.lines.get(lid, 1)
            if mode_param:
                pendings.append({id_param: lid, mode_param: props[mode_param]["enum"][1], **({qty: units_held} if qty else {})})
            elif qty and self.sets_count(tool_name):
                pendings.append({id_param: lid, qty: 0})
            elif qty and isinstance(props[qty].get("minimum"), int) and props[qty]["minimum"] <= -units_held:
                pendings.append({id_param: lid, qty: -units_held})
            else:
                return None
        return pendings

    def _add_calls(self, tool_name: str, props: dict[str, Any], lines: dict[str, int], id_param: str, qty: str | None,
                        mode_param: str | None) -> list[dict[str, Any]] | None:
        pendings: list[dict[str, Any]] = []
        for lid, want_units in lines.items():
            units_held = self.lines.get(lid, 0) if lid in self.owner.reviewer.cart_ids() else 0
            if units_held == want_units:
                continue
            if units_held and not self.sets_count(tool_name):
                return None
            pendings.append({id_param: lid, **({mode_param: props[mode_param]["enum"][0]} if mode_param else {}), **({qty: want_units} if qty else {})})
        return pendings

    def _confirmed_cart(self) -> tuple[dict[str, int], list[str]] | None:
        """The vouched-for cart: the reviewed lines plus the cart lines of other kinds, and the lines to take off."""
        cat_key = self.owner.ledger.label_key()
        plan = self._confirmed_lines(cat_key)
        if self.owner.reviewer.confirmed not in plan:
            return None
        cart_now = self.owner.reviewer.cart_ids()
        plan_kinds = {self._kind_of(s_, cat_key) for s_ in plan}
        reserve = {lid_: self.lines.get(lid_, 1) for lid_ in cart_now if lid_ not in plan and not self.owner.ledger.gone(lid_) and self._kind_of(lid_, cat_key) not in plan_kinds}
        to_drop = [lid_ for lid_ in cart_now if lid_ not in plan and lid_ not in reserve]
        if len(plan) + len({self._kind_of(lid_, cat_key) for lid_ in reserve}) < len(self.owner.brief.wants):
            return None
        return {**reserve, **plan}, to_drop

    def _still_confirmed(self, lines: dict[str, int]) -> bool:
        reviewer = self.owner.reviewer
        reviewer.refresh_cart()
        reviewer.check_cart(list(lines), self.owner.relay.hard_spend())
        return not (not reviewer.verdicts.get(reviewer.verdict_key(list(lines))) or self.owner.relay.left() < 1 or self.owner.done
                    or any(self.owner.ledger.gone(lid_) for lid_ in lines))

    def _confirmed_try(self, checkout_tool: str, call_args: dict[str, Any], lines: dict[str, int], to_drop: list[str], replies: list[str]) -> bool:
        """One checkout of the reviewed cart; True when a refusal leaves room for the retry."""
        go_on, spoken = self.owner.clerk.run_gate("order")
        if spoken:
            replies.append(spoken)
        if not go_on:
            return False
        env_reply = self.owner.relay.env_call(checkout_tool, call_args)
        self.owner.relay.trace.append({"kind": "completed", "lines": lines, "drop": to_drop, "reply": _clipped(env_reply, 200)})
        if env_reply is None:
            return False
        self.checkout_count += 1
        if not env_reply.get("finished"):
            self.refusal_step = self.owner.steps_taken
        self.owner.scout.absorb_result(checkout_tool, env_reply.get("observation"))
        replies.append(self.owner.relay.render(env_reply.get("observation")))
        return not (self.owner.done or env_reply.get("finished"))

    def _confirmed_ready(self) -> tuple[dict[str, int], list[str], list[tuple[str, dict[str, Any]]]] | None:
        """The reviewed cart and its cart calls, once the attributes are read and the check still confirms it."""
        reviewer = self.owner.reviewer
        plan_pair = self._confirmed_cart()
        if plan_pair is None:
            return None
        lines, to_drop = plan_pair
        if self.owner.dispatch._fit_report(list(lines)):
            # the planned lines break each other's model codes: Kyo does not buy them itself
            self.owner.relay.trace.append({"kind": "fit_refused", "lines": list(lines)})
            return None
        pendings = self.cart_calls(lines, to_drop)
        if not pendings or len(pendings) + 1 > self.owner.relay.steps_left():
            return None
        self.owner.scout.open_attrs(list(lines), len(pendings) + 1, by_steps=True)
        reviewer.check_cart(list(lines), self.owner.relay.hard_spend())
        if not reviewer.verdicts.get(reviewer.verdict_key(list(lines))):
            return None
        return lines, to_drop, pendings

    def _checkout_args(self, checkout_tool: str) -> tuple[Any, bool]:
        faults: list[str] = []
        call_args = _fit_schema(dict(self.owner.brief.parsed.get("buy_args") or {}), self.owner.schemas.get(checkout_tool), "args", faults)
        return call_args, bool(faults)

    @staticmethod
    def _is_removal_value(field_spec: dict[str, Any], datum: Any) -> bool:
        """An enum argument set past its first value: the call takes units off."""
        return isinstance(field_spec.get("enum"), list) and len(field_spec["enum"]) >= 2 and datum in field_spec["enum"][1:]

    def _note_line(self, lid: str, qty: int | None, is_removal: bool, sets_units: Any) -> None:
        units_held = self.lines.get(lid, 0)
        if is_removal:
            units_held = units_held - qty if qty else 0
        elif sets_units:
            units_held = qty if qty is not None else 1
        else:
            units_held = units_held + (qty if qty is not None else 1)
        if units_held > 0:
            self.lines[lid] = units_held
        else:
            self.lines.pop(lid, None)

    def _lines_of_list(self, datum: list[Any], pieces: Any, flat_units: int | None, lines: list[tuple[str, int | None]]) -> bool:
        """Lines from a list argument; True when the list states the whole cart (line objects, or an empty list of them)."""
        full_cart = bool(not datum and isinstance(pieces, dict) and (pieces.get("type") == "object" or "properties" in pieces))
        catalog = self.owner.ledger.rows
        for cart_id in datum:
            if isinstance(cart_id, dict):
                full_cart = True
                lid = next((v_ for v_ in cart_id.values() if isinstance(v_, str) and v_ in catalog), None)
                qty = next((_as_int(v_) for v_ in cart_id.values() if _as_int(v_) is not None and not isinstance(v_, str)), 1)
                if lid:
                    lines.append((lid, qty))
            elif isinstance(cart_id, str) and cart_id in catalog:
                lines.append((cart_id, flat_units))
        return full_cart


class Dispatcher:
    """The model's tool calls run with Kyo's guards, and the forced finish."""

    def __init__(self, owner: "Errand") -> None:
        self.owner = owner
        self.recent_marks: list[str] = []
        self.memo: dict[str, tuple[int, int, str]] = {}
        self.decline_stops = 0
        self.decline_nudged = False
        self.late_stop_done = False
        self.fallback_held = False
        self.short_cart_holds = 0
        self.over_budget_told = False
        self.fallback_searched = False
        self.sweep_note = ""
        self.fit_holds = 0

    def offered_tools(self) -> list[dict[str, Any]]:
        remaining = self.owner.relay.left()
        ask_tool = self.owner.brief.roles.get("ask")
        reserve: set[str] | None = None
        end_set = {self.owner.brief.roles.get("buy"), self.owner.brief.roles.get("refuse")} - {None}
        if remaining <= self.owner.relay.kept_steps():
            reserve = set(end_set)
            if len(self.owner.till.lines) < len(self.owner.brief.wants) and self.owner.relay.steps_left() >= 2:
                reserve |= {self.owner.brief.roles.get("cart")} - {None}
        elif remaining <= self.owner.relay.kept_steps() + 3:
            reserve = set(end_set) | {self.owner.brief.roles.get("cart"), self.owner.brief.roles.get("cart_view")}
        elif remaining <= len(self.owner.brief.wants) + 5 and self.owner.till.checkout_count == 0:
            reserve = set(end_set) | {self.owner.brief.roles.get("cart"), self.owner.brief.roles.get("cart_view"), self.owner.brief.roles.get("offer")}
        tool_defs = [t_ for t_ in self.owner.tool_defs if t_["function"]["name"] != ask_tool]
        if reserve is not None and end_set:
            narrowed_tools = [t_ for t_ in tool_defs if t_["function"]["name"] in reserve]
            tool_defs = narrowed_tools or tool_defs
        return tool_defs

    def merge_reads(self, pendings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        roles, kit = self.owner.brief.roles, self.owner.kit
        open_tool, compare_tool = roles.get("open"), roles.get("compare")
        id_param = kit.id_param_of(open_tool)
        side_param = kit.side_param()
        if not open_tool or not compare_tool or not id_param or not side_param or open_tool == compare_tool or kit.side_bare():
            return pendings
        chosen = self._picks_to_open(pendings, open_tool, id_param)[: kit.safe_side_cap()]
        if len(chosen) < 2:
            return pendings
        self.owner.relay.trace.append({"kind": "joined", "ids": [lid_ for _, lid_ in chosen]})
        return self._merged_call(pendings, chosen, compare_tool, side_param)

    def run_calls(self, pendings: list[dict[str, Any]]) -> list[tuple[str, str]]:
        pendings = self.merge_reads(pendings)
        return [self._one_call(call_spec, len(pendings)) for call_spec in pendings]

    def finish(self, retry_ask: bool = False) -> str:
        if self.owner.done or self.owner.relay.credit_out or self.owner.relay.steps_left() <= 0:
            return ""
        checkout_tool, decline_tool = self.owner.brief.roles.get("buy"), self.owner.brief.roles.get("refuse")
        view_fresh = self.owner.till.cart_shown is not None and self.owner.till.cart_view_step >= self.owner.till.cart_step
        has_lines = bool(self.owner.reviewer.cart_ids()) if view_fresh else bool(self.owner.till.lines)
        if checkout_tool and not has_lines and self.owner.relay.steps_left() >= 2:
            # no line is known to be held (the last one was taken out): the closest pick is bought before a blind checkout
            self.owner.reviewer.closest_pick()
            if self.owner.reviewer.fallback_ids and self._fallback_buy_now(checkout_tool):
                return ""
        held = self.owner.reviewer.cart_ids() or list(self.owner.till.lines)
        if checkout_tool and held and all(self.owner.ledger.gone(lid_) for lid_ in held) and self.owner.relay.steps_left() >= 3:
            # every held line is sold out: ordering it again is refused or wrong, the closest in-stock pick is bought instead
            self.owner.reviewer.closest_pick()
            if self.owner.reviewer.fallback_ids and self._fallback_buy_now(checkout_tool):
                return ""
        if checkout_tool and has_lines and self._over_budget():
            # the budget fell below the cart (often a note at the checkout): never order it as it is
            return self._within_budget_or_end(checkout_tool)
        if checkout_tool and has_lines and self._cart_short() and self.owner.relay.steps_left() >= 3 and self.owner.relay.time_turns() >= 3:
            # a line per requested item is missing: an order without it fails, so the closest listing the request allows fills it
            self.owner.reviewer.closest_pick()
            if self.owner.till.buy_confirmed() or self.owner.done:
                return ""
        if checkout_tool and has_lines:
            self._fit_swap()
        if checkout_tool and (has_lines or (self.owner.till.cart_touched and not view_fresh and self.owner.relay.steps_left() >= 2)):
            self.owner.scout.open_attrs(self.owner.reviewer.cart_ids() or list(self.owner.till.lines), 1, by_steps=True)
            turned_down = self._final_checkout(checkout_tool, retry_ask)
            if turned_down is not None:
                return turned_down
        self.owner.reviewer.closest_pick()
        if self.owner.till.buy_confirmed() or self.owner.done:
            return ""
        if checkout_tool and self._fallback_buy_now(checkout_tool):
            return ""
        if checkout_tool and self._reviewed_buy_now(checkout_tool):
            return ""
        if decline_tool:
            return self._final_decline(decline_tool)
        return ""

    def _cart_short(self) -> bool:
        """The cart holds lines of fewer kinds than the request has items (by the listings' category labels)."""
        brief, till = self.owner.brief, self.owner.till
        cat_key = self.owner.ledger.label_key()
        if len(brief.wants) < 2 or not cat_key:
            return False
        held = [lid_ for lid_ in (self.owner.reviewer.cart_ids() or list(till.lines)) if not self.owner.ledger.gone(lid_)]
        return 0 < len({till._kind_of(lid_, cat_key) for lid_ in held}) < len(brief.wants)

    def _reviewed_buy_now(self, checkout_tool: str) -> bool:
        """The review named an in-stock fitting listing for every item but the episode is ending: Kyo carts those picks
        and checks out rather than end without a purchase."""
        reviewer, till, brief, ledger = self.owner.reviewer, self.owner.till, self.owner.brief, self.owner.ledger
        cart_tool = brief.roles.get("cart")
        picks: dict[str, int] = {}
        for item_ix, lids in reviewer.review_choice.items():
            pos = _as_int(item_ix)
            live = [lid_ for lid_ in lids if not ledger.gone(lid_)]
            if live and pos and 1 <= pos <= len(brief.wants):
                picks[live[0]] = brief.wants[pos - 1].qty
        if not cart_tool or len(picks) < len(brief.wants):
            return False
        to_drop = [lid_ for lid_ in till.lines if lid_ not in picks]
        to_add = {lid_: n_ for lid_, n_ in picks.items() if lid_ not in till.lines}
        if self.owner.relay.steps_left() < len(to_add) + len(to_drop) + 1:
            return False
        if to_add or to_drop:
            pendings = till.cart_calls(to_add, to_drop)
            if not pendings or till._set_cart(cart_tool, pendings) is None or self.owner.done:
                return False
        self.owner.relay.trace.append({"kind": "reviewed_buy", "lines": picks})
        self._final_checkout(checkout_tool, False)
        return True

    def _fallback_buy_now(self, checkout_tool: str, drop_others: bool = False) -> bool:
        """The last resort when the request wants the closest listing ordered: Kyo carts its pick and checks out itself."""
        reviewer, till, brief = self.owner.reviewer, self.owner.till, self.owner.brief
        cart_tool = brief.roles.get("cart")
        picks: dict[str, int] = {}
        for item_ix, lids in reviewer.review_choice.items():
            pos = _as_int(item_ix)
            if lids and lids[0] in reviewer.fallback_ids and not self.owner.ledger.gone(lids[0]) and pos and 1 <= pos <= len(brief.wants):
                picks[lids[0]] = brief.wants[pos - 1].qty
        sold_held = [lid_ for lid_ in till.lines if (drop_others or self.owner.ledger.gone(lid_)) and lid_ not in picks]
        if not picks or not cart_tool or self.owner.relay.steps_left() < len(picks) + len(sold_held) + 1:
            return False
        pendings = till.cart_calls(picks, sold_held)
        if not pendings or till._set_cart(cart_tool, pendings) is None or self.owner.done:
            return False
        self.owner.relay.trace.append({"kind": "fallback_buy", "lines": picks})
        self._fit_swap()
        self._final_checkout(checkout_tool, False)
        return True

    def _one_call(self, call_spec: dict[str, Any], turn_size: int) -> tuple[str, str]:
        allowed = {t_["function"]["name"] for t_ in self.offered_tools()}
        call_id = call_spec["id"]
        if call_spec.get("joined"):
            return call_id, 'Already covered by the side-by-side result above: Kyo opened these listings together in one step.'
        fn = call_spec.get("function") or {}
        pending = _Pending(call_id, fn.get("name"), fn.get("arguments"))
        refusal_text = self._refuse_call(pending, allowed)
        if refusal_text is None:
            refusal_text = self._wasteful(pending, turn_size)
        if refusal_text is None:
            refusal_text = self._final_holds(pending)
        if refusal_text is not None:
            return call_id, refusal_text
        return call_id, self._execute(pending)

    def _refuse_call(self, pending: _Pending, allowed: set[str]) -> str | None:
        """The checks that need no state but the call itself: a text when the call is not run."""
        refusal_note = self._tool_problem(pending.name, allowed)
        if refusal_note:
            return refusal_note
        call_args, refusal_note = self._args_of(pending)
        if refusal_note:
            return refusal_note
        pending.args = call_args
        return None

    def _wasteful(self, pending: _Pending, turn_size: int) -> str | None:
        """Calls that would waste a step: sold-out pages, empty removals, repeats and views with nothing new."""
        self._side_trim(pending)
        pending.id_arg = self.owner.kit.id_param_of(pending.name)
        for guard in (self._sold_page, self._nothing_to_remove, self._zero_add):
            refusal_note = guard(pending)
            if refusal_note:
                return refusal_note
        pending.is_final = self.owner.till.is_checkout(pending.name, pending.args) or self.owner.till.is_decline(pending.name, pending.args)
        if pending.is_final and turn_size > 1:
            return 'Not run (no step used): send the finishing call on its own, after you have read your other results.'
        cart_short = pending.name == self.owner.brief.roles.get("cart") and len(self.owner.till.lines) < len(self.owner.brief.wants) and self.owner.relay.steps_left() >= 2
        if not pending.is_final and not cart_short and self.owner.relay.left() <= self.owner.relay.kept_steps() and (self.owner.brief.roles.get("buy") or self.owner.brief.roles.get("refuse")):
            return "Not run (no step used): the last steps are saved for checking out (with the shopper's approval of the basket when that is due) or ending without a purchase."
        pending.mark = pending.name + _dumps(pending.args)
        for guard in (self._same_call, self._memo_reply, self._seen_view, self._seen_side):
            refusal_note = guard(pending)
            if refusal_note is not None:
                return refusal_note
        return None

    def reading_tools(self) -> set[str]:
        return {self.owner.brief.roles.get(r_) for r_ in ("search", "open", "offer", "compare")} - {None}

    def _final_holds(self, pending: _Pending) -> str | None:
        """What stands between a finishing call and the store: the checks before a checkout or an ending, the late and
        early decline holds, the reviewed cart Kyo completes itself, and the steps the request asks for."""
        hold_text = self._finish_gate(pending)
        if hold_text is None:
            hold_text = self._decline_gate(pending)
        if hold_text is None and pending.is_final and self.owner.relay.left() >= 2:
            hold_text = self._due_steps(pending)
        return hold_text

    def _short_cart(self) -> str | None:
        """A checkout missing some requested item's line while the request wants the closest listing ordered: Kyo completes the cart, or says which items are missing."""
        brief, till = self.owner.brief, self.owner.till
        if len(brief.wants) < 2 or self.short_cart_holds >= 2 or self.owner.relay.steps_left() < 3:
            return None
        cat_key = self.owner.ledger.label_key()
        held = self.owner.reviewer.cart_ids() or list(till.lines)
        kinds = {till._kind_of(lid_, cat_key) for lid_ in held}
        if not cat_key or len(kinds) >= len(brief.wants):
            return None
        self.short_cart_holds += 1
        if brief.parsed.get("fallback_buy") is True and self.short_cart_holds == 1:
            self.owner.reviewer.closest_pick()
            done = till.buy_confirmed()
            if done:
                return done + (" [episode finished]" if self.owner.done else "")
        missing = [w_.name for w_ in brief.wants if not any(set(_token_set(w_.name)) & set(_token_set(k_)) for k_ in kinds)]
        by_kind: dict[str, list[str]] = {}
        for lid_ in held:
            by_kind.setdefault(till._kind_of(lid_, cat_key), []).append(lid_)
        doubled = "; ".join(f"{', '.join(v_)} are all {k_ or 'one kind'}" for k_, v_ in by_kind.items() if len(v_) > 1)
        return ("Not run yet (no step used). The cart holds lines of only " + str(len(kinds)) + " kinds for the " + str(len(brief.wants)) + " requested items"
                + (" (no line yet for: " + ", ".join(missing) + ")" if missing else "") + (" (" + doubled + ": each requested item needs a line of its own kind)" if doubled else "")
                + ". Put the best listing you have read of each missing item's kind in the cart at its requested quantity"
                + (" (the closest one when none fits fully, as the request asks)" if brief.parsed.get("fallback_buy") is True else "") + ", then check out.")

    def _checkout_holds(self, pending: _Pending) -> str | None:
        short_text = self._short_cart()
        if short_text:
            return short_text
        if not self.owner.till.lines and not self.owner.reviewer.cart_ids():
            self.owner.reviewer.closest_pick()
            done = self.owner.till.buy_confirmed()
            if done:
                return done + (" [episode finished]" if self.owner.done else "")
            fallback_text = self._fallback_hold()
            if fallback_text:
                return fallback_text
            return "Not run yet (no step used). The cart is empty: put each requested item's chosen listing in the cart at the requested quantity first, then check out."
        wider_text = self.owner.scout.widen_search()
        if wider_text:
            return wider_text + "\nCart now: " + self.owner.reviewer.cart_text()
        reread_text = self.owner.reviewer.refresh_cart()
        held = self._cart_holds()
        if held is not None:
            return held
        return self._check_holds(reread_text)

    def _look_at_cart(self) -> str:
        faults: list[str] = []
        cart_view_args = _fit_schema({}, self.owner.schemas.get(self.owner.brief.roles["cart_view"]), "args", faults)
        cart_obs = self.owner.relay.env_call(self.owner.brief.roles["cart_view"], cart_view_args) if not faults else None
        if cart_obs is None:
            return ""
        self.owner.till.cart_shown = cart_obs.get("observation")
        self.owner.till.cart_view_step = self.owner.steps_taken
        seen_change = self.owner.scout.absorb_result(self.owner.brief.roles["cart_view"], self.owner.till.cart_shown)
        if seen_change:
            self.owner.bulletin.queued_notes = []
        return seen_change

    def _decline_stops(self) -> str | None:
        if self.owner.relay.left() <= 3:
            return self._short_fit_stop()
        review_text = self.owner.reviewer.review_decline()
        if review_text:
            return "Not run yet (no step used). " + review_text
        if self.owner.reviewer.suggested and self.owner.relay.left() > 4 and self.decline_stops == 0:
            self.decline_stops = 1
            self.owner.relay.trace.append({"kind": "refuse_blocked", "leads": list(self.owner.reviewer.suggested)})
            return "Not run yet (no step used). The review found listings that seem to fit: " + ", ".join(self.owner.reviewer.suggested) + ". Open them if needed, cart the best fitting one (keeping to the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."
        if self.owner.reviewer.suggested and self.owner.relay.left() > 4 and self.decline_stops == 1:
            self.decline_stops = 2
            top_pick = self.owner.reviewer.suggested[0]
            self.owner.reviewer.check_cart([top_pick], self.owner.relay.hard_spend())
            if self.owner.reviewer.verdicts.get(self.owner.reviewer.verdict_key([top_pick])) and top_pick in self.owner.reviewer.suggested:
                self.owner.reviewer.confirmed = top_pick
                self.owner.relay.trace.append({"kind": "refuse_blocked", "confirmed": top_pick})
                return f"Not run yet (no step used). An independent check of {top_pick}'s records says it meets every requirement. Cart it at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."
        return None

    def _short_fit_stop(self) -> str | None:
        """Few steps left and the review named an opened listing that fits: hold the ending once so it gets bought."""
        ready = [lid_ for lid_ in self.owner.reviewer.suggested if lid_ in self.owner.ledger.opened and not self.owner.ledger.gone(lid_)]
        if not ready or self.decline_stops >= 2 or self.owner.relay.left() < len(self.owner.brief.wants) + 1:
            return None
        self.decline_stops = 2
        self.owner.relay.trace.append({"kind": "refuse_blocked", "short": ready})
        return f"Not run yet (no step used). The review found that the opened page of {', '.join(ready[:3])} meets every requirement. Cart the best of them at the requested quantity (keeping to the pick rule) and check out with the steps left; end without a purchase only if its record clearly breaks a requirement."

    def _late_stop(self) -> str | None:
        lines = [lid_ for lid_ in self.owner.till.top_cart if not self.owner.ledger.gone(lid_)]
        view_fresh = self.owner.till.cart_shown is not None and self.owner.till.cart_view_step >= self.owner.till.cart_step
        missing_ids = [lid_ for lid_ in lines if lid_ not in self.owner.reviewer.cart_ids() or (lid_ in self.owner.till.dropped and not view_fresh)]
        if len(lines) < len(self.owner.brief.wants) or len(missing_ids) + 1 > self.owner.relay.left():
            return None
        self.late_stop_done = True
        self.owner.relay.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing_ids})
        redo_text = f"Put {', '.join(missing_ids)} back in the cart at the requested quantities, then check out." if missing_ids else 'Check out the cart as it stands.'
        return ('Not run yet (no step used). Ending without a purchase scores nothing when the store has a fitting listing, and running short of steps does not show that it has none. '
                + redo_text + ' End without buying only if your searches showed that some requested item has no listing meeting its requirements; then send the ending call again.')

    def _execute(self, pending: _Pending) -> str:
        tool_name, call_args, side_param = pending.name, pending.args, pending.side_key
        env_reply = self.owner.relay.env_call(tool_name, call_args)
        if env_reply is None:
            return 'The call did not go through (network or server trouble). You can send it again.'
        observed = env_reply.get("observation")
        if side_param and isinstance(call_args.get(side_param), list) and len(call_args[side_param]) > 1:
            self._side_cap_seen(observed, call_args[side_param], pending)
        self.recent_marks = (self.recent_marks + [pending.mark])[-3:]
        self._learn(tool_name, call_args, observed)
        reader_tools = self.reading_tools()
        non_search = {tool for role_name, tool in self.owner.brief.roles.items() if role_name != "search"}
        paged_text = self.owner.scout.more_pages(tool_name, call_args, observed) if tool_name not in non_search and not pending.is_final else ""
        self._views_done(tool_name, observed)
        self._cart_after(tool_name, call_args, observed, env_reply)
        return self._call_text(pending, env_reply, observed, paged_text, non_search, reader_tools)

    def _side_cap_seen(self, observed: Any, sent: list[Any], pending: _Pending) -> None:
        shown_n = _dumps(observed)
        if not any(str(lid_) in shown_n for lid_ in sent):
            self.owner.kit.side_bad_n = min(self.owner.kit.side_bad_n or 999, len(sent))
            pending.extra += f"\n[Kyo] The side-by-side view showed nothing for {len(sent)} listings: send them again, at most {self.owner.kit.side_cap()} per call: {', '.join(map(str, sent))}."
        else:
            self.owner.kit.side_good_n = max(self.owner.kit.side_good_n, len(sent))

    def _views_done(self, tool_name: str, observed: Any) -> None:
        """A single-listing view: whether it shows attributes, and which tool is then the attributes or the offer view."""
        busy_tools = (self.owner.brief.roles.get("cart"), self.owner.brief.roles.get("buy"), self.owner.brief.roles.get("refuse"), self.owner.brief.roles.get("ask"))
        if not self.owner.kit.id_param_of(tool_name) or tool_name in busy_tools:
            return
        listing_seen = self.owner.ledger.record_of(observed) if isinstance(observed, dict) else None
        if not listing_seen:
            return
        self.owner.kit.shows_attrs[tool_name] = self.owner.kit.shows_attrs.get(tool_name, False) or bool(_attrs_of(observed))
        if tool_name == self.owner.brief.roles.get("open") and not self.owner.kit.shows_attrs[tool_name]:
            other_tool = next((t_ for t_ in self.owner.schemas if t_ != tool_name and self.owner.kit.id_param_of(t_) and self.owner.kit.shows_attrs.get(t_) is not False
                          and t_ not in busy_tools + (self.owner.brief.roles.get("search"),)), None)
            if other_tool:
                self.owner.brief.roles["open"] = other_tool
        if not self.owner.kit.shows_attrs[tool_name] and tool_name != self.owner.brief.roles.get("open") and not self.owner.brief.roles.get("offer"):
            self.owner.brief.roles["offer"] = tool_name
        if tool_name == self.owner.kit.offer_tool():
            self.owner.ledger.offer_step[listing_seen] = self.owner.steps_taken

    def _cart_moved(self, tool_name: str, call_args: dict[str, Any], observed: Any) -> None:
        self.owner.till.note_cart(tool_name, call_args)
        self.owner.till.cart_obs = _dumps(observed)
        self.recent_marks = []
        unit_values = [_as_int(v_) for v_ in call_args.values() if not isinstance(v_, str) and _as_int(v_) is not None]
        for datum in call_args.values():
            if unit_values and isinstance(datum, str) and datum in self.owner.ledger.rows:
                if unit_values[0] <= 0 and not _mentions(observed, datum):
                    self.owner.till.dropped.add(datum)
                elif unit_values[0] > 0:
                    self.owner.till.dropped.discard(datum)
        lines = [lid_ for lid_ in self.owner.reviewer.cart_ids() if not self.owner.ledger.gone(lid_) and lid_ not in self.owner.till.dropped]
        if len(lines) >= len(self.owner.till.top_cart):
            self.owner.till.top_cart = lines

    def _call_text(self, pending: _Pending, env_reply: dict[str, Any], observed: Any, paged_text: str, non_search: set[Any], reader_tools: set[str]) -> str:
        tool_name = pending.name
        text = self.owner.relay.render(observed) + paged_text + pending.extra + self.sweep_note
        self.sweep_note = ""
        if tool_name not in non_search and not pending.is_final:
            text += self.owner.scout.cheapest_rows([observed] + self.owner.scout.auto_pages)
        plain_text = text
        text += self._appended(pending, env_reply, observed)
        if tool_name in reader_tools:
            self.memo[pending.mark] = (self.owner.epoch, self.owner.steps_taken, plain_text)
        if env_reply.get("finished"):
            text += " [episode finished]"
        return text

    def _final_checkout(self, checkout_tool: str, retry_ask: bool) -> str | None:
        """None: go on to the ending without a purchase; a text: stop here (empty, or the refusal for the agent)."""
        faults: list[str] = []
        call_args = _fit_schema(dict(self.owner.brief.parsed.get("buy_args") or {}), self.owner.schemas.get(checkout_tool), "args", faults)
        if faults:
            return None
        if self.owner.relay.steps_left() >= 2:
            self.owner.clerk.run_gate("order")
        if self.owner.done or self.owner.relay.steps_left() <= 0:
            return ""
        env_reply = self.owner.relay.env_call(checkout_tool, call_args)
        self.owner.relay.trace.append({"kind": "forced_buy", "reply": _clipped(env_reply, 200)})
        if self.owner.done or self.owner.relay.steps_left() <= 0:
            return ""
        if env_reply is not None and retry_ask:
            self.owner.till.checkout_count += 1
            self.owner.till.refusal_step = self.owner.steps_taken
            self.owner.scout.absorb_result(checkout_tool, env_reply.get("observation"))
            return "the checkout was refused: " + self.owner.relay.render(env_reply.get("observation"))
        return None

    def _final_decline(self, decline_tool: str) -> str:
        faults: list[str] = []
        call_args = _fit_schema(self.owner.till.bare_decline(decline_tool, dict(self.owner.brief.parsed.get("refuse_args") or {})), self.owner.schemas.get(decline_tool), "args", faults)
        if faults:
            return ""
        if self.owner.relay.steps_left() >= 2:
            self.owner.clerk.run_gate("end")
        if self.owner.done or self.owner.relay.steps_left() <= 0:
            return ""
        env_reply = self.owner.relay.env_call(decline_tool, call_args)
        self.owner.relay.trace.append({"kind": "forced_refuse", "reply": _clipped(env_reply, 200)})
        if env_reply is not None and not self.owner.done:
            self.owner.scout.absorb_result(decline_tool, env_reply.get("observation"))
            return "ending without a purchase was refused: " + self.owner.relay.render(env_reply.get("observation"))
        return ""

    def _side_trim(self, pending: _Pending) -> None:
        """A side-by-side call longer than the store has shown it takes is cut to that length."""
        pending.side_key = self.owner.kit.side_param() if pending.name == self.owner.brief.roles.get("compare") else None
        side_param = pending.side_key
        if side_param and isinstance(pending.args.get(side_param), list) and len(pending.args[side_param]) > self.owner.kit.side_cap() and self.owner.kit.side_bad_n:
            room_size = self.owner.kit.side_cap()
            pending.extra = f"\n[Kyo] Sent the first {room_size} listings only: this store shows at most {room_size} side by side. Not shown yet: {', '.join(map(str, pending.args[side_param][room_size:]))}."
            pending.args = {**pending.args, side_param: pending.args[side_param][:room_size]}

    def _sold_page(self, pending: _Pending) -> str | None:
        id_param, call_args = pending.id_arg, pending.args
        if id_param and pending.name in (self.owner.brief.roles.get("open"), self.owner.brief.roles.get("offer")) and str(call_args.get(id_param)) in self.owner.ledger.sold_step:
            return f"Not run (no step used): {call_args[id_param]} sold out at step {self.owner.ledger.sold_step[str(call_args[id_param])]} and will not return. Replace its cart line with another fitting listing."
        return None

    def _zero_add(self, pending: _Pending) -> str | None:
        """A cart call that would put 0 units of a listing not in the cart: it adds nothing, so it is not run."""
        till, tool_name = self.owner.till, pending.name
        if tool_name != self.owner.brief.roles.get("cart"):
            return None
        schema = self.owner.schemas.get(tool_name) or {}
        id_param, qty, mode_param = till._line_call(tool_name, schema, schema.get("properties") or {})
        lid = str(pending.args.get(id_param)) if id_param else ""
        if not qty or _as_int(pending.args.get(qty)) != 0 or lid in till.lines or lid not in self.owner.ledger.rows:
            return None
        if mode_param and pending.args.get(mode_param) not in (None, ((schema.get("properties") or {}).get(mode_param) or {}).get("enum", [None])[0]):
            return None
        return f"Not run (no step used): {qty} is the number of units to put in the cart, and 0 puts none. Send {qty} as the requested quantity (1 unless the request says more)."

    def _nothing_to_remove(self, pending: _Pending) -> str | None:
        call_args = pending.args
        if pending.name != self.owner.brief.roles.get("cart") or not any((_as_int(v_) or 0) < 0 for v_ in call_args.values() if not isinstance(v_, str)):
            return None
        till = self.owner.till
        cart_obs = _dumps(till.cart_shown) if till.cart_view_step >= till.cart_step else ""
        not_held = [v_ for v_ in call_args.values() if isinstance(v_, str) and v_ in self.owner.ledger.rows and v_ not in till.lines and v_ not in cart_obs and v_ not in till.cart_obs]
        if not_held:
            return f"Not run (no step used): {not_held[0]} is not in the cart, so there is nothing to remove. Cart now: " + self.owner.reviewer.cart_text()
        return None

    def _same_call(self, pending: _Pending) -> str | None:
        if not pending.is_final and pending.name != self.owner.brief.roles.get("cart") and pending.mark in self.recent_marks:
            return 'Not run (no step used): you just made this exact call and nothing has changed; its result is above.'
        return None

    def _memo_reply(self, pending: _Pending) -> str | None:
        remembered = self.memo.get(pending.mark)
        if pending.name in self.reading_tools() and remembered and remembered[0] == self.owner.epoch:
            return f"Not run (no step used): the same call at step {remembered[1]} gave this result, and nothing has changed since:\n" + remembered[2][:3000]
        return None

    def _seen_view(self, pending: _Pending) -> str | None:
        id_param, call_args, tool_name = pending.id_arg, pending.args, pending.name
        if id_param and tool_name in self.reading_tools() and tool_name != self.owner.brief.roles.get("compare") and self.owner.ledger.all_seen(tool_name, str(call_args.get(id_param))):
            return f"Not run (no step used): {call_args[id_param]} has shown all this view offers since the last change. Its record: " + self.owner.reviewer.record_line(str(call_args[id_param]))
        return None

    def _seen_side(self, pending: _Pending) -> str | None:
        """A side-by-side call over listings shown in full already: refused when all are, trimmed when some are."""
        side_param, call_args, tool_name = pending.side_key, pending.args, pending.name
        if not side_param or not isinstance(call_args.get(side_param), list) or not call_args[side_param]:
            return None
        reserve = [lid_ for lid_ in call_args[side_param] if not self.owner.ledger.all_seen(tool_name, str(lid_))]
        if not reserve:
            return ("Not run (no step used): these listings have shown all this view offers since the last change. Their records:\n"
                    + "\n".join(f"{lid_}: {self.owner.reviewer.record_line(str(lid_))}" for lid_ in call_args[side_param])[:4000])
        if len(reserve) < len(call_args[side_param]):
            pending.extra += f"\n[Kyo] Left out {', '.join(str(lid_) for lid_ in call_args[side_param] if lid_ not in reserve)}: already shown in full, nothing changed since."
            pending.args = {**call_args, side_param: reserve}
            pending.mark = tool_name + _dumps(pending.args)
        return None

    def _picks_to_open(self, pendings: list[dict[str, Any]], open_tool: str, id_param: str) -> list[tuple[int, str]]:
        ledger = self.owner.ledger
        chosen: list[tuple[int, str]] = []
        for pos, call_spec in enumerate(pendings):
            fn = call_spec.get("function") or {}
            call_args = _first_object(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            lid = str((call_args or {}).get(id_param) or "").strip()
            if fn.get("name") == open_tool and call_args and set(call_args) == {id_param} and lid in ledger.rows and lid not in ledger.sold_step:
                chosen.append((pos, lid))
        return chosen

    @staticmethod
    def _merged_call(pendings: list[dict[str, Any]], chosen: list[tuple[int, str]], compare_tool: str, side_param: str) -> list[dict[str, Any]]:
        top_pick = chosen[0][0]
        joined = {lid_ for lid_, _ in chosen}
        result_rows = []
        for pos, call_spec in enumerate(pendings):
            if pos == top_pick:
                result_rows.append({"id": call_spec["id"], "type": "function", "function": {"name": compare_tool, "arguments": _dumps({side_param: [lid_ for _, lid_ in chosen]})}})
            elif pos in joined:
                result_rows.append({"id": call_spec["id"], "joined": True})
            else:
                result_rows.append(call_spec)
        return result_rows

    def _learn(self, tool_name: str, call_args: dict[str, Any], observed: Any) -> None:
        """Id parameters, the records the result carries, and what the side-by-side or reading views show."""
        ledger, roles = self.owner.ledger, self.owner.brief.roles

        def all_ids(v_: Any) -> bool:
            listed = v_ if isinstance(v_, list) else [v_]
            return bool(listed) and all(isinstance(x_, str) and x_ in ledger.rows for x_ in listed)

        ledger.id_args.update((tool_name, k_) for k_, v_ in call_args.items() if all_ids(v_))
        ledger.absorb(observed)
        if tool_name == roles.get("compare"):
            rows = ledger.rows_in(observed)
            if rows:
                views = self.owner.kit.shows_attrs
                views[tool_name] = views.get(tool_name, False) or any(_attrs_of(r_) for _, r_ in rows)
        if tool_name in self.reading_tools() - {roles.get("search")}:
            ledger.note_view(tool_name, observed)

    def _cart_after(self, tool_name: str, call_args: dict[str, Any], observed: Any, env_reply: dict[str, Any]) -> None:
        roles, till = self.owner.brief.roles, self.owner.till
        if tool_name == roles.get("cart"):
            self._cart_moved(tool_name, call_args, observed)
            views = self.owner.scout.cap_sweep()
            cheaper = self.owner.scout.cheapest_rows(views) if views else ""
            if views:
                self.sweep_note = ("\n[Kyo listed each cart line's category below that line's price (one step each)]" + (cheaper or " nothing cheaper is buyable there")
                                   + (". Open the cheaper rows that might fit and swap in the cheapest that fits every requirement." if cheaper else "."))
        if tool_name == roles.get("cart_view"):
            till.cart_shown = observed
            till.cart_view_step = self.owner.steps_taken
            till.learn_from_view(observed)
        if till.is_checkout(tool_name, call_args):
            till.checkout_count += 1
            if not env_reply.get("finished"):
                till.refusal_step = self.owner.steps_taken

    def _appended(self, pending: _Pending, env_reply: dict[str, Any], observed: Any) -> str:
        """What Kyo appends to a result: evidence, a zero-count warning, a notice read, a refusal's re-read, a stand-in hint."""
        added = self._evidence_text(observed) + self._zero_note(pending, observed) + self._note_text(observed) + self._refusal_text(pending, env_reply, observed)
        swap_hint = self.owner.reviewer.swap_note()
        if swap_hint:
            added += "\n" + swap_hint
        bulletin = self.owner.bulletin
        if bulletin.queued_notes:
            added += "\n[notes seen on Kyo's own calls] " + " | ".join(bulletin.queued_notes)
            bulletin.queued_notes = []
        return added

    def _over_budget(self) -> bool:
        """The cart's known total is above the whole-order budget as it stands now (a total more than twenty times the
        budget is taken for a unit mismatch, not a breach)."""
        brief = self.owner.brief
        total = self.owner.till.cart_total()
        if total is None or not isinstance(brief.budget, (int, float)) or brief.budget <= 0 or brief.parsed.get("budget_scope") == "each":
            return False
        return brief.budget + 1e-6 < total <= brief.budget * 20

    def _within_budget_or_end(self, checkout_tool: str) -> str:
        """An over-budget cart is never checked out: the closest pick within the budget is bought instead, or the
        episode ends without a purchase."""
        self.owner.reviewer.closest_pick()
        if self.owner.reviewer.fallback_ids and self._fallback_buy_now(checkout_tool, drop_others=True):
            return "Not run: the cart was over the budget, so Kyo put the closest listing within it in the cart and checked out instead." + (" [episode finished]" if self.owner.done else "")
        decline_tool = self.owner.brief.roles.get("refuse")
        if decline_tool and not self.owner.done:
            self.owner.relay.trace.append({"kind": "over_budget_end", "total": self.owner.till.cart_total(), "budget": self.owner.brief.budget})
            turned_down = self._final_decline(decline_tool)
            return "Not run: the cart is over the budget and nothing within it was found, so Kyo ended without a purchase." + (" " + turned_down if turned_down else "") + (" [episode finished]" if self.owner.done else "")
        return ""

    def _budget_hold(self, pending: _Pending) -> str | None:
        if not self.owner.till.is_checkout(pending.name, pending.args) or not self._over_budget():
            return None
        if not self.over_budget_told and self.owner.relay.left() > 3:
            self.over_budget_told = True
            return (f"Not run (no step used): the cart totals {self.owner.till.cart_total():g}, more than the budget of {self.owner.brief.budget:g} as it stands now."
                    " Replace lines with cheaper listings that meet every requirement, or end without a purchase if none is within the budget.")
        return self._within_budget_or_end(pending.name) or None

    def _finish_gate(self, pending: _Pending) -> str | None:
        if not pending.is_final:
            return None
        budget_text = self._budget_hold(pending)
        if budget_text:
            return budget_text
        if self.owner.till.is_checkout(pending.name, pending.args):
            fit_text = self._fit_gate()
            if fit_text:
                return fit_text
        if self.owner.relay.left() <= 2:
            return self._late_empty_checkout(pending)
        return self._checkout_holds(pending) if self.owner.till.is_checkout(pending.name, pending.args) else self._decline_stops()

    def _late_empty_checkout(self, pending: _Pending) -> str | None:
        """A checkout with an empty cart in the last steps fails and wastes a step: Kyo buys the closest listing itself
        when two steps allow it, otherwise the call is held with what to do instead."""
        till = self.owner.till
        if not till.is_checkout(pending.name, pending.args) or till.lines or self.owner.reviewer.cart_ids():
            return None
        self.owner.reviewer.closest_pick()
        if self.owner.reviewer.fallback_ids and self._fallback_buy_now(pending.name):
            return "Not run: the cart was empty, so Kyo put the closest listing in the cart and checked out instead." + (" [episode finished]" if self.owner.done else "")
        return "Not run (no step used): the cart is empty, so a checkout would fail. Put the best fitting listing you have read in the cart first, then check out."

    def _decline_gate(self, pending: _Pending) -> str | None:
        """An ending without a purchase: held late or early, or replaced by a reviewed cart Kyo buys itself."""
        till, relay = self.owner.till, self.owner.relay
        if not till.is_decline(pending.name, pending.args):
            return None
        if relay.left() <= 5 and not self.late_stop_done:
            hold_text = self._late_stop()
            if hold_text is not None:
                return hold_text
        if not self.owner.reviewer.suggested and not self.decline_nudged and relay.left() > max(6, self.owner.max_steps // 4):
            self.decline_nudged = True
            return f"Not run yet (no step used). {relay.left()} steps are still left, so before ending search in ways you have not tried yet: browse each requested item's category (with a price limit when the search takes one) and page through it with the token or offset the store returns, going on from the last page you read, try one-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        self.owner.reviewer.closest_pick()
        done = till.buy_confirmed()
        if done:
            return done + (" [episode finished]" if self.owner.done else "")
        fallback_text = self._fallback_hold()
        if fallback_text:
            return fallback_text
        checkout_tool = self.owner.brief.roles.get("buy")
        if self.owner.reviewer.fallback_ids and checkout_tool and self._fallback_buy_now(checkout_tool):
            return "Not run: as the request asks when nothing fits, Kyo put the closest listing in the cart and checked out instead." + (" [episode finished]" if self.owner.done else "")
        if checkout_tool and self._reviewed_buy_now(checkout_tool):
            return "Not run: the review found a fitting listing for every item, so Kyo put them in the cart and checked out instead." + (" [episode finished]" if self.owner.done else "")
        pending.args = till.bare_decline(pending.name, pending.args)
        return None

    def _fallback_hold(self) -> str | None:
        """The request wants the closest listing ordered when nothing fits, and Kyo could not buy it itself: the agent is told to, once."""
        reviewer = self.owner.reviewer
        ready = [lid_ for lid_ in reviewer.fallback_ids if not self.owner.ledger.gone(lid_)]
        if not ready and reviewer.fallback_done and not self.fallback_searched and self.owner.relay.steps_left() >= 5 and self.owner.relay.left() >= 3:
            self.fallback_searched = True
            return ("Not run (no step used). The request wants the closest listing ordered when nothing fits, but no opened listing of the requested kind is both in stock and within the budget."
                    " Search the item's category with a price limit at the budget (when the search takes one), open the cheapest in-stock rows, then send the ending call again.")
        if not ready or self.fallback_held or self.owner.relay.steps_left() < 2:
            return None
        self.fallback_held = True
        lines = "; ".join(f"item {k_}: {v_[0]}" for k_, v_ in reviewer.review_choice.items() if v_ and v_[0] in ready)
        return ("Not run (no step used). The request says what to do when nothing fits: " + str(self.owner.brief.parsed.get("fallback") or "order the closest listing")
                + ". The closest listing by Kyo's review is " + (lines or ", ".join(ready)) + ". Put it in the cart at the requested quantity and check out; do not end without a purchase.")

    def _due_steps(self, pending: _Pending) -> str | None:
        go_on, spoken = self.owner.clerk.run_gate("order" if self.owner.till.is_checkout(pending.name, pending.args) else "end")
        if not go_on:
            return "Not run yet. " + spoken + "\nCart now: " + self.owner.reviewer.cart_text()
        if spoken:
            pending.extra = "\n[Kyo] " + spoken + pending.extra
        return None

    def _cart_holds(self) -> str | None:
        """A cart view showing a change, or a sold-out line to replace first."""
        till, reviewer = self.owner.till, self.owner.reviewer
        if till.cart_touched and till.cart_view_step < till.cart_step and self.owner.brief.roles.get("cart_view") and self.owner.relay.left() > 3:
            seen_change = self._look_at_cart()
            if seen_change:
                return "Not run yet (no step used). The cart view showed a change: " + seen_change + "\nCart now: " + reviewer.cart_text() + '\nApply the change, fix the cart where it needs it, then check out.'
        swap_hint = reviewer.swap_note()
        hold_text = [lid_ for lid_ in reviewer.cart_ids() if lid_ in self.owner.ledger.sold_step] if till.sold_stops < 2 else []
        if swap_hint or hold_text:
            till.sold_stops += 0 if swap_hint else 1
            return "Not run yet (no step used). " + (swap_hint or f"{', '.join(hold_text)} in the cart has sold out: replace that line first.") + "\nCart now: " + reviewer.cart_text()
        return None

    def _fit_faults(self) -> str:
        return " | ".join(text_ for _, _, text_ in self._fit_report())

    def _fit_report(self, planned: list[str] | None = None) -> list[tuple[str, list[str], str]]:
        """Cart lines whose page names a family of model codes they serve ("AB-45xx") checked against the full codes the
        other cart lines' pages carry with the same letters: a line serving none of them does not fit the cart."""
        ledger = self.owner.ledger
        lines = [lid_ for lid_ in (planned if planned is not None else (self.owner.reviewer.cart_ids() or list(self.owner.till.lines))) if not ledger.gone(lid_)]
        texts = {lid_: _dumps(ledger.opened.get(lid_) or ledger.rows.get(lid_) or {}) for lid_ in ledger.rows}
        faults: list[tuple[str, list[str], str]] = []

        def serves(code: tuple[str, str], fams: set[tuple[str, str]], bar: set[tuple[str, str]]) -> bool:
            return code not in bar and any(code[0] == f_[0] and code[1].startswith(f_[1]) for f_ in fams)

        for acc in lines:
            families = {(m_.group(1), m_.group(2)) for m_ in _FAMILY_RE.finditer(texts.get(acc, ""))}
            if not families:
                continue
            letters = {f_[0] for f_ in families}
            excluded = {(m_.group(1), m_.group(2)) for m_ in _CODE_RE.finditer(texts.get(acc, "")) if m_.group(1) in letters}
            hosts = {h_: {(m_.group(1), m_.group(2)) for m_ in _CODE_RE.finditer(texts.get(h_, "")) if m_.group(1) in letters}
                     for h_ in lines if h_ != acc}
            hosts = {h_: c_ for h_, c_ in hosts.items() if c_}
            if not hosts or any(serves(c_, families, excluded) for codes in hosts.values() for c_ in codes):
                continue
            host_codes = sorted({f"{a_}-{n_}" for codes in hosts.values() for a_, n_ in codes})
            kind_key = ledger.label_key()
            kind = str((ledger.rows.get(acc) or {}).get(kind_key) or "") if kind_key else ""
            better = [lid_ for lid_ in ledger.opened if lid_ not in lines and not ledger.gone(lid_)
                      and (not kind or str((ledger.rows.get(lid_) or {}).get(kind_key) or "") == kind)
                      and any(serves(c_, {(m_.group(1), m_.group(2)) for m_ in _FAMILY_RE.finditer(texts.get(lid_, ""))},
                                     {(m_.group(1), m_.group(2)) for m_ in _CODE_RE.finditer(texts.get(lid_, ""))}) for codes in hosts.values() for c_ in codes)][:4]
            fams = ", ".join(sorted(f"{a_}-{n_}xx" for a_, n_ in families))
            faults.append((acc, better, f"{acc} serves models {fams}, but the cart's other lines carry {', '.join(host_codes)}"
                          + (f"; opened listings of its kind that serve them: {', '.join(better)}" if better else
                             "; search its kind with the code family the other line carries, or switch that other line to one this listing serves")))
        return faults

    def _fit_gate(self) -> str | None:
        """A checkout whose cart breaks the listings' own model codes: Kyo swaps in a fitting opened listing when it knows
        one, else holds the checkout (twice at most) with what does not fit."""
        report = self._fit_report()
        if not report or self.fit_holds >= 2 or self.owner.relay.steps_left() < 3:
            return None
        self.fit_holds += 1
        if any(better for _, better, _ in report):
            self._fit_swap()
            report = self._fit_report()
            if not report:
                return "Not run yet (no step used). Kyo swapped a cart line whose model codes did not fit the others; cart now: " + self.owner.reviewer.cart_text() + ". Check out again."
        self.owner.relay.trace.append({"kind": "fit_hold", "faults": " | ".join(t_ for _, _, t_ in report)[:300]})
        return ("Not run yet (no step used). The listings' own model codes show a cart line that does not fit another: " + " | ".join(t_ for _, _, t_ in report)
                + ". Fix that line, then check out.\nCart now: " + self.owner.reviewer.cart_text())

    def _fit_swap(self) -> None:
        """Before Kyo's own checkout: a cart line that serves none of the other lines' model codes is swapped for an opened
        listing of its kind that does, when one is known and the steps allow."""
        till, cart_tool = self.owner.till, self.owner.brief.roles.get("cart")
        for acc, better, text in self._fit_report():
            if not better or not cart_tool or self.owner.relay.steps_left() < 3 or self.owner.done:
                return
            pick = next((lid_ for lid_ in better if self.owner.ledger.buyable(lid_)), None)
            if not pick:
                continue
            wanted = {lid_: n_ for lid_, n_ in till.lines.items() if lid_ != acc}
            wanted[pick] = till.lines.get(acc, 1)
            pendings = till.cart_calls(wanted, [acc])
            if not pendings or till._set_cart(cart_tool, pendings) is None:
                return
            self.owner.relay.trace.append({"kind": "fit_swap", "out": acc, "in": pick, "why": text[:200]})

    def _check_holds(self, reread_text: str) -> str | None:
        reviewer, scout = self.owner.reviewer, self.owner.scout
        fit_text = self._fit_faults() if self.fit_holds < 2 else ""
        if fit_text and self.owner.relay.left() >= 4:
            self.fit_holds += 1
            self.owner.relay.trace.append({"kind": "fit_hold", "faults": fit_text[:300]})
            return ("Not run yet (no step used). The listings' own model codes show a cart line that does not fit another: " + fit_text
                    + ". Fix that line, then check out.\nCart now: " + reviewer.cart_text())
        chain, hold = reviewer.trace_links()
        if hold:
            return "Not run yet (no step used). " + chain + "\nCart now: " + reviewer.cart_text()
        scout.cap_sweep()
        read_texts = scout.open_attrs(reviewer.cart_ids(), self.owner.relay.kept_steps() + 1, by_steps=True) + scout.cheaper_reads()
        verdict_text = reviewer.check_cart()
        if verdict_text:
            view = ("For the check, Kyo read these listings' attributes (one step per listing):\n" + "\n".join(read_texts) + "\n") if read_texts else ""
            return "Not run yet. " + chain + view + verdict_text + "\nCart now: " + reviewer.cart_text()
        if reread_text and self.owner.relay.cost > self.owner.relay.soft_spend():
            return "Not run yet (no step used). Kyo just read the cart's listings again:\n" + reread_text + '\nCheck out again if every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return None

    def _tool_problem(self, tool_name: Any, allowed: set[str]) -> str | None:
        if self.owner.done:
            return 'Not run: the episode has ended.'
        if tool_name == self.owner.brief.roles.get("ask"):
            return "Not run (no step used): all messages to the shopper go through Kyo's interviewer."
        if tool_name not in self.owner.schemas:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."
        if tool_name not in allowed:
            return "Not run (no step used): not enough steps are left for that; use: " + ", ".join(sorted(allowed)) + "."
        return None

    def _args_of(self, pending: _Pending) -> tuple[dict[str, Any], str | None]:
        """The call's arguments read, bent to the schema and with titles turned into ids; or the reason they cannot be."""
        raw = pending.raw
        if isinstance(raw, dict):
            call_args = raw
        else:
            call_args = _first_object(raw) if (raw or "").strip() else {}
            if call_args is None:
                return {}, 'Not run (no step used): the arguments were not a single JSON object.'
        faults: list[str] = []
        call_args = _fit_schema(call_args, self.owner.schemas[pending.name], "args", faults)
        if faults:
            return call_args, "Not run (no step used): " + "; ".join(faults[:6])
        call_args, wrong_text = self.owner.till.titles_to_ids(pending.name, call_args)
        if wrong_text:
            return call_args, "Not run (no step used): " + wrong_text + "."
        return call_args, None

    def _evidence_text(self, observed: Any) -> str:
        proof_text = self.owner.bulletin.evidence(observed)
        return "\n" + proof_text if proof_text else ""

    def _zero_note(self, pending: _Pending, observed: Any) -> str:
        call_args, ledger = pending.args, self.owner.ledger
        if pending.name != self.owner.brief.roles.get("cart") or not any(_as_int(v_) == 0 for v_ in call_args.values() if not isinstance(v_, str)):
            return ""
        if any(isinstance(v_, str) and v_ in ledger.rows and _mentions(observed, v_) for v_ in call_args.values()):
            return "\n[Kyo] A count of 0 kept that line in the cart: this tool adds the count to the line, so a negative count removes units."
        return ""

    def _note_text(self, observed: Any) -> str:
        bulletin = self.owner.bulletin
        bulletin.sold_flag = False
        lines = bulletin.new_notes(observed)
        if not lines:
            return ""
        notice_meaning = bulletin.parse_note(lines)
        return "\n[note read] " + notice_meaning + ". Apply this change to your picks before you check out." if notice_meaning else ""

    def _refusal_text(self, pending: _Pending, env_reply: dict[str, Any], observed: Any) -> str:
        bulletin = self.owner.bulletin
        if not (self.owner.till.is_checkout(pending.name, pending.args) and not env_reply.get("finished")
                and (bulletin.sold_flag or any(cart_id in self.owner.interviewer.reply_text(observed) for cart_id in bulletin.sold_texts))):
            return ""
        view = self.owner.reviewer.find_sold()
        return "\n[Kyo read the cart's lines again after the refusal]\n" + view if view else ""


class Errand:
    """One episode from the first reading of the request to the finish."""

    def __init__(self, issue: dict[str, Any]):
        self._make_parts()
        env_part = issue["environment"]
        self.binding = dict(env_part["binding"])
        self.view = env_part["policy_view"]
        self.query = str(self.view.get("query") or "")
        self.tool_defs = [t_ for t_ in self.view.get("tools") or [] if isinstance(t_, dict) and isinstance(t_.get("function"), dict) and isinstance(t_["function"].get("name"), str)]
        self.schemas = {t_["function"]["name"]: t_["function"].get("parameters") if isinstance(t_["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t_ in self.tool_defs}
        self.max_steps = self._max_steps_of(self.view)
        self.steps_taken = 0
        self.done = bool(self.view.get("finished"))
        self.failed_turns = 0
        self.turn_count = 0
        self.chat: list[dict[str, Any]] = []
        self.epoch = 0

    def run(self) -> list[dict[str, Any]]:
        if self.done:
            return self.relay.trace
        self.brief.parse_task()
        self.relay.trace.append({"kind": "brief", "brief": self.brief.parsed})
        first_view = self.view.get("opening")
        if first_view is not None:
            self.ledger.absorb(first_view)
        self.chat = [{"role": "system", "content": SHOPPER_GUIDE}, {"role": "user", "content": self._first_prompt(first_view)}]
        self.drive(int(MAX_TURNS * self.relay.cap_scale()), self.relay.hard_spend())
        if not self.done:
            self._wrap_up()
        self.relay.trace.append({"kind": "end", "finished": self.done, "steps": self.steps_taken, "cost": round(self.relay.cost, 5), "calls": self.relay.llm_calls})
        return self.relay.trace

    def drive(self, turn_cap: int, spend_limit: float) -> None:
        idle_count = [0]
        while not self.done and not self.relay.credit_out and self.relay.left() > 0 and self.turn_count < turn_cap and self.relay.cost < spend_limit:
            if self.interviewer.should_ask():
                summary = self.interviewer.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.brief.status()})
                continue
            if self.interviewer.asked and self.reviewer.walk_due():
                walk_text = self.reviewer.walk_links()
                if walk_text:
                    self.chat.append({"role": "user", "content": walk_text + "\n" + self.brief.status()})
                continue
            if self._one_turn(idle_count) == "stop":
                break


    def _calls_in(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        pendings = []
        for item_ix, call_spec in enumerate(message.get("tool_calls") or []):
            if not isinstance(call_spec, dict):
                continue
            fn = call_spec.get("function") if isinstance(call_spec.get("function"), dict) else {}
            raw = fn.get("arguments")
            args_parsed = raw if isinstance(raw, dict) else (_first_object(raw) if isinstance(raw, str) and raw.strip() else {})
            pendings.append({"id": str(call_spec.get("id") or f"k{self.turn_count}_{item_ix}"), "type": "function",
                          "function": {"name": str(fn.get("name") or ""), "arguments": _dumps(args_parsed if isinstance(args_parsed, dict) else {})}})
        return pendings

    def _ask_before_commit(self, pendings: list[dict[str, Any]], content: str) -> bool:
        """A turn that would commit (cart, checkout, ending) before the shopper was asked: the interview goes first."""
        commit_tools = {self.brief.roles.get("cart"), self.brief.roles.get("buy"), self.brief.roles.get("refuse")} - {None}
        if self.interviewer.asked or not self.brief.roles.get("ask") or self.relay.messages_left() <= 0 or not any(u_["function"]["name"] in commit_tools for u_ in pendings) or self.relay.left() <= 8:
            return False
        self.chat.append({"role": "assistant", "content": content or None, "tool_calls": pendings})
        for call_spec in pendings:
            self.chat.append({"role": "tool", "tool_call_id": call_spec["id"], "content": 'Not run (no step used): the shopper is asked first.'})
        summary = self.interviewer.interview()
        self.chat.append({"role": "user", "content": (summary or 'The interviewer found nothing to ask.') + "\n" + self.brief.status()})
        return True

    def _one_turn(self, idle_count: list[int]) -> str:
        """One model turn: its tool calls run and answered, or an idle or failed turn counted."""
        self.relay.trim_history()
        self.turn_count += 1
        message = self.relay.ask_models(self.chat, [LIGHT_MODEL, SPARE_MODEL], 3000, self.dispatch.offered_tools())
        if message is None:
            self.failed_turns += 1
            if self.relay.spent_out or self.relay.llm_calls >= self.relay.call_cap() or self.failed_turns > FAILED_TURNS_MAX:
                return "stop"
            self.relay.pause(FAILED_TURN_PAUSE_S)
            return "next"
        pendings = self._calls_in(message)
        content = self._text_of(message)
        if not pendings:
            return self._no_call_turn(idle_count, content)
        idle_count[0] = 0
        if not self._ask_before_commit(pendings, content):
            self._run_turn_calls(pendings, content)
        return "next"

    def _first_prompt(self, first_view: Any) -> str:
        opening_text = "SHOPPER'S REQUEST:\n" + self.query
        if first_view is not None:
            opening_text += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.relay.render(first_view)
        return opening_text + "\n\nStart by searching each item's category and opening several listings in full, so the interviewer can see which properties these listings have.\n\n" + self.brief.status()

    def _wrap_up(self) -> None:
        """Kyo's own ending; a refused ending gets a short extra loop with the store's reason, then a last try."""
        relay = self.relay
        turned_down = self.dispatch.finish(retry_ask=True)
        if turned_down and not self.done and not relay.credit_out and not relay.no_credit and relay.left() > 1:
            self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned_down + '\nApply what the store says, then end the episode.\n' + self.brief.status()})
            self.drive(self.turn_count + 10, self.relay.hard_spend() + 0.01)
        if not self.done:
            self.dispatch.finish()

    @staticmethod
    def _text_of(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        return _dumps(content) if content else ""

    def _no_call_turn(self, idle_count: list[int], content: str) -> str:
        idle_count[0] += 1
        self.chat.append({"role": "assistant", "content": content or "(no call)"})
        self.chat.append({"role": "user", "content": 'Call one of the tools to go on.\n' + self.brief.status()})
        return "stop" if idle_count[0] >= 3 else "next"

    def _run_turn_calls(self, pendings: list[dict[str, Any]], content: str) -> None:
        self.chat.append({"role": "assistant", "content": content or None, "tool_calls": pendings})
        self.relay.trace.append({"kind": "turn", "text": content[:300], "calls": [u_["function"] for u_ in pendings]})
        replies = self.dispatch.run_calls(pendings)
        for pos, (call_id, text) in enumerate(replies):
            if pos == len(replies) - 1:
                text += "\n" + self.brief.status()
            self.chat.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def _make_parts(self) -> None:
        self.relay = Relay(self)
        self.kit = Toolkit(self)
        self.brief = Brief(self)
        self.ledger = Ledger(self)
        self.scout = Scout(self)
        self.interviewer = Interviewer(self)
        self.clerk = Clerk(self)
        self.bulletin = Bulletin(self)
        self.reviewer = Reviewer(self)
        self.till = Till(self)
        self.dispatch = Dispatcher(self)

    @staticmethod
    def _max_steps_of(policy: dict[str, Any]) -> int:
        """The episode's step limit from the policy view; 30 when it states none."""
        step_items = _as_int(policy.get("max_steps"))
        if step_items is None and isinstance(policy.get("max_steps"), float):
            step_items = int(policy["max_steps"])
        return step_items if step_items and step_items > 0 else 30


def agent_main(task_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        errand = Errand(task_data)
    except Exception as fault:
        return [{"kind": "crash", "error": f"{type(fault).__name__}: {fault}"[:400]}]
    try:
        return errand.run()
    except Exception as fault:
        errand.relay.trace.append({"kind": "crash", "error": f"{type(fault).__name__}: {fault}"[:400]})
        try:
            if not errand.done:
                errand.dispatch.finish()
        except Exception as late_fault:
            errand.relay.trace.append({"kind": "crash", "error": f"{type(late_fault).__name__}: {late_fault}"[:400]})
        return errand.relay.trace


__all__ = ["agent_main"]
