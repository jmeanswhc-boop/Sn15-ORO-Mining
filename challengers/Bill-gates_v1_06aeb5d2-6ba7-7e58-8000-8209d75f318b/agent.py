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

LOOP_CALLS = 42
ALL_CALLS = 60
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 100000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
# store calls time out after 30 s so the finish still fits
_env = ProxyClient(timeout=30, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts (a value with one more word, "<shade> <colour>", meets "<colour>"), but only as one unbroken phrase: a word inside the required value makes a different value ("<a> <x> <b>" does not meet "<a> <b>"). A value with different words does not count (an abbreviation or code is not the word it stands for, and two values that share only one word are different values). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (the colour of one part is not the colour of the whole), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- When a listing's page says that another listing decides one of its properties (go by that one, not by this item), the listing has that other listing's value for that property, whatever its own attribute says.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device) or a model the request names, choose them together from the listings' own words (a list of what it fits, a model or series code that both pages carry, a list that continues on another listing's page). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") never rule a listing out; but among fitting listings that are in stock and within budget, prefer one that also meets a wish, unless the request asks for the cheapest or for the most or least of a measure.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right (one that also meets a wish, when the rows you have seen show one): do not waste steps hunting for something better.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1; when the request calls the product a pair ("1 pair of <product>", "sold as a pair", "in a pack of 1 pair"), "pack size: pair" is also one of its musts, but "1 set of" or "1 pack of" alone only names the unit and is no must), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request for one "meant for a <setting> occasion" gives "occasion: <setting>", "a <n>-<unit> <part>" gives "<part> length: <n> <unit>"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("lid colour: unlike the lid colour on the teapot that Brisk lists with a steel handle"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". But a colour, a material, a brand or maker, or the make or model it is meant for, written in front of the product ("<value> <product>"), IS a must as well ("<property>: <value>"), kept in "name" too. So is the kind of part named in front of, or for, a store-department name that joins two kinds with "&" or "and" ("<kind> <a> & <b>", "<kind> for the <a> & <b>"): "type: <kind>"; a word in front of a product named by one kind stays in "name" and is never a "type". This applies only when those words say what the thing itself is; a position or side, a colour, a material, a maker or model, a style or a condition written there keeps its own property ("position: <value>", "make: <value>") and is never a "type". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price, the smallest total or the best value for money (the most affordable, the best deal, the least spent), "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
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
Name a hinted property with the words of the listed key of that item that it is about, when one matches it; else by its plain name.
2. Then the listed keys, ranked. Unwritten preferences are most often about what the product is made of (its material), how many pieces one listing holds (its pack size), where it goes or which side (its position), its colour, and which kind or variant it is (its type, format, effect or power source); next its size or capacity. Among an item's listed keys, take first the one of these kinds whose example values differ between listings. Technical figures a buyer rarely picks by (megapixels, memory, storage, processor, connectivity, input voltage, wattage), condition, assembly time, care, packaging, pattern or shape details, extras such as pockets, and the length, width, height, depth or weight of the product's body come last. Avoid keys seen on only one listing.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose single value lists several purchase variants at once (a mix of sizes, colours and pack counts in one value; a key giving one pack size or piece count per listing is an ordinary property), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- The item of a need is the item the ANSWER speaks of: when the answer names another requested item than the question did, use the item the answer names.
- "needs": every firm preference an answer states, with the exact value or limit and its unit (a lower limit with its unit, one value, or a value ruled out). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units (a limit keeps the unit or noun that follows its number).
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "<shade> <colour>" for "<colour>", units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (the colour of one part is not the colour of the whole), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "<shade> <colour>" for "<colour>", limits include the edge); a key for a different property, even a related one (the colour of one part is not the colour of the whole), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
A record marked READ shows the listing's attributes, but its page is not opened yet: when it is a link, give its id (Kyo opens the page) and treat the chain as not done.
A record is that link only when it shows every value the requirement (or the previous page) uses to pick that listing out, each in the record's own words or units (the same value written or measured another way counts); a listing, even from the same shop, that lacks one of those values or shows a different one is not it: give "" as its id, and in "about" name every value the listing must show.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


PATTERNS = """Listings in this store state identifiers and compatibility with sentence patterns that recur across listings. Below are such patterns, numbered, with every code replaced by <C> (and <C>+ standing for a list of one or more codes that all mean the same), each followed by one example exactly as a listing shows it. For each pattern, say what each <C> means for the listing whose text carries the sentence, in order:
"own": names this listing itself (its model, its part number, its series or line),
"fits": a model, series or part that this listing works with, fits or takes,
"family": a whole group of models this listing fits, written with placeholder characters (put the placeholder character in "wild"),
"except": a model or part left out of what this listing fits,
"more": another listing whose page continues this listing's list of what it fits,
"subject": the listing the sentence is about when that is not the listing carrying it; the codes after it then say what that subject fits,
"none": a code that says nothing about identity or compatibility (a size, a capacity, a speed, a product number nobody refers to).
Say "series": true when the pattern's "fits" codes are short codes standing for groups of longer codes that begin with them.
Return ONE JSON object and nothing else: {"patterns": [{"n": pattern number, "slots": ["own" or "fits" or "family" or "except" or "more" or "subject" or "none", one entry per <C> in order], "wild": "" or the placeholder character, "series": true or false}]}"""

# a code a listing carries: an all-capitals word followed by a number ("<AB> 1234"), or a token mixing letters and
# digits (placeholder characters such as * or # included), joined by hyphens; LIST joins codes into one list
CODE = r"(?<![A-Za-z0-9*#])(?:[A-Z]{2,6}[ -](?=[A-Za-z0-9*#]*\d)[A-Za-z0-9*#]+(?:-[A-Za-z0-9*#]+)*|(?=[A-Za-z0-9*#-]*\d)(?=[A-Za-z0-9*#-]*[A-Za-z*#])[A-Za-z0-9*#]+(?:-[A-Za-z0-9*#]+)*)(?![A-Za-z0-9*#])"
LIST = r"(?:\s*(?:,|/|&|\band\b|\bor\b)\s*)"
FIT_CALLS = 2               # pattern readings per episode (one reads every recurring pattern seen so far)
CODE_WAITS = 3              # checkouts a code mismatch may hold back
CODE_SEARCHES = 3           # searches for an accepted code Kyo runs itself, per episode
EARLY_ROUNDS = 2            # chain readings right after the shopper's answers (the checkout keeps its own)

SPEND_SOFT = 0.045          # past this spend, late reviews run on the light model only
SPEND_HARD = 0.065          # the tool loop's spend limit; the code then finishes the episode
SPEND_STOP = 0.073          # hard ceiling for any new LLM request outside a repair
REPAIR_ROOM = 0.012         # spend a repair (a refused checkout, a review's verdict to act on) may add
REPAIR_CEILING = 0.085      # the most any task may reach, repairs included
REPAIR_SECONDS = 70.0       # how long a repair's extra room lasts
HEAVY_CHECK = 0.012         # typical price of one heavy-model review: it starts only when its verdict can be acted on
EXPLORE_UNTIL = 210.0       # seconds since agent_main began: the tool loop ends and the code finishes
CALLS_UNTIL = 228.0         # latest start for an LLM request: a 50 s timeout plus the finish stay under the 300 s kill
ASK_UNTIL = 160.0           # latest start for a question to the shopper


class Meter:
    """Per-task limits on money and wall time. A task over the run key's cap scores 0 and the harness kills a task at
    300 s, so every limit here sits well inside both."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.repair_cap = 0.0
        self.repair_until = -1.0
        self.explore_until = EXPLORE_UNTIL

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def room(self) -> float:
        return self.repair_cap if self.elapsed() < self.repair_until else 0.0

    def can_call(self, cost: float) -> bool:
        return cost < max(SPEND_STOP, self.room()) and self.elapsed() < CALLS_UNTIL

    def can_wait(self, pause: float) -> bool:
        return self.elapsed() + pause < CALLS_UNTIL

    def exploring(self, cost: float, cap: float) -> bool:
        return cost < max(cap, self.room()) and self.elapsed() < self.explore_until

    def asking(self, cost: float) -> bool:
        return cost < SPEND_HARD and self.elapsed() < ASK_UNTIL

    def check_models(self, cost: float) -> list[str]:
        """The heavy model only while its verdict can still be acted on inside the hard limit; else the light one."""
        return [HEAVY, LIGHT] if cost <= SPEND_SOFT and cost + HEAVY_CHECK <= SPEND_HARD else [LIGHT]

    def open_repair(self, cost: float, late: bool = False) -> float:
        """Room to repair (a refused checkout or ending, a review's verdict to act on), whatever was spent before, for a
        while; a late repair (after the tool loop ended) may also run past the usual end of the loop."""
        # repeated repairs do not keep stacking room: one repair reaches SPEND_STOP + REPAIR_ROOM at most
        self.repair_cap = min(REPAIR_CEILING, max(self.room(), min(cost + REPAIR_ROOM, SPEND_STOP + REPAIR_ROOM), SPEND_HARD + 0.01))
        self.repair_until = self.elapsed() + REPAIR_SECONDS
        if late or self.elapsed() > EXPLORE_UNTIL - 30:
            self.explore_until = CALLS_UNTIL - 5
        return self.repair_cap


def _code_norm(code: str) -> str:
    return re.sub(r"[^A-Z0-9*#]", "", str(code).upper())


def _sentences(text: str) -> list[str]:
    # "no." and "nos." (a number follows) end no sentence
    return [s.strip() for s in re.split(r"(?<=[.!?])(?<!\b[Nn]o\.)(?<!\b[Nn]os\.)\s+|[\n;|]", text) if s.strip()]


def _patterns(record: dict[str, Any]) -> list[tuple[str, list[str], str]]:
    """The code-carrying sentences and bracketed tags of a listing's own words, each as (its pattern with every code
    written <C>, a list of codes written <C>+ and every other number <n>, its codes, the words as shown)."""
    out = []
    for value in record.values():
        if not isinstance(value, str) or not re.search(r"\d", value):
            continue
        parts = [m.group(0) for m in re.finditer(r"[\[(][^\])]{2,40}[\])]", value)]
        parts += [s for s in _sentences(value) if len(s) <= 180]
        for part in parts:
            codes = [m.group(0) for m in re.finditer(CODE, part)]
            if not codes or len(codes) > 12:
                continue
            shape = re.sub(r"\d+", "<n>", re.sub(CODE, "<C>", part))
            shape = re.sub(r"<C>(?:" + LIST + r"<C>)+", "<C>+", shape)
            out.append((re.sub(r"\s+", " ", shape).strip().lower()[:160], codes, part[:200]))
    return out


def _code_hits(pattern: tuple[str, str, str, bool], code: str) -> bool:
    """Whether a fit code (one code, a series a longer code begins with, or a family with placeholder characters) covers
    a listing's own code."""
    text, kind, wild, series = pattern
    want, have = _code_norm(text), _code_norm(code)
    if not want or not have:
        return False
    if kind == "family" and wild:
        mark = _code_norm(wild)[:1]
        if mark:
            rx = "".join("[0-9A-Z]" if ch == mark else re.escape(ch) for ch in want)
            return bool(re.fullmatch(rx, have) or re.fullmatch(rx, re.sub(r"^[A-Z]+", "", have)))
    if want == have:
        return True
    head = _code_norm(re.split(r"[- ]", str(code).strip())[0])
    return head == want or (series and have.startswith(want))


def _patterns_meet(one: tuple[str, str, str, bool], two: tuple[str, str, str, bool]) -> bool:
    """Two fit codes of two listings name a common model: the same code, or one family covering the other's code."""
    if _code_norm(one[0]) == _code_norm(two[0]):
        return True
    return (one[1] == "family" and _code_hits(one, two[0])) or (two[1] == "family" and _code_hits(two, one[0]))


def _kind_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in ("and", "the", "for", "with", "of", "a", "an")}


def _same_word(one: str, two: str) -> bool:
    """One word, allowing a plural ending (two letters at most) on either."""
    if one == two:
        return True
    short, long = sorted((one, two), key=len)
    return len(short) >= 3 and long.startswith(short) and len(long) - len(short) <= 2


def _said_count(request: str, name: str, others: list[str] = ()) -> int | None:
    """The one number the request writes right before this product's head noun ("need 2 <...> <noun>"), if any."""
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if len(w) >= 3]
    if not words:
        return None
    head = words[-1][:-1] if len(words[-1]) > 3 and words[-1].endswith("s") else words[-1]
    # another item's name that carries this head noun before its own ("<noun> pad") makes the noun ambiguous
    if any(_same_word(head, w) for o in others for w in [x for x in re.findall(r"[a-z]+", o.lower()) if len(x) >= 3][:-1]):
        return None
    pat = r"(?:\bneed|\bwant|\bget|\bbuy|,|\band|\bplus)\s+(\d{1,2})\s+((?:[^\W\d][\w&/-]*\s+|[&/]\s+){0,3}?)" + re.escape(head) + r"s?\b"
    found = set()
    for hit in re.finditer(pat, request, re.I):
        # the words between the number and the head noun are this product's own, and a name of several words shows one of them
        between = [w for w in re.findall(r"[a-z]+", hit.group(2).lower()) if len(w) >= 3]
        if all(any(_same_word(b, w) for w in words) for b in between) and (len(words) == 1 or between):
            found.add(int(hit.group(1)))
    return found.pop() if len(found) == 1 else None


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
        self.meter = Meter()
        self.ref_notes: dict[str, str] = {}
        self.fit_roles: dict[str, dict[str, Any]] = {}
        self.fit_seen: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.fit_calls = 0
        self.code_waits = 0
        self.pairs_cache: tuple[Any, list[tuple[int, int, str]]] = (None, [])
        self.fit_about_cache: tuple[Any, list[Any]] = (None, [])
        self.cat_key_cache: tuple[int, str | None] = (-1, None)
        self.label_items: dict[str, tuple[int, int]] = {}
        self.code_searches = 0
        self.searched_words: set[str] = set()
        self.early_rounds = 0
        self.fit_decline_held = False
        self.cheaper_fit_told = False
        self.last_checked = False
        self.late_cart_told = False
        self.empty_held = False
        self.held_step = -1
        self.bare_buys: dict[str, int] = {}
        self.built_counts: dict[str, int] = {}
        self.built_step = -1
        self.buy_step = -1
        self.refuse_step = -1
        self.sent_cart: dict[str, int] = {}

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
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or not self.meter.can_call(self.cost):
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
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL or not self.meter.can_wait(WAITS[tries]):
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
                self.trace.append({"kind": "spend", "model": model, "cost": usage.get("cost"), "in": usage.get("prompt_tokens"),
                                   "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "out": usage.get("completion_tokens")})
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
        # above the soft limit a heavy model is no fallback for a light one's malformed reply
        models = [m for m in models if m != HEAVY or self.cost <= SPEND_SOFT] or models[:1]
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS or not self.meter.can_call(self.cost):
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

    # ---------- what fits what, read from the codes the listings carry ----------
    def fit_pairs(self) -> list[tuple[int, int, str]]:
        """Which requested items must fit which, read from the store's own codes once their patterns are known:
        (item, the item it must fit or 0, the model code the request names). A model code in a sentence of the request
        binds the items that sentence names (or, when it names none, every item whose listings state what they fit).
        Between two items, the one whose listings carry no fit list is the anchor when the other's lists name its codes,
        or the part when the other's lists take its codes; two such links on seen listings are enough."""
        if not self.fit_roles:
            return []
        mark = (len(self.seen_order), len(self.fit_roles), sum(len(r) for r in self.rows.values()))
        if self.pairs_cache[0] == mark:
            return self.pairs_cache[1]
        kinds = {n: [i for i in self.seen_order if self.item_of(i) == n][:20] for n in range(1, len(self.items) + 1)}
        listing_fits = {n: any(self.fit_marks(i)["fits"] for i in ids) for n, ids in kinds.items()}
        pairs: list[tuple[int, int, str]] = []
        # a code in the request is a model only when some seen listing's fit list names it (not a size, a year, a figure)
        fit_pats = [p for i in self.seen_order for p in self.fit_marks(i)["fits"]]
        for sentence in _sentences(self.request):
            for clause in re.split(r";|,\s*(?=(?:and\s+)?(?:\d+|one|two|three|four|five|a|an)\b)|\band\s+(?=\d)", sentence):
                models = [m.group(0) for m in re.finditer(CODE, clause) if re.search(r"\d", m.group(0))]
                models = [c for c in models if any(_code_hits(p, c) for p in fit_pats)]
                if not models:
                    continue
                named = self.items_named(clause) or self.items_named(sentence)
                for n in (named or list(kinds)):
                    if listing_fits.get(n):
                        for code in models:
                            if not any(p[0] == n and p[2] == code for p in pairs):
                                pairs.append((n, 0, code))
        for n, ids in kinds.items():
            for m, others in kinds.items():
                if n == m or not ids or not others or any(p[0] == n for p in pairs) or any(p[:2] == (m, n) for p in pairs):
                    continue
                if not self.request_ties(n, m):
                    continue
                links = sum(1 for i in ids for j in others if self.code_link(i, j, listing_fits[n], listing_fits[m]))
                if links >= 2:
                    pairs.append((n, m, ""))
        self.pairs_cache = (mark, pairs)
        return pairs

    def items_named(self, text: str) -> list[int]:
        """The requested items a piece of the request names (by the words of their names)."""
        said = _kind_words(text)
        return [n for n, item in enumerate(self.items, 1) if any(_same_word(w, v) for w in _kind_words(item.name) if len(w) > 2 for v in said)]

    def request_ties(self, part: int, anchor: int) -> bool:
        """The request ties two items: a clause that speaks of fitting names the anchor item (beyond words of the part's
        own name) and the part, or names the anchor alone; a sentence that speaks of fitting for every or all items
        counts as such a clause for each item it covers."""
        own = _kind_words(self.items[part - 1].name)
        theirs = {w for w in _kind_words(self.items[anchor - 1].name) if len(w) > 2}
        fit = r"\b(?:fits?|fitting|compatible|works? with|suits?|match(?:es)?|goes with)\b"
        for sentence in _sentences(self.request):
            if not re.search(fit, sentence, re.I):
                continue
            every = re.search(r"\b(?:every\w*|all|each|both)\b", sentence, re.I)
            for clause in ([sentence] if every else [c for c in re.split(r";|,", sentence) if re.search(fit, c, re.I)]):
                rest = {v for v in _kind_words(clause) if not any(_same_word(v, w) for w in own)}
                if not any(_same_word(w, v) for w in theirs for v in rest):
                    continue
                named = self.items_named(clause)
                if every or part in named or named == [anchor]:
                    return True
        return False

    def code_link(self, part: str, anchor: str, part_lists: bool, anchor_lists: bool) -> bool:
        """A direct code link that makes `part` a part of `anchor`. A list of series (short codes that longer own codes
        begin with) is a device's list of the parts it takes: the anchor's series list takes the part's own code. A list
        of whole codes is a part's list of the models it fits: the part's list names the anchor's own code."""
        mine, theirs = self.fit_marks(part), self.fit_marks(anchor)
        if not part_lists and mine["own"]:
            if any(p[3] and _code_hits(p, o) for p in self.fit_accepts(anchor) for o in mine["own"]):
                return True
        if not anchor_lists and theirs["own"]:
            return any(not p[3] and _code_hits(p, o) for p in mine["fits"] for o in theirs["own"])
        return False

    def item_for_label(self, label: str) -> int:
        """The requested item a store category label belongs to (0 when none): its own label, else the item sharing
        the largest part of the label's words and of its own words."""
        if not label:
            return 0
        hit = self.label_items.get(label)
        if hit is not None and hit[0] == len(self.items):
            return hit[1]
        theirs = _kind_words(label)
        scores = []
        for n, item in enumerate(self.items, 1):
            if item.shelf and _kind_words(item.shelf) == theirs:
                scores.append((9.0, -n))
                continue
            mine = _kind_words(item.shelf or "") | _kind_words(item.name)
            named = _kind_words(item.name) or mine
            shared = [w for w in theirs if any(_same_word(w, v) for v in mine)]
            scores.append(((len(shared) / max(1, len(theirs)) + len([v for v in named if any(_same_word(v, w) for w in theirs)]) / max(1, len(named))) if shared else 0.0, -n))
        best = max(scores) if scores else (0.0, 0)
        number = -best[1] if best[0] > 0 else 0
        self.label_items[label] = (len(self.items), number)
        return number

    def item_of(self, ident: str) -> int:
        """The requested item whose kind a listing's category label fits best (0 when none does)."""
        if self.cat_key_cache[0] != len(self.rows):
            self.cat_key_cache = (len(self.rows), self.category_key())
        key = self.cat_key_cache[1]
        return self.item_for_label(str((self.rows.get(ident) or {}).get(key) or "") if key else "")

    def fit_shapes(self, ident: str) -> list[tuple[str, list[str], str]]:
        """A listing's code-carrying sentences, leaving out its id and a value that is a lone code (an attribute value
        such as a model number says nothing on its own about what it fits)."""
        record = {k: v for k, v in (self.rows.get(ident) or {}).items() if k != self.id_key and v != ident}
        return [(s, c, w) for s, c, w in _patterns(record) if s not in ("<c>", "<c>+")]

    def fit_learn(self) -> None:
        """Ask the model once for the meaning of each pattern that recurs on the requested kinds' listings (the store's
        own way of writing models, series and compatibility); the code then reads every listing by it."""
        if self.fit_calls >= FIT_CALLS or self.cost > SPEND_HARD or self.meter.elapsed() > EXPLORE_UNTIL:
            return
        wanted = set(range(1, len(self.items) + 1))
        seen: dict[str, set[str]] = {}
        codes: dict[str, set[str]] = {}
        sample: dict[str, str] = {}
        for ident in self.seen_order:
            if self.item_of(ident) not in wanted:
                continue
            for shape, found, words in self.fit_shapes(ident):
                seen.setdefault(shape, set()).add(ident)
                codes.setdefault(shape, set()).add("|".join(found))
                sample.setdefault(shape, words)
        # the store's own way of writing codes: a short sentence that recurs on several listings with different codes
        # (a long one, or one repeated with the same codes, is a copied description)
        fresh = sorted((s for s, ids in seen.items() if len(ids) >= 2 and len(codes[s]) >= 2 and s not in self.fit_roles and len(s) <= 70 and len(s.split()) <= 12),
                       key=lambda s: -len(seen[s]))[:14]
        if len(fresh) < (1 if not self.fit_roles else 3):
            return
        self.fit_calls += 1
        prompt = PATTERNS + "\n\nPATTERNS:\n" + "\n".join(f"{n}. {s}\n   example: {sample[s]}" for n, s in enumerate(fresh, 1))
        got = self.ask_json(prompt, [LIGHT], 3000) or {}
        if not _items(got.get("patterns")):
            # a failed reading leaves the patterns unread, so a later check reads them again
            self.trace.append({"kind": "fit_learn_failed"})
            return
        for entry in _items(got.get("patterns")):
            number = _count(entry.get("n")) if isinstance(entry, dict) else None
            if not number or not 1 <= number <= len(fresh):
                continue
            slots = [str(s).strip().lower() for s in _items(entry.get("slots"))]
            self.fit_roles[fresh[number - 1]] = {"slots": slots, "wild": str(entry.get("wild") or "").strip()[:1], "series": entry.get("series") is True}
        for shape in fresh:
            self.fit_roles.setdefault(shape, {"slots": [], "wild": "", "series": False})
        self.fit_seen.clear()
        self.trace.append({"kind": "fit_learn", "patterns": {s: self.fit_roles[s] for s in fresh}})

    def fit_marks(self, ident: str) -> dict[str, Any]:
        """A listing's own codes, what it fits (each as (code, kind, placeholder, series)), what it leaves out, the
        listings its list continues on, and what it says other listings fit."""
        record = self.rows.get(ident) or {}
        mark = (len(self.fit_roles), len(record), self.cat_key_cache[1])
        cached = self.fit_seen.get(ident)
        if cached and cached[0] == mark:
            return cached[1]
        marks: dict[str, Any] = {"own": set(), "fits": [], "except": set(), "more": set(), "about": []}
        for shape, codes, words in self.fit_shapes(ident):
            role = self.fit_roles.get(shape)
            if not role or not role["slots"]:
                continue
            slots, subject = role["slots"], None
            # each code's place among the pattern's <C> and <C>+ tokens: every code of one list shares its token's slot
            marked = re.sub(CODE, "\x00", words)
            token = [k for k, run in enumerate(re.finditer("\x00(?:" + LIST + "\x00)*", marked)) for _ in range(run.group(0).count("\x00"))]
            for index, code in enumerate(codes):
                place = token[index] if index < len(token) else len(slots)
                slot = slots[place] if place < len(slots) else "none"
                entry = (code, slot, role["wild"], role["series"])
                if slot == "subject":
                    subject = code
                elif subject is not None and slot in ("fits", "family"):
                    marks["about"].append((subject, entry))
                elif slot == "own":
                    marks["own"].add(code)
                elif slot in ("fits", "family"):
                    marks["fits"].append(entry)
                elif slot in ("except", "more"):
                    marks[slot].add(code)
        self.fit_seen[ident] = (mark, marks)
        return marks

    def fit_about(self) -> list[tuple[str, tuple[str, str, str, bool]]]:
        """What listings say other listings fit, as (that listing's code, what it fits), gathered once per change."""
        mark = (len(self.seen_order), len(self.fit_roles), sum(len(r) for r in self.rows.values()))
        if self.fit_about_cache[0] != mark:
            found = [(_code_norm(subject), entry) for ident in self.seen_order for subject, entry in self.fit_marks(ident)["about"]]
            self.fit_about_cache = (mark, found)
        return self.fit_about_cache[1]

    def fit_named(self, code: str) -> list[str]:
        """Listings seen whose own code is this code."""
        want = _code_norm(code)
        return [i for i in self.seen_order if any(_code_norm(c) == want for c in self.fit_marks(i)["own"])]

    def fit_accepts(self, anchor: str) -> list[tuple[str, str, str, bool]]:
        """What a listing fits: its own list, the lists on the pages it continues on, and what other listings say it fits."""
        marks = self.fit_marks(anchor)
        found = list(marks["fits"])
        for code in marks["more"]:
            for other in self.fit_named(code):
                found += self.fit_marks(other)["fits"]
        own = {_code_norm(c) for c in marks["own"]}
        found += [entry for subject, entry in self.fit_about() if subject in own]
        return found

    def fit_between(self, part: str, anchor: str | None, named: str = "") -> bool | None:
        """Whether a listing fits the anchor listing (or the model the request names), by both listings' codes: True,
        False, or None when neither side states anything to judge by."""
        mine = self.fit_marks(part)
        if anchor:
            theirs = self.fit_marks(anchor)
            own = set(theirs["own"])
            accepts = self.fit_accepts(anchor)
        else:
            theirs = {"own": set(), "fits": [], "except": set()}
            own = {m.group(0) for m in re.finditer(CODE, named)}
            accepts = []
        if any(_code_hits((x, "fits", "", False), o) for x in mine["except"] for o in own):
            return False
        said = False
        if mine["fits"] and own:
            said = True
            if any(_code_hits(p, o) for p in mine["fits"] for o in own):
                return True
        if accepts and mine["own"]:
            said = True
            if any(_code_hits(p, o) for p in accepts for o in mine["own"]):
                return True
        if anchor and mine["fits"] and theirs["fits"]:
            # two listings that must suit one model: some model both their lists name
            said = True
            if any(_patterns_meet(p, q) for p in mine["fits"] for q in theirs["fits"]):
                return True
        mine_own = {_code_norm(c) for c in mine["own"]}
        for subject, entry in self.fit_about():
            if subject in mine_own and own:
                said = True
                if any(_code_hits(entry, o) for o in own):
                    return True
        return False if said else None

    def code_search(self, words: str) -> list[str]:
        """One search Kyo runs itself (an accepted code with the item's name); returns the listings it showed."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        field = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        words = " ".join(words.split())[:100]
        if not tool or not field or not words or self.code_searches >= CODE_SEARCHES or self.left() < 6 or words in self.searched_words:
            return []
        args: dict[str, Any] = {field: words}
        for k, v in props.items():
            if k in (schema.get("required") or []) and k != field and isinstance(v, dict) and isinstance(v.get("enum"), list) and v["enum"]:
                args[k] = v["enum"][0]
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return []
        self.code_searches += 1
        self.searched_words.add(words)
        reply = self.act(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get("observation"))
        found = [i for i, _ in self.records_in(reply.get("observation"))]
        self.trace.append({"kind": "code_search", "words": words, "rows": found[:6]})
        return found

    def open_page(self, ident: str) -> str:
        """Open one listing's page (the page view) for a check; returns what it showed."""
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        if not tool or not target or ident in self.opened or self.left() < 6:
            return ""
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
        if faults:
            return ""
        reply = self.act(tool, args)
        if reply is None:
            return ""
        self.observe(tool, reply.get("observation"))
        return f"{ident}: " + self.show(reply.get("observation"))[:700]

    def fit_pages(self, anchor: str) -> list[str]:
        """Open the pages an anchor's list continues on (searching a page's code first when it was not seen)."""
        shown = []
        for code in sorted(self.fit_marks(anchor)["more"]):
            named = self.fit_named(code)
            if not named:
                self.code_search(code)
                named = self.fit_named(code)
            for ident in named[:1]:
                seen = self.open_page(ident)
                if seen:
                    shown.append(seen)
        return shown

    def fit_candidates(self, item: int, anchor: str | None, named: str) -> list[str]:
        """Seen listings of an item's kind that fit the anchor by their codes, buyable, cheapest first."""
        ids = [i for i in self.seen_order if self.item_of(i) == item and self.buyable(i) and self.fit_between(i, anchor, named)]
        return sorted(ids, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))

    def fit_rejects(self, ident: str, cart: list[str]) -> bool:
        """True when the listing's codes say it does not fit its anchor cart line (or the model the request names)."""
        n = self.item_of(ident)
        for item, other, named in self.fit_pairs():
            if item != n:
                continue
            anchor = next((i for i in cart if self.item_of(i) == other and not self.sold_out(i)), None) if other else None
            if other and not anchor:
                continue
            if self.fit_between(ident, anchor, named) is False:
                return True
        return False

    def fit_codes(self, anchor: str | None, named: str) -> list[str]:
        if not anchor:
            return [m.group(0) for m in re.finditer(CODE, named)][:3]
        return list(dict.fromkeys(p[0] for p in self.fit_accepts(anchor)))[:6]

    def fit_check(self) -> str:
        """Before a checkout: each cart line that must fit another line (or a model the request names) does so by the
        listings' own codes, and under a cheapest pick no cheaper fitting listing already read is passed over.
        Returns why the checkout waits ('' when it may go on)."""
        if self.code_waits >= CODE_WAITS or self.left() < 6 or self.meter.elapsed() > EXPLORE_UNTIL:
            return ""
        if self.may_need_fit():
            self.fit_learn()
        pairs = self.fit_pairs()
        if not pairs:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        lines = {n: [i for i in cart if self.item_of(i) == n] for n in range(1, len(self.items) + 1)}
        shown: list[str] = []
        faults: list[str] = []
        for item, other, named in pairs:
            anchors = (lines.get(other) or []) if other else [None]
            if other and not anchors:
                continue
            anchor = anchors[0]
            if anchor:
                shown += self.fit_pages(anchor)
            for part in lines.get(item) or []:
                name = self.items[item - 1].name
                verdict = self.fit_between(part, anchor, named)
                if verdict is not False:
                    if verdict and self.brief.get("pick") == "cheapest" and not self.cheaper_fit_told and self.cost <= SPEND_SOFT:
                        price = self.price_of(part)
                        cheaper = [i for i in self.fit_candidates(item, anchor, named) if i != part and i in self.full
                                   and price is not None and (self.price_of(i) or float("inf")) < price]
                        if cheaper:
                            self.cheaper_fit_told = True
                            faults.append(f"{part} (item {item}, {name}) fits, but the pick is the cheapest and {', '.join(cheaper[:3])} also fit and cost less: check their other requirements and take the cheapest that meets them")
                    continue
                where = f"{anchor} (item {other})" if anchor else f"the model the request names ({named})"
                codes = self.fit_codes(anchor, named)
                good = [i for i in self.fit_candidates(item, anchor, named) if i != part]
                if not good and codes:
                    self.code_search(f"{codes[0]} {name}")
                    good = [i for i in self.fit_candidates(item, anchor, named) if i != part]
                text = f"{part} (item {item}, {name}) does not fit {where}"
                if codes:
                    text += f", which takes {', '.join(codes)}"
                own = sorted(self.fit_marks(part)["own"])[:3]
                if own:
                    text += f"; {part} carries {', '.join(own)}"
                if good:
                    text += f". Seen {name} listings that fit it (cheapest first): {', '.join(good[:4])}: open one to check its other requirements, then swap the line"
                else:
                    swaps = [i for i in self.seen_order if other and self.item_of(i) == other and self.buyable(i) and self.fit_between(part, i, "")]
                    text += ". Among the listings read so far none of that kind matches those codes: search them together with the item's name" + (
                        f"; or change item {other} for a listing whose codes this line matches: {', '.join(swaps[:3])}" if swaps else "")
                faults.append(text)
        if not faults:
            return ""
        self.code_waits += 1
        self.trace.append({"kind": "code_mismatch", "lines": faults})
        return ("[Kyo opened the pages a cart line's list continues on]\n" + "\n".join(shown) + "\n" if shown else "") + \
            "Codes that do not match: " + " | ".join(faults[:3]) + ". Correct those lines, then check out."

    def fit_sets(self) -> str:
        """Before an ending without a purchase: whole sets that fit together among the listings seen, by their codes,
        each priced in full (every item at its quantity), inside the budget, cheapest first."""
        pairs = self.fit_pairs()
        if not pairs or not self.fit_roles:
            return ""
        parts = {item for item, _, _ in pairs}
        others = sorted({o for _, o, _ in pairs if o})
        if len(others) > 1 or any(o in parts for o in others) or (others and any(not o for _, o, _ in pairs)):
            # several anchor kinds, a chain or mixed anchors: no single set can be named with confidence
            return ""
        free = {n: min((i for i in self.seen_order if self.item_of(i) == n and self.buyable(i) and self.price_of(i) is not None),
                       key=lambda i: self.price_of(i) or 0, default=None)
                for n in range(1, len(self.items) + 1) if n not in parts and n not in others}
        if any(v is None for v in free.values()):
            return ""
        priced = isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each"
        options = []
        for other in others or [0]:
            for anchor in ([i for i in self.seen_order if self.item_of(i) == other and self.buyable(i) and (not priced or self.price_of(i) is not None)] if other else [None]):
                picks = {other: anchor} if anchor else {}
                for item, o, named in pairs:
                    if o != other:
                        continue
                    good = [i for i in self.fit_candidates(item, anchor, named) if not priced or self.price_of(i) is not None]
                    if not good:
                        break
                    picks[item] = good[0]
                else:
                    picks.update(free)
                    if len(picks) < len(self.items) or any(p is None for p in picks.values()):
                        continue
                    total = sum((self.price_of(p) or 0) * self.items[n - 1].qty for n, p in picks.items())
                    if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget:
                        continue
                    options.append((total, ", ".join(f"item {n}: {p}" for n, p in sorted(picks.items()))))
        options.sort()
        return "; ".join(f"{text} (total {total:g})" for total, text in options[:3])

    def may_need_fit(self) -> bool:
        """Whether reading the listings' code patterns can matter: several items, or a model code in the request."""
        return len(self.items) > 1 or any(re.search(r"\d", m.group(0)) for m in re.finditer(CODE, self.request))

    def fit_settled(self, number: int, label: str) -> bool:
        """A question about what an item fits, when the request already says what it must fit: the answer is known."""
        if not any(item == number for item, _, _ in self.fit_pairs()) and not any(
                re.search(r"\bfits?\b|compatib|works? with", s, re.I) and any(re.search(r"\d", m.group(0)) for m in re.finditer(CODE, s))
                and number in (self.items_named(s) or [number]) for s in _sentences(self.request)):
            return False
        return bool(re.search(r"compatib|\bfits?\b|\bfitment\b", label, re.I))

    def is_settled(self, number: int, label: str) -> bool:
        """The request or an answer already gives this property for the item. The item's own name words are taken off
        both sides first, so "<item> size" is "size" for that item while "<part> colour" stays apart from "colour"."""
        item = self.items[number - 1]
        own = _stem(item.name) | _stem(item.shelf or "")
        known = {_stem(t.split(":")[0]) for t in item.told + item.musts}
        known |= {k - own for k in known if k - own}
        mine = _stem(label)
        return bool(mine) and (mine in known or (mine - own) in known)

    def early_trail(self) -> str:
        """Right after the shopper's answers: follow each requirement whose value is on another listing (Kyo opens the
        pages it can name; a link no seen listing matches is left to the checkout's chain hold), so the value is in the ledger before the products are
        chosen; this uses its own small budget, the checkout keeps all of its own. Those pages only give the value:
        the listings along a chain are not bought."""
        live = {need for _, need in self.needs_now()}
        if not [need for _, need in self.refs if need in live and need not in self.ref_values] or self.left() < 12:
            return ""
        text, _ = self.trail(early=True)
        return ("\n[Kyo followed the pages these requirements point at; those pages only give the value, the listings on them are not to be bought]\n" + text) if text else ""

    def best_link(self, about: str, ident: str) -> str:
        """Of the seen listings, the one sharing most words with what the chain says about this link (at least three,
        and more than the one the model named): one shop's near twins differ in a word or two."""
        words = {w for w in _kind_words(about) if len(w) > 2}
        if not words:
            return ident

        def hits(i: str) -> int:
            record = self.full.get(i) or self.rows.get(i) or {}
            text = " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float))) + " " + _dump(_fact_block(record) or {})
            return len(words & _kind_words(text))

        mine = hits(ident) if ident in self.rows else -1
        best = max(self.seen_order, key=hits, default=ident)
        return best if best != ident and hits(best) > mine and hits(best) >= 3 else ident

    def page_notes(self, ident: str, need: str) -> str:
        """Sentences of a page that speak of a requirement's property: a rule the page states about the value goes
        with the value itself."""
        prop = {w for w in _kind_words(need.split(":")[0]) if len(w) > 2}
        record = self.full.get(ident) or self.rows.get(ident) or {}
        lines = [s for v in record.values() if isinstance(v, str) for s in _sentences(v) if prop & _kind_words(s) and len(s) <= 220]
        return " ".join(dict.fromkeys(lines))[:300]

    def ref_line(self, need: str) -> str:
        """A requirement whose value was read from another listing's page, as the ledger shows it."""
        prop, _, rest = need.partition(":")
        value = self.ref_values[need]
        notes = self.ref_notes.get(need)
        return (f"{prop.strip()}: as worded below, against {value}, the {prop.strip()} on the page it points at"
                + (f" (that page says: {notes})" if notes else "")
                + f" [the requirement: '{rest.strip()[:160]}'; the shop and the other values in it only pick out that other listing and are no requirements on this item]")

    def code_to_id(self, value: str) -> tuple[str, str]:
        """A code passed where a listing id belongs: the one seen listing whose own tag or name carries it. A listing
        that only refers to the code (sends the reader to it) is not that listing."""
        token = value.strip().lower()
        pat = r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])"
        owners, mentions = [], []
        for i, row in self.rows.items():
            texts = [(str(k).lower(), v.lower()) for k, v in row.items() if k != self.id_key and isinstance(v, str)]
            if any(re.search(r"[\[(]\s*" + re.escape(token) + r"\s*[\])]", v) or (re.search(r"name|title", k) and re.search(pat, v)) for k, v in texts):
                owners.append(i)
            elif any(re.search(pat, v) for _, v in texts):
                mentions.append(i)
        if len(owners) == 1:
            return owners[0], ""
        if owners:
            return value, f"{value!r} is a code that several listings carry ({', '.join(owners[:6])}), not a listing id; pass one listing id"
        if mentions:
            return value, f"{value!r} is a code that {', '.join(mentions[:3])} refers to, not a listing id: search for {value!r} and pass the id of the listing that carries it"
        ids = list(self.rows)[:20]
        head = ids[0][:2] if ids else ""
        if head and all(i.startswith(head) for i in ids) and not token.startswith(head.lower()):
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)"
        return value, ""

    # ---------- ending ----------
    def cart_full(self) -> bool:
        """The cart holds a line of each requested item's kind (by the listings' category labels)."""
        held = {self.item_of(i) for i in self.cart_ids() if not self.sold_out(i)}
        return bool(held) and all(n in held for n in range(1, len(self.items) + 1))

    def last_set(self) -> list[str]:
        """The cheapest full set among the listings seen: per item, its kind's cheapest buyable listing whose attributes
        were read (fitting the set's anchor where an item must fit one), within the budget; [] when there is none."""
        anchors: dict[int, str | None] = {}
        picks: dict[int, str] = {}
        pairs = self.fit_pairs()
        parts = {item for item, _, _ in pairs}
        if any(o in parts for _, o, _ in pairs):
            return []
        for number in range(1, len(self.items) + 1):
            if number in parts:
                continue
            ids = [i for i in self.seen_order if self.item_of(i) == number and self.buyable(i) and i in self.full and self.price_of(i) is not None]
            if not ids:
                return []
            picks[number] = min(ids, key=lambda i: self.price_of(i) or 0)
            anchors[number] = picks[number]
        for item, other, named in pairs:
            anchor = anchors.get(other) if other else None
            if self.fit_roles:
                # with the codes read, only a part that fits counts: a set with a part that does not fit is no set
                good = [i for i in self.fit_candidates(item, anchor, named) if i in self.full and self.price_of(i) is not None]
            else:
                good = sorted((i for i in self.seen_order if self.item_of(i) == item and self.buyable(i) and i in self.full and self.price_of(i) is not None),
                              key=lambda i: self.price_of(i) or 0)
            if not good:
                return []
            picks[item] = good[0]
        total = sum((self.price_of(i) or 0) * self.items[n - 1].qty for n, i in picks.items())
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget:
            return []
        return [picks[n] for n in sorted(picks)]

    def restore_built(self) -> bool:
        """A forced finish holding a cart without a line for each item: put back the cart the agent last sent to checkout,
        when it held a line for each item and the agent did not choose to end without buying since; True when it is back."""
        full = {i: n for i, n in self.sent_cart.items() if not self.sold_out(i)}
        cart = self.roles.get("cart")
        if not cart or len(full) < len(self.items) or self.refuse_step > self.buy_step:
            return False
        calls = self.cart_calls(full, [i for i in self.cart_ids() if i not in full])
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
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        self.trace.append({"kind": "restored", "lines": full})
        return True

    def link_misses(self, need: str, ident: str) -> list[str]:
        """Values the requirement uses to pick out the listing it points at that the opened, read record `ident` lacks,
        counting only words that some listing in this store shows as a value (an attribute, or the seller's name)."""
        record = self.full.get(ident) or {}
        if ident not in self.opened or _fact_block(record) is None:
            return []
        words = lambda text: set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text)).lower()))
        prop, _, rest = need.partition(":")
        named = words(rest)
        said = {w for w in named - words(prop) if (len(w) >= 3 or w.isdigit()) and w in _kind_words(w)}
        values: set[str] = set()
        keys: set[str] = set()
        for other in self.seen_order:
            seen = self.full.get(other) or self.rows.get(other) or {}
            block = _fact_block(seen) or {}
            for key, value in block.items():
                keys |= words(key)
                # a value picks the listing out only when the requirement names its property too
                if {w for w in words(key) if len(w) > 2 and w in _kind_words(w)} & named:
                    values |= words(value)
            for key, value in seen.items():
                if isinstance(value, str) and re.search(r"shop|seller|brand|maker|vendor|store", str(key), re.I):
                    values |= words(value)
        return sorted((said & values) - keys - words(_dump(record)) - words(_dump(self.rows.get(ident))))

    def order_tie(self, number: int, text: str) -> bool:
        """A must that only ties this item to another item of this same order is settled by fit, not followed as a pointer."""
        prop, _, value = str(text).partition(":")
        if not re.search(r"\b(?:fits?|fitting|compatib\w*|works? with|goes with)\b", prop + " " + value[:80], re.I):
            return False
        if re.search(r"\bpage\b|listing|\bsold by\b|\bsells?\b|\bseller\b|\bshop\b|\bfrom [A-Z]\w*'s\b", value):
            return False
        return bool([n for n in self.items_named(value) if n != number])

    def reroute(self, heard: dict[str, Any], batch: list[tuple[int, str, str]]) -> dict[str, Any]:
        """A need goes to the item the answer speaks of: when the answer holding its value names, by words only one
        item's name carries, exactly one other item than the question asked about, the need moves there."""
        if len(self.items) < 2:
            return heard
        names = [_kind_words(item.name) for item in self.items]
        own = [{w for w in ws if len(w) > 2 and not any(_same_word(w, o) for j, other in enumerate(names) if j != i for o in other)}
               for i, ws in enumerate(names)]
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or entry.get("points") is True:
                continue
            value = {w for w in _kind_words(str(entry.get("need") or "").split(":", 1)[-1]) if len(w) > 2}
            if not value:
                continue
            at = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            for number, _, answer in batch:
                said = _kind_words(answer)
                if number != at or not all(any(_same_word(v, w) for w in said) for v in value):
                    continue
                named = [n for n, ws in enumerate(own, 1) if any(_same_word(w, s) for w in ws for s in said)]
                if len(named) == 1 and named[0] != at:
                    self.trace.append({"kind": "reroute", "need": entry.get("need"), "from": at, "to": named[0]})
                    entry["item"] = named[0]
                break
        return heard

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
        # room for the light model's thinking: a reply cut at its limit would send the brief to the heavy one
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
            if count == 1:
                # the request writes a count for this product ("2 <products> sold as a pair"): that count is the quantity
                said = _said_count(self.request, str(entry.get("name") or ""), [str(e.get("name") or "") for e in _items(brief.get("items")) if isinstance(e, dict) and e is not entry])
                # (not a number that belongs to a stated value, such as a size or capacity in its musts)
                stated = " ".join(str(m) for m in _items(entry.get("musts")))
                count = said if said and said > 1 and not re.search(rf"(?<![\d.]){said}(?![\d.])", stated) else count
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
            if text and (number, text) not in self.refs and not self.order_tie(number, text):
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

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, self.meter.check_models(self.cost), 500) or {}
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
            needs = [self.ref_line(m) if m in self.ref_values else m for m in item.musts + item.told if m not in item.dropped]
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
            parts.append(f"wish (prefer a fitting listing that meets it; never a reason to reject one): {wish}")
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
        if (self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items)
                or self.cost + 0.004 * len(self.items) > SPEND_HARD - REPAIR_ROOM or self.meter.elapsed() > EXPLORE_UNTIL - 60 - 10 * len(self.items)):
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
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and not self.fit_rejects(i, cart)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

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
        count = self.msgs_free()
        if count <= 0 or not self.meter.asking(self.cost):
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
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, self.meter.check_models(self.cost), 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5 or self.fit_settled(number, label) or self.is_settled(number, label):
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # all messages but the last first; the last re-asks a question whose answer wanted a single question
        # (named with example values this time), else it goes to the next pick
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        # the heavy model reads a batch holding a long answer (one that points at another listing); short ones the light model reads
        heard = self.hear(batch, (HEAVY, LIGHT) if any(len(str(a).split()) > 35 for _, _, a in batch) else (LIGHT, HEAVY))
        self.take_heard(self.reroute(heard, batch), told)
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
            if again:
                batch = self.send([again])
                if batch:
                    sent += batch
                    # one answer: the light model reads it
                    self.take_heard(self.reroute(self.hear(batch, (LIGHT, HEAVY)), batch), told)
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary + self.early_trail()

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
            if entry.get("points") is True and (number, need) not in self.refs and not self.order_tie(number, need):
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
            got = self.ask_json(prompt, self.meter.check_models(self.cost) if nudged else [LIGHT, HEAVY], 2000)
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
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_HARD) -> str:
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
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids + [i for i in self.seen_order if self.fit_rejects(i, ids)], 15)
        got = self.ask_json(prompt, self.meter.check_models(self.cost), 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b)) and not self.fit_rejects(str(b), ids)][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self, early: bool = False) -> tuple[str, bool]:
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
        while pending and (self.early_rounds < EARLY_ROUNDS if early else self.trails < 5) and self.meter.can_call(self.cost) and self.meter.elapsed() < EXPLORE_UNTIL:
            if early:
                self.early_rounds += 1
            else:
                self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                elif ident in self.full:
                    records.append("READ " + _dump(_clip(self.full[ident], 200))[:1200])
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
                # the description of each link decides which seen listing it is (a near twin from one shop is not it)
                ids = [self.best_link(str(l.get("about") or ""), str(l.get("id") or "").strip()) if str(l.get("id") or "").strip() else "" for l in links]
                named = str(links[0].get("id") or "").strip() if links else ""
                if ids and ids[0] and ids[0] != named and self.link_misses(need, ids[0]) and not self.link_misses(need, named):
                    # a word-count pick that lacks a value the requirement names does not replace the listing the model named
                    ids[0] = named
                miss = self.link_misses(need, ids[0]) if ids and ids[0] else []
                if miss:
                    # a near twin read in full that lacks a value the requirement names is not the first link
                    self.trace.append({"kind": "link_miss", "id": ids[0], "missing": miss})
                    links = [{"id": "", "about": f"the listing the requirement describes, one showing {', '.join(miss)} ({ids[0]} does not)"}]
                    ids = [""]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    notes = self.page_notes(ids[-1], need) if ids[-1] else ""
                    if notes:
                        self.ref_notes[need] = notes
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
        if early and lost:
            text += "Not among the listings seen yet: " + " | ".join(lost[:2]) + ". Search for it by its shop, brand or words and open its page.\n"
        hold = not early and bool(lost) and self.trail_blocks < 2 and self.left() > 5
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
        if self.cost > SPEND_HARD:
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
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.meter.check_models(self.cost), 1200)
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
        # only the heavy model answers this review: the light one sends back an empty reply that is billed all the same
        if self.audited or self.cost > SPEND_HARD or HEAVY not in self.meter.check_models(self.cost):
            return ""
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
        # by item in turn (listings that fit and were read first, then the cheaper), so an early good row is not
        # pushed out of the window by many newer ones
        lanes = []
        for number in range(1, len(self.items) + 1):
            lane = [i for i in reversed(self.seen_order) if self.item_of(i) == number and not self.sold_out(i)]
            lane.sort(key=lambda i: (self.fit_rejects(i, []), i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
            lanes.append(lane)
        order: list[str] = []
        while any(lanes) and len(order) < 30:
            for lane in lanes:
                if lane and len(order) < 30:
                    order.append(lane.pop(0))
        order += [i for i in reversed(self.seen_order) if i not in order and not self.sold_out(i)]
        records = []
        opened = 0
        for ident in order:
            record = self.full.get(ident) if ident in self.full and opened < 14 else None
            opened += 1 if record else 0
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (by item, read ones first):\n" + "\n".join(records)
        if HEAVY not in self.meter.check_models(self.cost):
            return None
        got = self.ask_json(prompt, [HEAVY], 1500)
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
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one);
        # a value that is not a seen id but a code that seen listings carry is checked on any listing tool
        if len(free) != 1:
            return args, ""
        if name != self.roles.get("cart") and (name, free[0]) not in self.id_args:
            value = args.get(free[0])
            if isinstance(value, str) and value.strip() not in self.rows and _id_like(value, True) and self.rows:
                fixed, problem = self.code_to_id(value)
                if problem:
                    return args, problem
                return {**args, free[0]: fixed}, ""
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            if isinstance(value, str) and _id_like(value, True):
                return self.code_to_id(value)
            if not isinstance(value, str) or len(value.strip()) < 8:
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

    def rearm_after_refused_end(self) -> None:
        """An ending without a purchase turned down (a change came with it): the episode goes on, its checks re-armed
        and room kept to act on the change."""
        self.refused_at = self.step
        self.meter.open_repair(self.cost)
        self.audited = False
        self.pushed = False
        self.fit_decline_held = False
        self.last_checked = False

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
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
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
            # every ending the agent sends counts, held back or not: a forced finish follows its last intent
            if self.is_buy(name, args):
                self.buy_step = self.step
                self.sent_cart = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in self.dropped}
            elif self.is_refuse(name, args):
                self.refuse_step = self.step
            if self.is_buy(name, args) and self.bare_buys.get(self.cart_mark(), 0) >= 2:
                out.append((cid, "Not run (no step used): the store turned down this checkout twice with the cart as it is now, without saying why. Look at the cart, put a line in it for each item at its requested quantity (replace any line the store cannot sell), then check out."))
                continue
            if self.is_buy(name, args) and self.cart_touched and not self.cart_ids() and self.left() >= 2 and not self.empty_held:
                # once only: when Kyo's record of the cart is wrong, the next checkout goes through
                self.empty_held = True
                out.append((cid, "Not run (no step used): the cart is empty, so a checkout would order nothing. Add the chosen listing of each item at its requested quantity first, then check out."))
                continue
            readers_now = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
            if (not ending and self.buys == 0 and len(self.items) == 1 and name in readers_now and not self.cart_ids() and self.left() <= 8
                    and self.full and self.roles.get("buy") and not self.late_cart_told):
                self.late_cart_told = True
                out.append((cid, f"Not run (no step used): only {self.left()} steps are left and the cart is empty. Put the best fitting listing you have read in the cart now and check out: a refused checkout (a line sold out, a changed price or budget) still needs steps to repair. Search again only if no listing you read fits."))
                continue
            if (not ending and self.buys == 0 and self.held_step != self.step and name not in (self.roles.get("cart"), self.roles.get("cart_view"))
                    and self.left() <= self.tail() + 3 + len(self.items) and self.roles.get("buy") and self.cart_full()):
                out.append((cid, f"Not run (no step used): only {self.left()} steps are left and the cart holds a line for every item. Check out now: a refused checkout (a line sold out, a changed price or budget) still needs steps to repair."))
                continue
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
                    chain, hold = self.trail()
                    if hold:
                        # a hold asks for a read: the next one goes through even near the end
                        self.held_step = self.step
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    fitted = self.fit_check()
                    if fitted:
                        self.held_step = self.step
                        out.append((cid, "Not run yet (no step used). " + chain + fitted + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        self.held_step = self.step
                        self.meter.open_repair(self.cost)
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5:
                        if self.may_need_fit() and not self.fit_decline_held:
                            self.fit_learn()
                        sets = self.fit_sets() if not self.fit_decline_held else ""
                        if sets:
                            self.fit_decline_held = True
                            self.trace.append({"kind": "fit_decline_held", "sets": sets})
                            out.append((cid, "Not run yet (no step used). By the codes the listings carry, these whole sets fit together inside the budget: " + sets + ". Open them to check every other requirement, cart the set that meets everything and check out; end without a purchase only if each set fails a requirement."))
                            continue
                        review = self.audit()
                        if review or self.leads:
                            self.meter.open_repair(self.cost)
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        planned = self.last_set() if not self.leads and not self.last_checked and self.left() > 4 + len(self.items) else []
                        if planned:
                            self.last_checked = True
                            self.check_cart(planned, SPEND_HARD + REPAIR_ROOM)
                            if self.verdicts.get(self.check_key(planned)):
                                self.trace.append({"kind": "last_set", "ids": planned})
                                out.append((cid, "Not run yet (no step used). An independent check of these listings' records says that, together, they meet every requirement: "
                                            + ", ".join(f"item {self.item_of(i)}: {i}" for i in planned) + ". Put each in the cart at its requested quantity and check out; end without a purchase only if one of their records clearly breaks a requirement."))
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
            # a push to search more needs money and time to search with, else it only runs the loop into its limit
            if (self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3
                    and self.cost < SPEND_HARD - REPAIR_ROOM and self.meter.elapsed() < EXPLORE_UNTIL - 60):
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
                    if len(lines) > len(self.built) or set(lines) != set(self.built):
                        self.built_step = self.step
                    self.built = lines
                    self.built_counts = {i: self.cart_lines.get(i, 1) for i in lines}
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
                    self.meter.open_repair(self.cost)
                    self.code_waits = min(self.code_waits, CODE_WAITS - 1)
                    self.fit_decline_held = False
                    if not self.loose_lines(observation) and not re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}", _dump(observation)):
                        # turned down with no note at all (an error): counted per cart, so the same cart is not resent forever
                        self.bare_buys[self.cart_mark()] = self.bare_buys.get(self.cart_mark(), 0) + 1
            elif self.is_refuse(name, args) and not reply.get("finished"):
                self.rearm_after_refused_end()
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
        # a cart without a line for each item scores nothing: the full cart the agent last sent to checkout goes back,
        # else the episode ends without a purchase
        now = set(self.cart_ids() if fresh else list(self.cart_lines))
        short = (len(now) < len(self.items) and not self.cart_full()
                 and not (now and self.buy_step > self.refuse_step and set(self.sent_cart) == now))
        if buy and short and self.restore_built():
            short, holding = False, True
        elif not holding:
            # an empty cart that could not be restored keeps the old blind checkout
            short = False
        if buy and not short and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
                    self.rearm_after_refused_end()
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= HISTORY_LIMIT:
            return
        # oldest results first, well under the limit, each once: the cached start of the history then changes only now and then
        target = int(0.75 * HISTORY_LIMIT)
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[:-4]:
            if size <= target:
                break
            content = self.chat[index].get("content") or ""
            if isinstance(content, str) and len(content) > 600 and "...(older result shortened" not in content:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                short = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"
                size -= len(_dump(content)) - len(_dump(short))
                self.chat[index]["content"] = short

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
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.meter.elapsed() < CALLS_UNTIL - 5:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                # the repair of a refused ending gets its own room, whatever the task spent before
                self.work(self.loop_calls + 10, self.meter.open_repair(self.cost, late=True))
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.meter.exploring(self.cost, spend_cap):
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
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2 or not self.meter.can_call(self.cost):
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
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and self.meter.asking(self.cost) and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
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
