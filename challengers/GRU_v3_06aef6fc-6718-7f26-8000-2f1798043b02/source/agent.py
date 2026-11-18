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
# per-episode spend: the race key holds USD 5 for all 90 episodes, and the store's shopper spends from it too. These limits bind a hard
# episode, not the average: agents with the same limits ran USD 3.8-4.6 a run
LIST_WIDE = 8               # properties of one item a message may name at most
LIST_STEPS = (8, 2, 2)      # names in an item's 1st, 2nd and later message: wide first, then, since a wide list passed over in silence mostly means a shopper who reads no lists, the likeliest names again in pairs
SPEND_SOFT = 0.043
SPEND_HARD = 0.078
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
ROW_CHARS = 160             # a field of a row in a list of results: the page holds the rest, and old results fill the turn model's context
PLAN_NAMES = 4              # names the planner picks for an item: only the head of a list needs its judgement, the keys fill the rest
OBS_LIMIT = 15000
HISTORY_LIMIT = 90000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
PAGE_KEY = r"(?i)page|next|token|cursor|offset|skip|begin|start|after"
BROWSE_MODE = r"(?i)brows|aisle|shelf|group|section|categor|list|filter|match"
SCAN_KEEP = 16              # steps a code scan of the searches leaves the turn model, besides the finish
SCAN_SECONDS = 15.0         # the scan's share of the clock, once and again for each item, at most 60 s; a third of it is kept for reading rows
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
# the sandbox ends an episode at 300 s whatever it is doing; past CLOCK_STOP no new model turn starts, and the store
# calls that finish the episode (no model needed) get the rest
CLOCK_STOP = 255.0
CLOCK_LATE = 230.0          # past this, a failed model call is not waited on and resent
CLOCK_WARN = 200.0          # past this, the turn model is told the time, not only the steps, is running out

# words that make a requirement a numeric limit ("at least 18.5 inches", "1.35 v or less")
LIMIT_WORDS = (r"at least|no less than|not less than|minimum(?: of)?|or more|and up|and above|at most|no more than|not more than|"
               r"up to|maximum(?: of)?|or less|or under|and under|and below|under|below|over|above|less than|more than")

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Lines in a tool result that begin with [Kyo], and results that begin "Not run yet", come from your own helper code, not from the store: follow them.
- A listing can carry one attribute twice under different spellings (spaces, underscores, capitals): a value under either counts.
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma, slash or "or" list is enough ("8 gb or 16 gb" states 16 gb). Values match as phrases: case, singular or plural and plain synonyms are fine, and a value that contains the required value as a phrase counts, its words together and in order ("deep teal" meets teal, "extra roomy fit" meets roomy); the same words split up by other words do not ("oak and walnut brown" does not meet oak brown). A required value written as one of the store's own labels for that property (such as "100 and up") is met by that label, not by a bare figure it would cover. A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- A value that says the same thing in other words meets the requirement ("roomy" for "spacious", "quick to fit" for "easy to fit", "slight" for "small"), above all for what the shopper told you in their own words: end without a purchase only when no listing means the same. A figure, or another of the labels listings use for that property, is not the same thing ("under an hour" does not say "quick"; "medium" does not say "small"), but a value short of a word that adds nothing is that value ("light assembly" says "light assembly needed").
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it means something else, the listing fails whatever other keys or the title say. A key spelled exactly as the property decides alone: longer keys that hold the same word ("size" against "bed_size_2") then say nothing, even when they list the wanted value. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words, and search any part number or code a note gives, since a code is no listing id; it is the one whose record shows EVERY detail given about it, so a near match such as another form of the product is a different listing: go through that shop's listings until one shows them all), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. When the requirement says the value goes with or pairs with that listing's value, the wanted value is the partner a page names for it ("chalk goes together with slate" with chalk on the last page means slate is wanted, not chalk). The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- A listing a requirement points at is a candidate like any other: when it meets every requirement, it can be the one to buy.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Copy a value written like a store's range label ("100 and up", "1-500", "2 gb & under") exactly as written: it is a value, not a limit to restate. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. Keep the request's own relation in such a must: "goes with" (the partner of that listing's value) and "other than" are not "same as", so write them as the request does. A rule that turns on what else is bought ("black if the laptop has 8 gb", "clear, except when the tablet is the 10 inch one") points at no listing: it is a must, not a lookup.
"fits": one entry for each sentence saying a requested product must fit, suit, be compatible with or work with another requested product or with a device model the request names, each {"part": the item's number in "items", counting from 1, "with": the number of the item it must fit, or null, "model": the device's model code as written, or ""}; else [].
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
Where a listed key of that item can carry the hint, name the hint by that key, since the shopper recognises a property only by its name; a name no listing of the item carries comes after the three keys most of its listings carry. Otherwise name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about what comes in the box or how many arrive (a box holding something other than expected, a delivery that gets counted) at its pack size or piece count, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, its colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


NAME_MUSTS = """A shopping request names each product with words that can also describe it ("3 blazers for business wear", "a sleeve in pink"). For each product below, give the requirements its name's describing words state that its musts do not already hold, each as "property: value" with the property a store would list it under ("for business wear" gives "occasion: business"). Return ONE JSON object and nothing else: {"musts": [{"item": item number, "must": "property: value"}]} (an empty list when the name adds nothing)."""
NAME_WORDS = r"\b(?:for|in|with|without)\s+([a-z0-9][a-z0-9 ,'&/-]*)"
LOOK_KEY = r"shape|pattern|decorat|print|design"
LOOKS_KYO = "a hint about its looks at colour or finish (whichever the request leaves open)."
LOOKS_KEYED = ("a hint about its looks at a look property among the item's listed keys (its shape, pattern, decoration or"
               " colour) that the request leaves open, the one most of the item's listings carry first.")


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike, near or going with the one on some shop's product, that product picked out by a few of its properties; keep that relation word in the need): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value. A figure of speech after the value ("snug as a bug", "and that is my last word") is not part of it.
- An answer that names a product ("the handle of the teapots") is about that item, whichever item the question asked about.
- Name the property with the answer's own words even when the question listed a similar name. Copy only the value: words that merely link the property to its value are not part of it.
- An answer that throws out everything apart from one value ("anything apart from X goes back", "X and nothing else") wants that value: the need is "property: X". A wanted value that itself begins with "other than", "not", "anything but" or "unlike" is an exclusion: copy it word for word, keeping that word.
- A sentence that only names a sum of money (what the shopper will now spend) is the new "budget": the number exactly as that sentence states it, never converted to another unit ("1500 cents" gives 1500). It is never a value of the property the question asked about."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit (when the request names none, the number as the note writes it: "1500 cents" gives 1500), or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


UNITS_ASK = """Each line below holds two unit words taken from a shopper's request and a store's listings, joined by "->". For each line say how many of the second make one of the first, as a plain number, or null when the two do not measure the same thing or a word is not a unit. Return ONE JSON object and nothing else: {"factors": [one number or null for each line, in order]}"""

CAPS = """A shopper's request may limit what one product may cost, apart from the budget for the whole order. Read the request and return ONE JSON object and nothing else: {"caps": [{"item": item number, "cap": the most one unit of that product may cost, as a number in the request's own unit}]} (an empty list when no single product has its own limit; the whole order's budget is not one)."""

CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, and keys differing only in spaces, underscores or capitals are one key, a value under either counts; a value meaning the same in other words or short of a word that adds nothing counts, a figure or another of the labels used for that property does not; one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value as a phrase counts, its words together and in order, such as "deep teal" for teal but not "oak and walnut brown" for oak brown, units converted, limits include the edge, a range must lie wholly inside, and a required value written as one of the store's own labels for that property, such as "100 and up", is met by that label, not by a bare figure it would cover); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key spelled exactly as the property decides alone, over longer keys holding the same word; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["a concrete problem in at most 15 words", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (a key spelled exactly as the property decides alone, over longer keys holding the same word; any key spelling, and keys differing only in spaces, underscores or capitals are one key, a value under either counts; a value meaning the same in other words or short of a word that adds nothing counts, a figure or another of the labels used for that property does not; one element of a list is enough, synonyms and unit conversions count, a value containing the required value as a phrase counts, its words together and in order, such as "deep teal" for teal but not "oak and walnut brown" for oak brown, limits include the edge, and a required value written as one of the store's own labels for that property, such as "100 and up", is met by that label, not by a bare figure it would cover); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown. A link is the listing whose record shows EVERY detail given about it: one that differs on any of them (another form, colour or size of the product) is a different listing, so give that link id "" and done false. When the requirement asks for the kind that works alongside, goes with or pairs with what the last page has, its "value" is the partner that page names for its own value ("its X goes together with Y" gives Y), not the page's own value."""

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


def _conditional(value: str) -> bool:
    """A rule that turns on another line of the order rather than on the listing itself."""
    return re.search(r"(?i)\b(if|when|unless|should|in case)\b", value) is not None


def _clip_reply(value: Any, limit: int, depth: int = 0, whole: bool = False, paged: bool = False) -> Any:
    """_clip for a store reply: a text standing alone in a list or a one-key reply may be a record no reader opened, so it keeps more."""
    if depth > 7:
        return "..."
    if isinstance(value, str):
        room = 4 * limit if whole else limit
        return value if len(value) <= room else value[:room] + "..."
    if isinstance(value, dict):
        results = any(re.search(r"(?i)count|total|next|page|more", k) for k in value)
        return {k: _clip_reply(v, limit, depth + 1, len(value) == 1, results) for k, v in value.items()}
    if isinstance(value, list):
        rows = paged and len(value) >= 3 and all(isinstance(v, dict) for v in value)
        return [_clip_reply(v, min(limit, ROW_CHARS) if rows else limit, depth + 1, True) for v in value]
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


def _plain_value(text: str) -> Any:
    """A field written out as text, as the value it stands for: a number, a yes/no, nothing, or the text itself."""
    text = text.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if re.fullmatch(r"-?(0|[1-9]\d*)", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def _whole_in(part: str, text: str) -> bool:
    """Whether one value stands whole inside another: 5 is not inside 15."""
    return re.search(r"(?<![a-z0-9])" + re.escape(part.lower()) + r"(?![a-z0-9])", text.lower()) is not None


def _pairs_text(text: str) -> dict[str, Any] | None:
    """Attributes written as 'a=b; c=d' or 'a: b; c: d' as a dict; None for any other text (a value may itself hold semicolons)."""
    sign = "="
    if "=" not in text or not re.match(r"\s*[^=;|]{1,64}=", text):
        parts = [p for p in re.split(r";\s*", text.strip()) if p.strip()]
        if len(parts) < 3 or not all(re.match(r"\s*[A-Za-z][A-Za-z0-9_ /-]{0,40}:(\s|$)", p) for p in parts):
            return None
        sign = ":"
    pairs: dict[str, str] = {}
    spellings: dict[str, str] = {}
    for part in re.split(r";\s*(?=[^;" + sign + r"|]{1,64}" + sign + ")", text):
        key, mark, value = part.partition(sign)
        if mark and key.strip():
            # one attribute under two spellings is one key, holding what either says
            name = spellings.setdefault(re.sub(r"[^a-z0-9]", "", key.lower()) or key.strip(), key.strip())
            value = value.strip()
            have = pairs.get(name)
            if have is None or _whole_in(have, value):
                pairs[name] = value
            elif not _whole_in(value, have):
                pairs[name] = have + ", " + value
    if len(pairs) == 1 and not re.fullmatch(r"[a-z][a-z0-9_ ]{0,30}", next(iter(pairs))):
        return None
    return pairs or None


def _keyed_part(part: str) -> tuple[str, str] | None:
    """'key: value' as its two sides; a capitalised word in a title ("FREE: earbuds") is no key."""
    found = re.match(r"\s*([A-Za-z][A-Za-z0-9_ -]{0,30}?)\s*[:=](?:\s*(.*))$", part, re.S)
    if not found or (found.group(1).isupper() and len(found.group(1)) > 3):
        return None
    return found.group(1), found.group(2) or ""


def _text_record(text: str) -> dict[str, Any] | None:
    """A record the store wrote out as one text, 'key: value | key: value ...' (or a JSON object), as a dict, so that code can read
    it and so that clipping cuts each field, not the whole record; None for any other text. A part with no key belongs to the
    value before it: titles hold the separator too."""
    text = text.strip()
    if text[:1] == "{":
        try:
            found = json.loads(text)
            return found if isinstance(found, dict) and len(found) >= 3 else None
        except ValueError:
            return None
    text = re.sub(r"\*\*([^*\n]{1,40}?)\*\*", r"\1", text)
    for mark in (" | ", " || ", "\n", "\t", " ; ", " \u2022 ", " \u00b7 ", ", "):
        parts = text.split(mark)
        if len(parts) < 3 or not _keyed_part(parts[0]):
            continue
        fields: list[list[str]] = []
        for part in parts:
            keyed = _keyed_part(part)
            if keyed and keyed[0] not in [key for key, _ in fields]:
                fields.append(list(keyed))
            else:
                fields[-1][1] += mark + part
        if len(fields) < (5 if mark == ", " else 3):      # (prose holds commas and the odd colon, a record many keys)
            continue
        record: dict[str, Any] = {}
        for key, value in fields:
            pairs = _pairs_text(value)
            record[key] = pairs if pairs else _plain_value(value)
        return record
    pairs = _pairs_text(text)
    return {key: _plain_value(value) for key, value in pairs.items()} if pairs and len(pairs) >= 4 else None


def _open_records(node: Any, depth: int = 0) -> Any:
    """A store reply with every record written out as text read into a dict; a reply that only wraps one record becomes that
    record. Anything else is left as it came, so dict replies are untouched."""
    if depth > 6:
        return node
    if isinstance(node, str):
        record = _text_record(node) if len(node) >= 20 else None
        return record if record is not None else node
    if isinstance(node, list):
        return [_open_records(value, depth + 1) for value in node]
    if isinstance(node, dict):
        opened = {key: value if isinstance(value, str) and len(node) >= 5 else _open_records(value, depth + 1) for key, value in node.items()}
        if len(opened) == 1:
            only = next(iter(opened.values()))
            if isinstance(only, dict) and not isinstance(next(iter(node.values())), dict):
                return only
        return opened
    return node


def _device_names(text: str) -> set[str]:
    """What a device's listing calls it: its model number and the names it is also sold under, upper case."""
    names = {found.group(1).upper() for found in re.finditer(r"(?i)\bmodel\s*(?:number|no\.?|#)\s*:?\s*([a-z0-9][a-z0-9-]{2,})", text)}
    names.update(found.group(1).strip().upper() for found in re.finditer(r"(?i)\bmodel\s*:\s*([a-z0-9][a-z0-9 -]{2,18}?)\s*(?:[.;|\"]|$)", text))
    for found in re.finditer(r"(?i)\balso sold as\s+([^.;|\"]+)", text):
        names.update(part.strip().upper() for part in re.split(r",|\band\b", found.group(1)) if part.strip())
    return names


def _fit_rule(text: str) -> tuple[list[str], set[str]] | None:
    """A part's 'Fits: ...' or 'Compatible with: ...' sentence as (patterns, exceptions), upper case; a pattern's trailing x's stand for
    any character. 'Part X also fits: ...' speaks of another part, so it is no rule of this one."""
    found = re.search(r"(?i)(?<!also )\b(?:fits|compatible with|works with)\s*:\s*([^.;|\"]+)", text)
    if not found:
        # other words for the same note: a label, then nothing but names shaped like model codes
        for note in re.finditer(r"([A-Za-z][A-Za-z ]{2,30}):\s*([^.;|\"]+)", text):
            names = [n.strip() for n in re.split(r",|\band\b", note.group(2).partition(", except ")[0]) if n.strip()]
            other = re.search(r"(?i)\bpart\s+\S+\s*$", text[: note.start()])     # (a note about another part is no rule of this one)
            if names and not other and not re.search(r"(?i)\b(part|model)\b", note.group(1)) \
                    and all(re.fullmatch(r"(?:[A-Z]{2,}[ -]?\d[0-9A-Za-z]*|[A-Z]{2,}\d*x+)", n) for n in names):
                found = re.search(re.escape(note.group(1)) + r":\s*([^.;|\"]+)", text)
                break
    if not found:
        return None
    body, _, skipped = found.group(1).partition(", except ")

    def names(part: str) -> list[str]:
        cleaned = (re.sub(r"(?i)\b(every|any|all|models?)\b", " ", piece).strip() for piece in re.split(r",|\band\b", part))
        return [re.sub(r"\s+", " ", piece) for piece in cleaned if piece]

    patterns = [re.sub(r"X+$", lambda m: m.group(0).lower(), name.upper()) for name in names(body)]
    return (patterns, {name.upper() for name in names(skipped)}) if patterns else None


def _rule_covers(rule: tuple[list[str], set[str]], names: set[str]) -> bool:
    """Whether a part's fit rule takes a device known by these names."""
    patterns, skipped = rule
    if names & skipped:
        return False
    return any(name == pattern or (pattern.endswith("x") and name.startswith(pattern.rstrip("x")) and len(name) == len(pattern))
               for pattern in patterns for name in names)


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


def _unit(word: str) -> str:
    """A unit word as written, singular and lower case ("Inches" and "inch" give "inch"); no table of units: two figures
    are compared only when they are written in the same unit."""
    word = word.lower().strip()
    if "/" in word or len(word) <= 2:
        return word
    return word[:-2] if re.search(r"(ch|sh|x)es$", word) else word[:-1] if word.endswith("s") else word


def _amounts(text: Any) -> list[tuple[str, float]]:
    """Each number in the text with the unit word written after it, as (unit, number); ("", number) without one."""
    low = re.sub(r"(\d),(\d{3})\b", r"\1\2", str(text).lower())
    low = re.sub("(\\d)\\s*[\"\u201d\u2033]", r"\1 inch", low)
    return [(_unit(unit), float(number)) for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+(?:/[a-z]+)?)?", low)]


def _fixed_head(text: str) -> str:
    """The part of a prompt before its first capitalised section heading ("\\n\\nREQUEST:\\n"), when it is long enough to be cached."""
    found = re.search(r"\n\n[A-Z][A-Z ]{2,}[^\n]{0,80}:\n", text)
    return text[: found.start()] if found and found.start() >= 1800 else ""


def _as_written(amount: float, text: str) -> float:
    """A sum as the text writes it: a reader can turn "25500 cents" into 255, and the store's prices are in the written unit."""
    figures = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))}
    return round(amount * 100, 2) if amount not in figures and round(amount * 100, 2) in figures else amount


def _details(text: str) -> list[str]:
    """The values that pick one listing out in a description of it ("... listing with color black and folded length 52.07 cm" gives
    black and 52.07): the figure of each detail, else its last word."""
    tail = re.split(r"(?i)\blisting\b", str(text))[-1]
    found = []
    for bit in re.split(r"(?i)\band\b|,|\bwith\b|\bwhose\b", tail):
        words = re.findall(r"[A-Za-z0-9.]+", bit)
        figure = [w for w in words if re.fullmatch(r"\d+(?:\.\d+)?", w)]
        if figure or len(words) >= 2:
            found.append(figure[-1] if figure else words[-1])
    return found[:3]


def _cents_said(text: Any) -> float | None:
    """The last sum written in cents in a text ("a new limit of 1500 cents" gives 1500), else None."""
    found = re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*cents?\b", str(text).lower())
    return float(found[-1].replace(",", "")) if found else None


def _shared_words(values: list[str]) -> bool:
    """Some word of three or more letters appears in two or more of the different values."""
    seen: dict[str, int] = {}
    for value in set(values):
        for word in set(re.findall(r"[a-z]{3,}", value.lower())):
            seen[word] = seen.get(word, 0) + 1
    return any(count >= 2 for count in seen.values())


def _phrase_in(value: str, text: str) -> bool:
    """The value's words appear together and in order in the text (case, a plural s and the joins between words aside):
    "deep teal" has teal and "8gb ram" has 8 gb, but "oak and walnut brown" does not have oak brown."""
    words = [w for w in re.findall(r"\d+(?:\.\d+)?|[a-z.]+", value.lower()) if w.strip(".")]
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"[\s\-_/&+,]*".join(re.escape(w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w) + r"(?:es|s)?" for w in words) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _own_keys(prop: str, block: dict[str, Any]) -> dict[str, Any]:
    """A record's attributes for a property: the key spelled as the property decides alone, longer keys holding its word only without one."""
    keys = {k: v for k, v in block.items() if _names(_stem(prop), _stem(k))}
    flat = lambda text: re.sub(r"[\s_-]+", " ", str(text).lower()).strip()      # noqa: E731
    return {k: v for k, v in keys.items() if flat(k) == flat(prop)} or keys


def _limit(need: str) -> tuple[str, float, str, bool] | None:
    """A must that sets a numeric limit: (min or max, the figure, its unit word, whether the edge itself is out); None
    for any other must."""
    body = (need.split(":", 1)[1] if ":" in need else need).lower()
    body = re.sub(r"(\d),(\d{3})\b", r"\1\2", body)
    word = re.search(r"\b(?:" + LIMIT_WORDS + r")(?![a-z])", body)
    if not word:
        return None
    number = r"(\d+(?:\.\d+)?)\s*([a-z]+(?:/[a-z]+)?)?"
    if re.match(r"or |and ", word.group(0)):
        # the limit word follows its figure ("1.35 v or less")
        before = list(re.finditer(number, body[: word.start()]))
        figure = before[-1] if before else None
    else:
        figure = re.search(number, body[word.end():]) or re.search(number, body)
    if not figure:
        return None
    lower = re.match(r"at least|no less|not less|minimum|or more|and up|and above|over|above|more than", word.group(0))
    strict = word.group(0) in ("under", "below", "over", "above", "less than", "more than")
    return ("min" if lower else "max"), float(figure.group(1)), _unit(figure.group(2) or ""), strict


def _limit_verdict(limit: tuple[str, float, str, bool], values: list[Any], factors: dict[tuple[str, str], float | None] | None = None) -> str | None:
    """'meets' or 'fails' when the values state figures in the limit's own unit word (a value with one bare figure is read
    in that unit) and all of them agree; None otherwise (another unit, a range, a pack of parts: left to the reader)."""
    side, amount, unit, strict = limit
    verdicts = set()
    for value in values:
        found = _amounts(value)
        same = [n for u, n in found if u == unit and u]
        slack = 1e-9
        if not same and unit:
            # the same measure in another unit: rounding in the conversion is no reason to fail a figure at the edge
            same = [n * factors[(u, unit)] for u, n in found if u and (factors or {}).get((u, unit))]
            slack = amount * 1e-3
        if not same and not any(u for u, _ in found) and len(found) == 1:
            same = [found[0][1]]
        for got in same:
            if side == "min":
                verdicts.add(got > amount if strict else got >= amount - slack)
            else:
                verdicts.add(got < amount if strict else got <= amount + slack)
    if len(verdicts) != 1:
        return None
    return "meets" if verdicts.pop() else "fails"


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
        self.lists_on = True
        self.lists_sent = 0
        self.named: set[tuple[int, str]] = set()
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
        self.held_at: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.audit_seen = -1
        self.audits = 0
        self.comp_held = 0
        self.cart_obs: Any = None
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
        self.swept = -1
        self.scanned = False
        self.also_fit: dict[str, set[str]] = {}
        self.search_log: list[dict[str, Any]] = []
        self.ceiling_mode: dict[str, Any] | None = None
        self.ceiling_tries = 0
        self.several_held = False
        self.shops_read: set[str] = set()
        self.shelf_seen: tuple[int, set[str]] = (-1, set())
        self.fit_unmet = 0
        self.looked = False
        self.narrow: dict[str, int] = {}
        self.caps: dict[int, float] = {}
        self.hinted: set[tuple[int, str]] = set()
        self.lanes: dict[int, list[str]] = {}
        self.lane_turns: dict[int, int] = {}
        self.lane_order: list[int] = []
        self.late_asked = False
        self.pairs: list[tuple[int, int | None, str]] = []
        self.searches: dict[str, dict[str, Any]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.page_held = False
        self.paged_obs: list[Any] = []
        self.t0 = time.monotonic()
        self.decline_held = False
        self.last_tried = -1
        self.audit_memo: tuple[int, Any] | None = None
        self.factors: dict[tuple[str, str], float | None] = {}
        self.settled_at = -1
        self.fell_back = False

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        """Seconds since the episode began (the sandbox stops it at 300)."""
        return time.monotonic() - self.t0

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
            if self.calls >= ALL_CALLS or self.clock() >= CLOCK_STOP:
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
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or self.clock() + WAITS[tries] >= CLOCK_LATE):
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
        if model.startswith("anthropic/") and len(messages) == 1 and isinstance(messages[0].get("content"), str):
            # a one-shot review opens with instructions that never change: they are cached, what follows them is not
            text = messages[0]["content"]
            head = _fixed_head(text)
            if head:
                sent = [{**messages[0], "content": [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
                                                    {"type": "text", "text": text[len(head):]}]}]
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
        many = self.compare_target() if name == self.roles.get("compare") else None
        if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room():
            # every path that lines listings up side by side keeps to what the store takes
            args = {**args, many: args[many][: self.compare_room()]}
        if name == self.roles.get("search"):
            self.search_log.append(dict(args))
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
                seen = data["observation"] = _open_records(data.get("observation"))
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                if name in (self.roles.get("cart"), self.roles.get("cart_view")):
                    self.cart_obs = seen
                if name == self.roles.get("cart"):
                    # a change can come with the hold itself and unannounced, so the step each line was held at is kept
                    for value in args.values():
                        if isinstance(value, str) and value in self.rows:
                            self.held_at[value] = self.step
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
        # a job the reading named no tool for (or a reading that never came): the tools' own words and schemas name it
        for job, tool in self.code_roles().items():
            if not self.roles.get(job) and tool not in self.roles.values():
                self.roles[job] = tool
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
        # the store keeps one line per product: two items naming the same product are one item (the larger quantity, all musts)
        merged: list[Item] = []
        became: dict[int, int] = {}
        for place, item in enumerate(self.items, 1):
            twin = next((m for m in merged if _stem(m.name) == _stem(item.name) or (m.shelf and item.shelf and _stem(m.shelf) == _stem(item.shelf))), None)
            if twin is None:
                merged.append(item)
                became[place] = len(merged)
            else:
                became[place] = merged.index(twin) + 1
                twin.qty = max(twin.qty, item.qty)
                twin.musts += [m for m in item.musts if m not in twin.musts]
        if len(merged) < len(self.items):
            self.trace.append({"kind": "merged_items", "names": [i.name for i in self.items]})
            self.items = merged
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        self.name_musts()
        self.read_caps()
        self.pairs = []
        for entry in _items(brief.get("fits")):
            part = became.get(_count(entry.get("part")) or 0) if isinstance(entry, dict) else None
            other = became.get(_count(entry.get("with")) or 0) if isinstance(entry, dict) else None
            model = str(entry.get("model") or "").strip().upper() if isinstance(entry, dict) else ""
            if part and 1 <= part <= len(self.items) and (model or (other and 1 <= other <= len(self.items) and other != part)):
                self.pairs.append((part, other if not model else None, model))
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs and not _conditional(text):
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = _as_written(budget, self.request)
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

    def read_caps(self) -> None:
        """Each product's own price limit: a must in words for the model and the checks, and a number for code."""
        names = "\n".join(f"item {n}: {item.name}" for n, item in enumerate(self.items, 1))
        got = self.ask_json(CAPS + "\n\nITEMS:\n" + names + "\n\nREQUEST:\n" + self.request, [LIGHT, HEAVY], 400) or {}
        for entry in _items(got.get("caps")):
            number = _count(entry.get("item")) if isinstance(entry, dict) else None
            cap = entry.get("cap") if isinstance(entry, dict) else None
            if number and 1 <= number <= len(self.items) and isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
                cap = _as_written(cap, self.request)
                self.caps[number] = float(cap)
                self.items[number - 1].musts.append(f"price: at most {cap:g} a unit")

    def code_roles(self) -> dict[str, str]:
        """The tool for each job read off the tools' names, descriptions and parameters alone (no model): a finish tool
        without required arguments that buys or that ends without buying, a messaging tool taking one free text, a
        cart tool taking a listing and a count or a mode, a search tool taking a query, a single-listing view taking
        one listing, a side-by-side view taking a list."""
        found: dict[str, str] = {}
        for tool in self.tools:
            fn = tool["function"]
            name = fn["name"]
            words = _words(name) + " " + str(fn.get("description") or "").lower()
            schema = self.schemas.get(name) or {}
            props = schema.get("properties") or {}
            texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            counts = [k for k, v in props.items() if isinstance(v, dict) and _kinds(v) and set(_kinds(v)) <= {"integer", "number"}]
            modes = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)]
            lists = [k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)]
            required = schema.get("required") or []
            leave = re.search(r"abstain|declin|give.?up|without (?:buy|purchas|order)|no purchase|stand.?down|walk away|not buy", words)
            # (a finish tool may take an optional reason text; it never requires an argument)
            if not required and leave and not counts and not lists:
                found.setdefault("refuse", name)
            elif not required and not counts and not lists and re.search(r"order|check.?out|purchase|buy|place|finali[sz]|settle|pay", words):
                found.setdefault("buy", name)
            elif len(texts) == 1 and not counts and not lists and re.search(r"shopper|customer|user|buyer|message|ask|say|tell|write|chat", words):
                found.setdefault("ask", name)
            elif texts and (counts or modes) and re.search(r"cart|basket|trolley|bag|hold|add|keep", words):
                found.setdefault("cart", name)
            elif lists and not counts and re.search(r"compar|side|several|weigh|line.?up|one or more|full page|details|in full", words):
                found.setdefault("compare", name)
            elif texts and (modes or counts or len(props) >= 2) and re.search(r"search|find|seek|look|browse|query|list", words):
                found.setdefault("search", name)
            elif len(required) == 1 and required[0] in texts and re.search(r"view|examin|detail|describ|show|inspect|open|page|listing|item|offer", words):
                found.setdefault("open", name)
            elif not required and not texts and re.search(r"cart|basket|trolley|bag", words):
                found.setdefault("cart_view", name)
        return found

    def name_musts(self) -> None:
        """Describing words in a product's own name ("for business wear") that none of its musts hold become musts: one
        model call, made only when such words are left over."""
        loose = []
        for number, item in enumerate(self.items, 1):
            held = " ".join(item.musts).lower()
            for words in re.findall(NAME_WORDS, item.name.lower()):
                if not all(w in held for w in re.findall(r"[a-z0-9]{3,}", words)):
                    loose.append(f"item {number}: {item.name} - musts now: " + ("; ".join(item.musts) or "none"))
                    break
        if not loose:
            return
        got = self.ask_json(NAME_MUSTS + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCTS:\n" + "\n".join(loose), [HEAVY, LIGHT], 400) or {}
        added = []
        for entry in _items(got.get("musts")):
            if not isinstance(entry, dict):
                continue
            number = _count(entry.get("item"))
            must = str(entry.get("must") or "").strip()
            if number and 1 <= number <= len(self.items) and ":" in must and must not in self.items[number - 1].musts:
                self.items[number - 1].musts.append(must)
                added.append((number, must))
        self.trace.append({"kind": "name_musts", "added": added})

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
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        if re.search(r"\bmodel [A-Z]{2,5} ?\d{3,5}\b", self.request):
            # a fit code the shopper gives for their own vehicle or device: the listing's fits line decides it
            parts.append("fit rule: a code the shopper gives for their own vehicle or device (such as model ABC 1234) is met by a listing whose"
                         " page or description says it fits that code or a pattern covering it (\"fits every ABC 12xx model\" covers ABC 1234);"
                         " a real make or model named in the listing's keys is what it was built for and does not decide this requirement")
        for rule in self.pairing_rules():
            parts.append("shop pairing rule seen on a listing's page (when a requirement goes with another listing's value, read it through this): " + rule)
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def pairing_rules(self) -> list[str]:
        """Shop rules that pair one value with another ("casual goes with school uniform", "pairs ford with kia"), read
        off the pages opened so far, as 'property: a -> b; c -> d' lines (the property as the text names it)."""
        out: list[str] = []
        seen: set[str] = set()
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            for value in record.values():
                if not isinstance(value, str) or not re.search(r"(?i)goes? (?:together )?with|\bpairs\b|pairings?\b|paired with", value):
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", value):
                    low = sentence.lower()
                    if not re.search(r"goes? (?:together )?with|\bpairs\b|pairings?\b|paired with", low):
                        continue
                    name = (re.search(r"rules? for ([a-z][a-z ]{1,30}?)\s*[:,]", low) or re.search(r"\b(?:our|its|the store's|the shop's|store|shop)\s+([a-z][a-z ]{1,30}?)\s+(?:rule|pairings?)\b", low)
                            or re.search(r"([a-z][a-z ]{1,30}?)\s+pairings?\s*(?:are)?\s*:", low))
                    body = re.split(r"pairings?(?: are)?\s*:|rules?[^:]{0,40}:|\bpairs\b", low, maxsplit=1)[-1]
                    body = re.sub(r",\s+(goes? (?:together )?with)", r" \1", body)   # ("its value, chalk, goes together with slate")
                    pairs = re.findall(r"(?:^|,|;|\band\b)\s*(?:and\s+)?([a-z0-9][a-z0-9 &/()-]{0,30}?)\s+(?:goes? together with|goes with|go with|with)\s+([a-z0-9][a-z0-9 &/()-]{0,30}?)(?=\s*(?:,|;|\.|$|\band\b))", body)
                    pairs = [(x.strip(), y.strip()) for x, y in pairs if x.strip() and y.strip()]
                    if len(pairs) < 2 and not (pairs and re.search(r"goes? (?:together )?with", low)):
                        continue
                    key = "; ".join(f"{x} -> {y}" for x, y in pairs[:8])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(f"{name.group(1).strip() if name else 'value'}: {key}")
        return out[:4]

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= 2 * len(self.items) + 7 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 14:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps. The shopper or the store often changes something at the first checkout, and answering that takes steps too."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock() >= CLOCK_WARN and self.buys == 0:
            covered = {self.item_of(i) for i in self.cart_ids() if not self.sold_out(i)} - {None}
            text += (f" TIME: {int(300 - self.clock())} s of the episode's clock are left, whatever the steps say. "
                     + ("Check out the cart now." if len(covered) >= len(self.items) else "Cart a fitting listing for each item now and check out."))
        elif self.cost >= 0.8 * SPEND_HARD and self.buys == 0:
            # the episode's spend is nearly used up: the model's own picks (with what it learned in this chat) make the
            # cart, not the code's fallback when the loop stops
            covered = {self.item_of(i) for i in self.cart_ids() if not self.sold_out(i)} - {None}
            if len(covered) < len(self.items):
                text += " SPEND: this episode's allowance is nearly used up. Cart your best fitting listing for every item now (by what you have read), then check out."
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
                    self.note_fits(_dump(node))
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

    def shelf_labels(self, seen: set[str]) -> set[str]:
        """The options of each enum that holds a shelf the rows showed: other enums (modes, sort orders) name no shelf."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            options = {o.strip().lower() for o in schema.get("enum") or [] if isinstance(o, str)}
            if options & seen:
                found.update(options)
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

    def key_table(self, keep: int = 14) -> list[tuple[str, str, int, list[str]]]:
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
                # values all different and long look like identifiers, unless they share words across listings, as a
                # spec does ("mediatek helio", "mediatek dimensity"): a processor or an interface stays in the table
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 \
                        and (re.search(r"\boptions?\b", name) or not _shared_words(values)):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:keep]:
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

    def text_key(self, props: dict[str, Any]) -> str | None:
        """The parameter of the search tool that takes words."""
        free = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and not re.search(PAGE_KEY, k)]
        named = [k for k in free if re.search(r"(?i)word|query|term|text|keyword|search|phrase|find", k)]
        return (named or free)[0] if len(named) == 1 or len(free) == 1 else None

    def word_args(self, words: str) -> dict[str, Any] | None:
        """The arguments of a search for these words. A store's search can take a mode as well: the model's own last call shows how
        this store is searched, so a mode of any name is copied from it; before any call, the option that does not read as browsing."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = self.single_target(tool) if tool else None
        if target:
            args = {target: words}
        else:
            key = self.text_key(props) or next((k for c in reversed(self.search_log) for k, v in c.items() if k in props and isinstance(v, str) and "enum" not in props[k] and not re.search(PAGE_KEY, k)), None)
            if not key:
                return None
            args = {key: words}
            modes = {k: [o for o in (v.get("enum") or []) if isinstance(o, str)] for k, v in props.items() if isinstance(v, dict) and v.get("enum") and len(v["enum"]) <= 4
                     and (k in (schema.get("required") or []) or re.search(r"(?i)type|mode|how|kind|method", k))}
            for mode, options in modes.items():
                said = [c[mode] for c in reversed(self.search_log) if isinstance(c.get(key), str) and c.get(mode) in options and not re.search(BROWSE_MODE, str(c[mode]))]
                plain = [o for o in options if not re.search(BROWSE_MODE, o)]
                if said or plain:
                    args[mode] = (said or plain)[0]
            if any(k not in args for k in schema.get("required") or []):
                return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def code_key(self, words: tuple[str, ...], kinds: tuple[type, ...]) -> str | None:
        """A row key found by code when the model named none: the key, among those whose name holds one of the words,
        that carries a value of the wanted kind in the most listing records."""
        hits: dict[str, int] = {}
        for record in self.rows.values():
            for key, value in record.items():
                if any(w in key.lower() for w in words) and isinstance(value, kinds) and (bool in kinds or not isinstance(value, bool)):
                    hits[key] = hits.get(key, 0) + 1
        return max(hits, key=lambda k: hits[k]) if hits else None

    def code_price(self, ident: str) -> float | None:
        """The price for code-made carts: under the model's price key, else under a key found by code."""
        price = self.price_of(ident)
        if price is not None or self.price_key:
            return price
        key = self.code_key(("price", "cost", "amount"), (int, float))
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def code_buyable(self, ident: str) -> bool:
        """Buyable for code-made carts: the model's stock key when it named one, else a yes/no stock key found by code."""
        if not self.buyable(ident):
            return False
        if self.stock_key:
            return True
        key = self.code_key(("stock", "hand", "avail"), (bool,))
        return (self.rows.get(ident) or {}).get(key) is not False if key else True

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(value, str) and value.strip().lower() in ("no", "false", "none", "0", "out of stock", "sold out", "unavailable"):
            return False
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
                    ident = self.ident(value, True)
                    if ident:
                        found.append((ident, value))
                        continue
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
        self.price_key = shape["price"] or self.price_key
        self.stock_key = shape["stock"] or self.stock_key
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
        self.note_narrow(observation)
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        self.note_search(name, args, plan)
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        compact = False
        if pick == "cheapest" and pages > 2 and self.pages_read + 2 <= AUTO_PAGES - 2 and self.left() - self.tail() - 2 >= 12:
            # a long result under a cheapest pick: the next two pages are read here, since a cheaper fit often sits on
            # them and more words seldom narrow a store's search (twice an episode, while steps allow); shown compactly
            pages = 2
            compact = True
        elif pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"page on with {param}={_dump(value)} (more words may not narrow a store's search), or search with the requirement's value.")
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
            if compact:
                rows = [self.row_line(i) for i, _ in self.records_in(seen) if self.buyable(i)]
                texts.append(f"{param}={_dump(value)}: buyable rows: " + ("; ".join(rows) or "none"))
            else:
                texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if after:
                self.note_search(name, call, after)
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def note_narrow(self, observation: Any) -> None:
        """For each row of a result, the fewest matches of any search that showed it: a store that wants every word found shows a row
        in a narrow search only when the listing holds all the words, even when the row itself does not show them."""
        rows = self.records_in(observation)
        if not rows or not isinstance(observation, dict):
            return
        self.price_key = self.price_key or self.code_key(("price", "cost"), (int, float))
        self.stock_key = self.stock_key or self.code_key(("stock", "hand", "avail"), (bool,))
        if self.stock_key and re.search(r"(?i)out|unavail|sold|no_?stock|hand_?made|second_?hand", self.stock_key):
            self.stock_key = None
        count = max((v for v in observation.values() if isinstance(v, int) and not isinstance(v, bool) and v >= len(rows)), default=len(rows))
        for ident, _ in rows:
            self.narrow[ident] = min(self.narrow.get(ident, count), count)

    def note_search(self, name: str, args: dict[str, Any], plan: tuple[int, int, str, Any]) -> None:
        """Where a search stands: the page to ask for next, or none once its last page was read."""
        count, _, param, value = plan
        base = {k: v for k, v in args.items() if k != param}
        key, given = name + _dump(base), args.get(param)
        if value is None and key not in self.searches and isinstance(given, int) and not isinstance(given, bool) and given > 1:
            value = 1       # (its last page came first: the pages before it are unread)
        self.searches[key] = {"tool": name, "args": base, "param": param, "next": value, "count": count}

    def page_on(self, search: dict[str, Any]) -> bool:
        """One more page of a search, read by code: a step."""
        faults: list[str] = []
        call = _shape({**search["args"], search["param"]: search["next"]}, self.schemas.get(search["tool"]), "args", faults)
        if faults:
            search["next"] = None
            return False
        reply = self.act(search["tool"], call)
        if reply is None or self.finished:
            search["next"] = None
            return False
        seen = reply.get("observation")
        self.observe(search["tool"], seen)
        inner = next((v for v in seen.values() if isinstance(v, dict)), seen) if isinstance(seen, dict) and len(seen) == 2 else seen
        after = self.paging(search["tool"], inner)
        search["next"] = after[3] if after else None
        self.read_cache[search["tool"] + _dump(call)] = (self.version, self.step, self.show(seen))
        self.pages_read += 1
        return True

    def search_words(self) -> list[tuple[int, str]]:
        """(item, words) of searches for the requirements in force not run yet: the value alone, since a store's search may want
        every word found and a listing can state the value without the product's own name."""
        tool = self.roles.get("search")
        if not tool or self.word_args("x") is None:
            return []
        out: list[tuple[int, str]] = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need in item.dropped or ":" not in need or _limit(need) or (any(need == ref for _, ref in self.refs) and need not in self.ref_values):
                    continue
                value = self.ref_values.get(need, need.split(":", 1)[1].strip())
                if re.match(r"(?i)(anything other than|not |no |unlike|other than|except)", value):
                    continue
                words = " ".join(value.split()[:2])
                args = self.word_args(words) if words else None
                if args is not None and tool + _dump(args) not in self.read_cache and tool + _dump(args) not in self.searches \
                        and words not in [w for _, w in out]:
                    out.append((number, words))
        return out

    def sweep(self) -> str:
        """Before a cheapest checkout, and again after any change of the task: the order must carry the lowest price among every
        fitting listing, so code finishes the searches, reads the rows below each line's price, cheapest first and several a
        step, and names what no attribute rules out. It runs on the steps the turn model would leave unused."""
        if self.brief.get("pick") != "cheapest" or not self.price_key or self.swept == self.version:
            return ""
        self.swept = self.version
        room = self.left() - self.tail() - 3 - len(self.items)
        # the sweep makes no model call: baskets of several items reach their checkout past the soft limit, and they need it most
        if room < 3 or self.cost > SPEND_HARD or self.clock() > CLOCK_WARN:
            return ""
        before, pages, start = len(self.full), 0, self.step
        self.wider_held = self.page_held = True     # (what those holds would ask of the turn model, code does here)
        for number in range(1, len(self.items) + 1):
            if self.step - start < room:
                self.descend(number, max(2, (room - (self.step - start)) // (len(self.items) - number + 1)))
        listed = self.ceiling_mode is not None
        if not listed:
            self.more_searches(room // 3)
        while not listed and self.step - start < room // 2 and not self.finished and self.clock() < CLOCK_WARN + 20:
            open_ones = [s for s in self.searches.values() if s["next"] is not None]
            if not open_ones:
                break
            search = min(open_ones, key=lambda s: s["count"])
            if self.page_on(search):
                pages += 1
        reading = self.read_below(room - (self.step - start))
        found = self.below_lines()
        basket = self.cheapest_basket() if self.pairs else {}
        if basket and set(basket.values()) != set(self.cart_ids()):
            total = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in basket.items())
            found = (f"cheapest basket by code, with each part fitting what it must fit: {', '.join(f'item {n}: {i}' for n, i in sorted(basket.items()))}"
                     f" (total {total:g}). Also: " + found)
        self.trace.append({"kind": "sweep", "pages": pages, "read": len(self.full) - before, "steps": self.step - start, "found": found[:300]})
        if not found:
            return ""
        return (f"Not run yet. The pick is the lowest price, so Kyo ran {self.step - start} steps of searching and reading for you"
                + (f" and read {reading} listings" if reading else "") + ". Listings priced below the cart's line that no attribute rules out, cheapest first: "
                + found + ". Take the cheapest one that truly meets every requirement in force, then check out.")

    def read_shops(self) -> str:
        """A requirement can point at a listing by its shop and a few details, and a shop holds several listings alike, so the one
        meant is easily mistaken. A shop's name finds its listings: code reads them all, a few steps, and shows them side by side."""
        tool = self.roles.get("search")
        compare, target = self.roles.get("compare"), self.compare_target()
        if not tool:
            return ""
        side = bool(compare and target) and not self.compare_bare()
        size = max(1, self.compare_room()) if side else 1
        sheets: list[str] = []
        for _, need in self.refs:
            if need in self.shops_read or need in self.ref_values or ":" not in need:
                continue
            self.shops_read.add(need)
            value = need.split(":", 1)[1]
            for name in list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]{3,}\b", value)))[:3]:
                args = self.word_args(name)
                if args is None or self.left() < 14 or self.clock() > CLOCK_WARN:
                    continue
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    return ""
                seen = reply.get("observation")
                self.observe(tool, seen)
                self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))

                def own(ident: str, record: dict[str, Any]) -> bool:
                    return any(isinstance(v, str) and v.strip().lower() == name.lower() and not re.search(r"(?i)brand|make", str(k)) for k, v in record.items())

                rows = [i for i, r in self.records_in(seen) if own(i, r)]
                plan = self.paging(tool, seen)
                if not rows or (plan and plan[0] > 40):
                    continue        # (a capital that names no shop, or too many listings to be one)
                # the details the requirement gives go into the search: a few rows to read instead of the whole shop
                close: list[str] = []
                if plan and plan[0] > 6:
                    details = _details(value)
                    for words in ([f"{name} {' '.join(details)}"] if len(details) > 1 else []) + [f"{name} {d}" for d in details]:
                        if self.left() < 14:
                            break
                        got = self.scan_search(words)
                        close = [i for i in (got[2] or []) if own(i, self.rows.get(i) or {})] if got else []
                        if close:
                            break
                if close:
                    plan = None
                if plan:
                    self.note_search(tool, args, plan)
                    search = self.searches.get(tool + _dump({k: v for k, v in args.items() if k != plan[2]}))
                    while search and search["next"] is not None and self.left() > 14 and not self.finished:
                        self.page_on(search)
                rows = close or [i for i in self.seen_order if own(i, self.rows.get(i) or {})]
                # the kind of product the words name, when they name one of the shop's shelves
                cat_key = self.category_key()
                low = value.lower()
                kind = [i for i in rows if cat_key and str((self.rows.get(i) or {}).get(cat_key) or "?").lower() in low]
                rows = kind or rows
                said = set(re.findall(r"[a-z0-9]{3,}", low))
                unread = sorted((i for i in rows if i not in self.full), key=lambda i: -len(said & set(re.findall(r"[a-z0-9]{3,}", _dump(self.rows.get(i) or {}).lower()))))
                if not side:
                    self.fill_facts(unread[:6], self.tail() + 10)
                    unread = []
                for start in range(0, min(len(unread), 3 * size), size):
                    faults: list[str] = []
                    call = _shape({target: unread[start:start + size]}, self.schemas.get(compare), "args", faults)
                    answer = self.act(compare, call) if not faults else None
                    if answer is None or self.finished:
                        break
                    shown = answer.get("observation")
                    self.observe(compare, shown)
                    self.note_view(compare, shown)
                lines = [self.shop_line(i) for i in rows[:12]]
                meant = [i for i in rows if self.shows_details(i, _details(value))]
                if len(meant) == 1:
                    lines.insert(0, f"{meant[0]} is the only one here showing every detail the requirement gives (figures in other units included).")
                sheets.append(f"{name}'s listings, all read, for the requirement \"{need}\":\n" + "\n".join(lines))
                self.trace.append({"kind": "shop", "name": name, "rows": len(rows), "read": len([i for i in rows if i in self.full])})
                break
        if not sheets:
            return ""
        return ("\n[Kyo] " + "\n".join(sheets) + "\nThe listing meant is the one showing EVERY detail the requirement gives; follow its own note if it sends you on. Open it in full, since the order counts only when each page of the chain was opened.")

    def shows_details(self, ident: str, details: list[str]) -> bool:
        """Whether a listing read in full shows each detail of a description: a word in its record, a figure in any unit the record allows."""
        record = self.full.get(ident)
        if not record or not details:
            return False
        text = _dump(record).lower() + " " + self.restated(_fact_block(record) or {}).lower()
        for detail in details:
            if re.fullmatch(r"\d+(?:\.\d+)?", detail):
                if not any(abs(float(n) - float(detail)) <= float(detail) * .005 for n in re.findall(r"\d+(?:\.\d+)?", text)):
                    return False
            elif not re.search(r"(?<![a-z0-9])" + re.escape(detail.lower()) + r"(?![a-z0-9])", text):
                return False
        return True

    def shop_line(self, ident: str) -> str:
        """One listing of a shop in a line: id, title, attributes, and any note in its record."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = _fact_block(record) or {}
        facts = "; ".join(f"{k}={v}" for k, v in block.items() if str(v).strip() and "_" not in str(k))
        notes = [v for k, v in record.items() if isinstance(v, str) and len(v) >= 30 and not re.search(r"(?i)title|name", str(k))]
        same = self.restated(block)
        return f"{ident}: {_clip(str(self.row_line(ident)), 90)} | {facts[:240]}" + (f" | {same}" if same else "") + (f" | {_clip(notes[0], 200)}" if notes else "")

    def more_searches(self, room: int) -> None:
        """The searches for the requirements in force not run yet, within `room` steps."""
        tool, start = self.roles.get("search"), self.step
        for _, words in self.search_words()[:2 * len(self.items)]:
            if self.step - start >= room or self.finished:
                break
            args = self.word_args(words)
            reply = self.act(tool, args) if args is not None else None
            if reply is None:
                continue
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
            plan = self.paging(tool, seen)
            if plan:
                self.note_search(tool, args, plan)

    def note_fits(self, text: str) -> None:
        """A listing can speak for another part ("Part X also fits: Y"): what it says is kept under that part's code."""
        for code, names in re.findall(r"(?i)\bpart\s+([a-z0-9][a-z0-9-]{2,})\b[^.:;|\"]{0,40}:\s*([^.;|\"]+)", text):
            self.also_fit.setdefault(code.upper(), set()).update(n.strip().upper() for n in re.split(r",|\band\b", names) if n.strip())

    def part_code(self, ident: str) -> str | None:
        """The code a listing's title carries in brackets."""
        found = re.search(r"\[([A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\]", _dump(self.rows.get(ident) or {}))
        return found.group(1).upper() if found else None

    def series_of(self, device: str) -> str | None:
        """The family a device takes its parts from: a code in its own words that parts carry at the head of theirs ("[AB7-1234]"), whatever
        sentence holds it."""
        heads = {code.split("-")[0] for code in (self.part_code(i) for i in self.rows) if code and "-" in code}
        words = set(re.findall(r"\b[A-Z][A-Z0-9]*\d[A-Z0-9]*\b", _dump(self.rows.get(device) or {})))
        shared = sorted(words & heads)
        return shared[0] if len(shared) == 1 else None

    def fit_verdict(self, ident: str, names: set[str]) -> bool | None:
        """Whether a part takes a device known by these names, by its own list and what other listings say of its code. None when
        nothing is stated, or the list sends the reader on to another part and what was read does not settle it."""
        own = _dump(self.rows.get(ident) or {})
        rule = _fit_rule(own)
        said = self.also_fit.get(self.part_code(ident) or "", set())
        if names & said or (rule and _rule_covers(rule, names)):
            return True
        if not rule and not said:
            return None
        return None if re.search(r"(?i)\bsee part\b", own) else False

    def fits_device(self, ident: str, device: str) -> bool | None:
        """Whether a part takes that device's listing; None when the records do not settle it."""
        other = _dump(self.rows.get(device) or {})
        series, code = self.series_of(device), self.part_code(ident)
        if series and code and "-" in code:
            return code.split("-")[0] == series
        names = _device_names(other)
        return self.fit_verdict(ident, names) if names else None

    def fit_search(self, room: int) -> int:
        """Parts that must fit a device: the device's own series code searched with the part's wanted value leaves the few rows that can
        fit it, where the part's shelf holds hundreds. Devices that show the most first, cheapest first. Returns the steps spent."""
        start, asked = self.step, set()
        for host in sorted({h for _, h, _ in self.pairs if h}):
            for _, device in self.basket_pool(host, self.items[host - 1]):
                code = self.series_of(device) or ""
                if not code or code in asked:
                    continue
                asked.add(code)
                for number in [part for part, other, _ in self.pairs if other == host]:
                    item = self.items[number - 1]
                    values = [re.split(r"\s+or\s+|\s*,\s*", need.split(":", 1)[1].strip())[0] for need in item.told + item.musts
                              if ":" in need and need not in item.dropped and not _limit(need) and not any(need == ref for _, ref in self.refs)]
                    for words in ([f"{values[0]} {code}"] if values else []) + [f"{self.shelf_name(number)} {code}"]:
                        if self.step - start >= room or self.finished:
                            self.trace.append({"kind": "fit_search", "codes": sorted(asked), "steps": self.step - start})
                            return self.step - start
                        got = self.scan_search(words)
                        if got and got[2]:
                            break
        self.trace.append({"kind": "fit_search", "codes": sorted(asked), "steps": self.step - start})
        return self.step - start

    def follow_parts(self, number: int, room: int) -> int:
        """Parts whose list says the rest is on another part: that part's code is searched, since its listing can name them."""
        start = self.step
        asked: set[str] = set()
        for ident in sorted((i for i in self.seen_order if self.on_item(i, number) and self.buyable(i)), key=lambda i: self.price_of(i) or 0.0):
            if self.step - start >= room or self.finished:
                break
            found = re.search(r"(?i)\bsee part\s+([a-z0-9][a-z0-9-]{2,})", _dump(self.rows.get(ident) or {}))
            if found and found.group(1).upper() not in asked:
                asked.add(found.group(1).upper())
                self.scan_search(found.group(1))
        return self.step - start

    def fits_model(self, ident: str, number: int) -> bool:
        """Whether a listing passes the model the request names for its item (a listing that says nothing passes)."""
        return all(self.fit_verdict(ident, {model.upper()}) is not False for part, _, model in self.pairs if part == number and model)

    def fit_lines(self, ids: list[str]) -> str:
        """What the fit rules say about these listings, a line each; empty when no rule or no device could be read."""
        lines = []
        for part, other, model in self.pairs:
            for ident in [i for i in ids if self.item_of(i) == part]:
                if model:
                    verdict = self.fit_verdict(ident, {model.upper()})
                    if verdict is not None:
                        lines.append(f"{ident} {'fits' if verdict else 'does NOT fit'} model {model}")
                for device in [i for i in ids if other and self.item_of(i) == other]:
                    verdict = self.fits_device(ident, device)
                    if verdict is not None:
                        lines.append(f"{ident} {'fits' if verdict else 'does NOT fit'} {device}")
        return "\n".join(lines)

    def basket_pool(self, number: int, item: Item) -> list[tuple[float, str]]:
        pool = self.fitting(number, item)
        least = min((unmet for unmet, _, _ in pool), default=0)
        return [(p, i) for unmet, p, i in pool if unmet == least][:12]

    def fitting(self, number: int, item: Item) -> list[tuple[int, float, str]]:
        """Listings read for an item that no attribute rules out, as (requirements not shown, price, id): those showing the most first, cheapest first."""
        pool = []
        for ident in self.full:
            price = self.price_of(ident)
            if self.on_item(ident, number) and self.buyable(ident) and price is not None and self.fits_model(ident, number):
                if self.fit_score(ident, item) is not None:
                    pool.append((self.fit_unmet, price, ident))
        return sorted(pool)

    def item_queries(self, item: Item) -> list[str]:
        """Searches for one item, sharpest first: a store shows only the first pages of a search, so a wide one can never reach a
        listing far down it, while two of the item's values, or a value with the product's name, leave few rows to read."""
        values: list[str] = []
        for need in item.told + item.musts:
            if need in item.dropped or ":" not in need or _limit(need) or (any(need == ref for _, ref in self.refs) and need not in self.ref_values):
                continue
            value = self.ref_values.get(need, need.split(":", 1)[1].strip())
            if re.match(r"(?i)(anything other than|not |no |unlike|other than|except)", value) or _conditional(value):
                continue
            named = set(re.findall(r"[a-z0-9]+", need.split(":", 1)[0].lower()))
            for part in re.split(r"(?i)\s+or\s+|,\s*", value):
                words = " ".join(([w for w in part.split() if w.lower() not in named] or part.split())[:2])
                if words and words.lower() not in [v.lower() for v in values]:
                    values.append(words)
        noun = (re.findall(r"[A-Za-z]{3,}", item.name) or [""])[-1]
        number = self.items.index(item) + 1
        out = [model for part, _, model in self.pairs if part == number and model]
        out += [f"{one} {two}" for n, one in enumerate(values[:3]) for two in values[n + 1:3]]
        out += [f"{value} {noun}".strip() if noun.lower() not in value.lower() else value for value in values]
        return list(dict.fromkeys(out))

    def shelf_name(self, number: int) -> str:
        """The store's own label for an item's shelf: the brief's, else the one most rows of the item carry, else the item's name."""
        item, key = self.items[number - 1], self.category_key()
        if item.shelf or not key:
            return item.shelf or item.name
        counts: dict[str, int] = {}
        for ident, row in self.rows.items():
            if isinstance(row.get(key), str) and row[key].strip() and self.item_of(ident) == number:
                counts[row[key].strip()] = counts.get(row[key].strip(), 0) + 1
        return max(counts, key=lambda k: counts[k]) if counts else item.name

    def item_limit(self, number: int) -> float | None:
        """The most one unit of an item can cost by the limits in force."""
        limit = self.caps.get(number)
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
            each = self.budget if self.brief.get("budget_scope") == "each" else self.budget / max(1, self.items[number - 1].qty)
            limit = min(limit, each) if limit is not None else each
        return limit

    def descend(self, number: int, room: int) -> int:
        """Under a cheapest pick, where the store can list a shelf below a price: list the rows under the best price so far, read them
        cheapest first until one shows every requirement, and go again below it, so a shelf of hundreds comes down to the few rows
        that decide. Waits for values still to be read off another page. Returns the steps spent."""
        item, start, last = self.items[number - 1], self.step, None
        live = {need for _, need in self.needs_now()}
        if any(need in live and need not in self.ref_values for _, need in self.refs):
            return 0
        while self.step - start < room and not self.finished and self.clock() < CLOCK_WARN:
            shown = [p for unmet, p, _ in self.fitting(number, item) if not unmet]
            top = min(shown) - 1 if shown else self.item_limit(number)
            if top is None or top <= 0 or top == last:
                break
            last = top
            self.ceiling_rows(number, top, min(2, room - (self.step - start)))
            rows = sorted((self.price_of(i), i) for i in self.seen_order if i not in self.full and self.on_item(i, number) and self.buyable(i)
                          and self.price_of(i) is not None and self.price_of(i) <= top and self.fits_model(i, number))
            hit = False
            for _, ident in rows:
                if self.step - start >= room or self.finished:
                    break
                self.fill_facts([ident], self.tail() + 3)
                if ident not in self.full:
                    break
                if self.fit_score(ident, item) is not None and not self.fit_unmet:
                    hit = True
                    break
            if not hit:
                break
        self.trace.append({"kind": "descend", "item": number, "steps": self.step - start, "top": last})
        return self.step - start

    def ceiling_args(self, number: int, top: float) -> list[dict[str, Any]]:
        """Ways to ask the search tool for an item's shelf below a price, by the tool's parameter names: the mode that kept to the
        limit before, else every mode, the store's own browsing mode first. Empty when the tool has no such parameters."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        shelf = next((k for k, v in props.items() if isinstance(v, dict) and re.search(r"(?i)group|section|shelf|categor|aisle|depart|collection", k)), None)
        price = next((k for k, v in props.items() if isinstance(v, dict) and {"integer", "number"} & set(_kinds(v)) and re.search(r"(?i)budget|ceiling|limit|max|under|price|cost", k)
                      and not re.search(PAGE_KEY, k)), None)
        label = self.shelf_name(number)
        if not shelf or not price or not label:
            return []
        whole = int(top) if "integer" in _kinds(props[price]) else top
        base = {shelf: label, price: whole}
        if self.ceiling_mode is not None:
            return [{**base, **self.ceiling_mode}]
        ways = [{k: o} for k, v in props.items() if isinstance(v, dict) and v.get("enum") and k != shelf and len(v["enum"]) <= 4 for o in sorted(v["enum"], key=lambda o: not re.search(BROWSE_MODE, str(o)))]
        return [{**base, **way} for way in ways] or [base]

    def ceiling_rows(self, number: int, top: float, room: int) -> int:
        """List an item's shelf below a price, a step a page, up to `room` steps. A store's filters are tested, not trusted: a reply
        with a row over the limit or off the shelf shows that way of asking keeps to neither. Returns the steps spent."""
        tool, start = self.roles.get("search"), self.step
        schema = self.schemas.get(tool or "")
        for args in self.ceiling_args(number, top):
            if self.step - start >= room or self.finished or self.ceiling_tries >= 3:
                break
            faults: list[str] = []
            call = _shape(args, schema, "args", faults)
            if faults:
                continue
            reply = self.act(tool, call)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            rows = [i for i, _ in self.records_in(seen)]
            if not rows:
                continue        # (nothing below that price is an answer, not a fault of this way of asking)
            if not all((self.price_of(i) or 0) <= top and self.on_item(i, number) for i in rows):
                self.ceiling_tries += 1
                continue
            props = (schema or {}).get("properties") or {}
            self.ceiling_mode = {k: v for k, v in call.items() if isinstance(props.get(k), dict) and props[k].get("enum") and len(props[k]["enum"]) <= 4}
            plan = self.paging(tool, seen)
            if plan:
                self.note_search(tool, call, plan)
                search = self.searches.get(tool + _dump({k: v for k, v in call.items() if k != plan[2]}))
                while search and search["next"] is not None and self.step - start < room and not self.finished:
                    self.page_on(search)
            break
        return self.step - start

    def scan_search(self, words: str) -> tuple[int, str, list[str] | None] | None:
        """One search by code: (matches, the key of its paging state, its first rows, None for a search already run), or None
        when it could not run."""
        tool = self.roles.get("search")
        args = self.word_args(words)
        if not tool or args is None:
            return None
        same = next((key for key in self.searches if key.lower() == (tool + _dump(args)).lower()), None)
        if same:
            return self.searches[same]["count"], same, None
        if tool + _dump(args) in self.read_cache:
            return None
        reply = self.act(tool, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_narrow(seen)
        self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
        plan = self.paging(tool, seen)
        if not plan:
            return None
        self.note_search(tool, args, plan)
        return plan[0], tool + _dump({k: v for k, v in args.items() if k != plan[2]}), [i for i, _ in self.records_in(seen)]

    def scan_keep(self) -> int:
        """Steps a code scan leaves the turn model: a fixed number would leave a short episode no scan at all."""
        return min(SCAN_KEEP, max(8, self.max_steps // 4))

    def value_rows(self) -> str:
        """Any other pick: what the shopper just told decides the order, so code searches it at once, a step or two an item, and shows
        the rows of the requested kind, instead of leaving the first search for that value to a later turn."""
        self.scanned = True
        if self.left() < self.tail() + self.scan_keep() or self.clock() > CLOCK_WARN:
            return ""
        lines = []
        for number, item in enumerate(self.items, 1):
            if not [need for need in item.told if need not in item.dropped]:
                continue
            for words in self.item_queries(item)[:2]:
                got = self.scan_search(words)
                rows = [i for i in (got[2] or []) if self.on_item(i, number) and self.buyable(i)][:4] if got else []
                if rows:
                    lines.append(f'item {number} ({item.name}), search "{words}": ' + "; ".join(self.row_line(i) for i in rows))
                    break
        self.trace.append({"kind": "value_rows", "lines": len(lines)})
        return ("\n[Kyo] Kyo searched what the shopper told you:\n" + "\n".join(lines)) if lines else ""

    def scan(self) -> str:
        """Under a cheapest pick, once the requirements are heard: the model's turns are what an episode runs out of, while half its
        steps go unused. Code runs sharp searches for each item, reads them to their last page, reads the rows in price order until
        one shows every requirement, and names the result, so the turn model judges a short list instead of searching."""
        if self.scanned or self.finished:
            return ""
        if self.brief.get("pick") != "cheapest" and not self.pairs:
            return self.value_rows()
        if not self.price_key:
            return ""
        self.scanned = True
        room = self.left() - self.tail() - self.scan_keep() - 2 * len(self.items)
        if room < 3:
            return ""
        self.wider_held = self.page_held = True
        before, start, began, pages = len(self.full), self.step, self.clock(), 0
        limit = min(60.0, SCAN_SECONDS * (1 + len(self.items)))

        def open_time(share_of: float = 1.0) -> bool:
            return self.clock() - began < limit * share_of and not self.finished

        # the first page of each search says how many rows it holds; the narrowest with rows of the item is read on
        share = max(2, room // (2 * len(self.items)))
        for number, item in enumerate(self.items, 1):
            used = self.step
            found = []
            for words in self.item_queries(item)[:4]:
                if self.step - used >= share - 1 or self.step - start >= room or not open_time(2 / 3):
                    break
                got = self.scan_search(words)
                if got:
                    found.append((got[2] is not None and not any(self.on_item(i, number) for i in got[2]), got[0], got[1]))
            # a search that spans shelves can open on other shelves: those come last, and stop after three pages with nothing for the item
            for _, _, key in sorted(found):
                search, dry = self.searches.get(key), 0
                while search and search["next"] is not None and dry < 3 and self.step - used < share and self.step - start < room and open_time(2 / 3):
                    known = len(self.seen_order)
                    pages += self.page_on(search)
                    dry = 0 if any(self.on_item(i, number) for i in self.seen_order[known:]) else dry + 1
        for number in sorted({part for part, _, _ in self.pairs}):
            if self.step - start < room and open_time():
                self.follow_parts(number, min(3, room - (self.step - start)))
        # what no words found: each item's shelf below the best price so far, where the store can list that
        for number, item in enumerate(self.items, 1):
            if self.brief.get("pick") == "cheapest" and self.step - start < room and open_time():
                self.descend(number, max(2, (room - (self.step - start)) // (len(self.items) - number + 1)))
        # rows in price order, hosts first, until one shows every requirement
        hosts = {h for _, h, _ in self.pairs if h}
        tool, target = self.roles.get("compare"), self.compare_target()
        side = bool(tool and target) and not self.compare_bare()
        size = max(1, self.compare_room()) if side else 1
        order = sorted(range(1, len(self.items) + 1), key=lambda n: n not in hosts)
        tried: set[str] = set()     # (a row whose view adds no attributes stays unread: read once, not again)
        live = {need for _, need in self.needs_now()}
        waiting = any(need in live and need not in self.ref_values for _, need in self.refs)
        active = not waiting
        rounds = [order[: len(hosts)], order] if hosts else [order]
        while rounds:
            numbers = rounds.pop(0)
            if rounds:
                limit_step = start + (room * 2) // 3
            else:
                limit_step = start + room
                if hosts:
                    self.fit_search(max(0, min(3 * len(self.items), start + room - self.step - 2)))
                active = not waiting
            while active and self.step < limit_step and open_time():
                active = False
                for number in numbers:
                    item = self.items[number - 1]
                    shown = [p for unmet, p, _ in self.fitting(number, item) if not unmet]
                    top = min(shown) if shown else float("inf")
                    devices = [i for part, h, _ in self.pairs if part == number and h for _, i in self.basket_pool(h, self.items[h - 1])]
                    rows = sorted((self.price_of(i), i) for i in self.seen_order if i not in self.full and i not in tried and self.on_item(i, number) and self.buyable(i)
                                  and self.price_of(i) is not None and self.price_of(i) < top and self.fits_model(i, number)
                                  and (not devices or any(self.fits_device(i, d) is not False for d in devices)))
                    batch = [i for _, i in rows[:size]]
                    if not batch or self.step >= limit_step or not open_time():
                        continue
                    if rounds and shown:
                        continue        # (a device that fits is known: its parts are searched next)
                    active = True
                    tried.update(batch)
                    if side:
                        faults: list[str] = []
                        args = _shape({target: batch}, self.schemas.get(tool), "args", faults)
                        reply = self.act(tool, args) if not faults else None
                        if reply is None:
                            active = False
                            break
                        seen = reply.get("observation")
                        self.observe(tool, seen)
                        self.note_view(tool, seen)
                    else:
                        self.fill_facts(batch, self.tail() + 3)
        self.swept = self.version
        lines = []
        for number, item in enumerate(self.items, 1):
            pool = self.fitting(number, item)
            shown = [f"{i} ({p:g})" for unmet, p, i in pool if not unmet][:4]
            rest = [f"{i} ({p:g})" for unmet, p, i in pool if unmet][:4]
            if shown or rest:
                lines.append(f"item {number} ({item.name}): " + (", ".join(shown) or "none shows every requirement") + (f"; not showing every requirement: {', '.join(rest)}" if rest else ""))
        self.trace.append({"kind": "scan", "room": room, "pages": pages, "read": len(self.full) - before, "steps": self.step - start, "seconds": round(self.clock() - began, 1), "found": " | ".join(lines)[:500]})
        if not lines:
            return ""
        basket = self.cheapest_basket() if self.pairs else {}
        for number, ident in basket.items():
            if self.fit_score(ident, self.items[number - 1]) is None or self.fit_unmet:
                basket = {}
                break
        total = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in basket.items())
        note = f"\nCheapest basket by code, each part fitting what it must fit: {', '.join(f'item {n}: {i}' for n, i in sorted(basket.items()))} (total {total:g})." if basket else ""
        return (f"\n[Kyo] Kyo ran the searches itself ({self.step - start} steps, {len(self.full) - before} listings read in price order, no turn of yours). "
                "Listings no attribute rules out, cheapest first:\n" + "\n".join(lines) + note
                + "\nThe searching is done: read the first listings of each line above, cart the cheapest that truly meets every requirement in force, and check out. Search again only for a requirement these lists cannot settle.")

    def cheapest_basket(self) -> dict[int, str]:
        """One listing per item from the pages read, at the lowest total: none ruled out by an attribute, and each part fitting
        the listing bought for the item it must fit. Each device choice is tried with the cheapest parts that fit it."""
        pools = {n: self.basket_pool(n, item) for n, item in enumerate(self.items, 1)}
        if not all(pools.values()):
            return {}
        qty = {n: item.qty for n, item in enumerate(self.items, 1)}
        links: dict[int, list[int]] = {}
        for part, other, _ in self.pairs:
            if other and other not in [p for p, _, _ in self.pairs]:
                links.setdefault(other, []).append(part)
        best: tuple[float, dict[int, str]] | None = None
        hosts = list(links)
        choices: list[dict[int, tuple[float, str]]] = [{}]
        for host in hosts:
            choices = [{**done, host: pick} for done in choices for pick in pools[host]][:200]
        for chosen in choices:
            total, cart, ok = 0.0, {}, True
            for number in pools:
                if number in chosen:
                    pick = chosen[number]
                else:
                    needs = [host for host in hosts if number in links[host]]
                    pick = next((c for c in pools[number] if all(self.fits_device(c[1], chosen[h][1]) is not False for h in needs)), None)
                if pick is None:
                    ok = False
                    break
                total += pick[0] * qty[number]
                cart[number] = pick[1]
            if ok and (best is None or total < best[0]):
                best = (total, cart)
        return best[1] if best else {}

    def below_lines(self) -> str:
        """Per item: pages read, buyable, priced below the item's line, that no attribute rules out, cheapest first."""
        cart = self.cart_ids()
        parts = []
        for number, item in enumerate(self.items, 1):
            lines = [self.price_of(i) for i in cart if self.item_of(i) == number and self.price_of(i) is not None]
            top = min(lines) if lines else float("inf")
            fits = []
            for ident in self.full:
                price = self.price_of(ident)
                if self.on_item(ident, number) and ident not in cart and self.buyable(ident) and price is not None and price < top:
                    hosts = [i for part, other, _ in self.pairs if part == number and other for i in cart if self.item_of(i) == other]
                    if self.fit_score(ident, item) is not None and self.fits_model(ident, number) \
                            and all(self.fits_device(ident, host) is not False for host in hosts):
                        fits.append((price, ident))
            if fits:
                parts.append(f"item {number}: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(fits)[:4]))
        return " | ".join(parts)

    def last_look(self) -> int:
        """About to end without a purchase, with steps to spare: a fitting listing can sit in a row never opened because its words
        differ from the requirement's, and an ending without a purchase is seldom right. Rows of the requested kinds inside the
        budget get read, cheapest first."""
        if self.looked or self.finished or self.cost > SPEND_HARD or self.clock() > CLOCK_WARN:
            return 0
        self.looked = True
        room = min(self.left() - self.tail() - 4, 8)
        read = self.read_below(room, self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else float("inf")) if room > 0 else 0
        self.trace.append({"kind": "last_look", "room": room, "read": read})
        return read

    def need_hits(self, ident: str) -> int:
        """How many requirements in force a listing's record states in the requirement's own words."""
        number = self.item_of(ident)
        text = _dump(self.full.get(ident) or self.rows.get(ident) or {}).lower()
        hits = 0
        for owner, need in self.needs_now():
            words = [w for w in re.findall(r"[a-z0-9]+", need.split(":", 1)[-1].lower()) if len(w) >= 2]
            hits += owner == number and bool(words) and all(w in text for w in words)
        return hits

    def read_below(self, room: int, cap: float | None = None) -> int:
        """Read rows of each item priced below its line (or, given a cap, every row an order could afford), cheapest first, turn by
        turn between the items, side by side when the store shows attributes that way, within `room` steps. Returns how many were read."""
        cart = self.cart_ids()
        lanes: list[list[str]] = []
        for number, _ in enumerate(self.items, 1):
            lines = [self.price_of(i) for i in cart if self.item_of(i) == number and self.price_of(i) is not None]
            units = 1 if self.brief.get("budget_scope") == "each" else max(1, self.items[number - 1].qty)
            top = (cap / units + 1 if cap != float("inf") else cap) if cap is not None else (min(lines) if lines else float("inf"))
            pool = [i for i in self.seen_order if i not in self.full and i not in cart and self.on_item(i, number) and self.buyable(i)
                    and self.price_of(i) is not None and self.price_of(i) < top]
            hosts = [h for part, h, _ in self.pairs if part == number and h]
            devices = [i for h in hosts for _, i in self.basket_pool(h, self.items[h - 1])]
            pool = [i for i in pool if self.fits_model(i, number) and (not devices or any(self.fits_device(i, d) is not False for d in devices))]
            lanes.append(sorted(pool, key=lambda i: (self.narrow.get(i, 10 ** 6) if cap is not None else 0, self.price_of(i) or 0.0)))
        tool, target = self.roles.get("compare"), self.compare_target()
        side = bool(tool and target) and not self.compare_bare()
        size = max(1, self.compare_room()) if side else 1
        before, spent = len(self.full), 0
        while spent < room and not self.finished and any(lanes) and self.clock() < CLOCK_WARN + 30:
            for lane in lanes:
                if not lane or spent >= room or self.clock() >= CLOCK_WARN + 30:
                    continue
                batch, lane[:] = lane[:size], lane[size:]
                spent += 1
                if side:
                    faults: list[str] = []
                    args = _shape({target: batch}, self.schemas.get(tool), "args", faults)
                    reply = self.act(tool, args) if not faults else None
                    if reply is None:
                        continue
                    seen = reply.get("observation")
                    self.observe(tool, seen)
                    self.note_view(tool, seen)
                    rows = self.records_in(seen)
                    if rows:
                        self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
                else:
                    self.fill_facts(batch, self.tail() + 3)
        return len(self.full) - before

    def row_line(self, ident: str) -> str:
        """One search row in a few words: its id, price and title."""
        row = self.rows.get(ident) or {}
        texts = [(k, v) for k, v in row.items() if isinstance(v, str) and v.strip()]
        title = next((v for k, v in texts if re.search(r"title", str(k), re.I)), None) \
            or next((v for k, v in texts if re.search(r"name", str(k), re.I) and not re.search(r"sold|seller|shop|brand", str(k), re.I)), None) \
            or max((v for _, v in texts), key=len, default="")
        price = self.price_of(ident)
        return f"{ident} ({price:g}) {title[:70]}" if price is not None else f"{ident} {title[:70]}"

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

    def page_hold(self) -> str:
        """Once, before a cheapest checkout that wider_hold left alone (too few steps for new searches): a search shows its
        rows a page at a time in no price order, so the searches that found the cart's lines are paged on for cheaper rows."""
        if self.page_held or self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 8 or self.cost > SPEND_SOFT \
                or self.clock() > CLOCK_WARN:
            return ""
        lines = [(i, self.price_of(i)) for i in self.cart_ids() if self.price_of(i) is not None and not self.sold_out(i)]
        if not lines:
            return ""
        self.page_held = True
        self.trace.append({"kind": "page_hold", "lines": [i for i, _ in lines]})
        return ("Not run yet (no step used). The pick is the cheapest fitting listing, and a search shows its rows a page at a time in no price"
                " order, so a cheaper fit can sit on a page not read yet. Before checking out, page on through the search that found each"
                " cart line (pass the next-page token the store returned) while pages remain, and read the attributes of any buyable row"
                " priced below that line: " + "; ".join(f"{i} at {p:g}" for i, p in lines) + ". Swap a line only for a row that meets every"
                " requirement in force, keep a step for the checkout, then check out.")

    def unlisted_musts(self) -> list[str]:
        """The request's own requirements (not the shopper's answers) on a property that no listing seen names anywhere, in an
        attribute key, a title or a description, once at least six listings were read in full: the store does not list that
        property, so no listing can show it either way (usually a word of the story or of the product's name read as a must)."""
        if len(self.full) < 6:
            return []
        spell = {"colour": "color", "grey": "gray", "centre": "center", "metre": "meter", "litre": "liter", "jewellery": "jewelry"}
        texts = [_dump(record).lower() for record in self.full.values()] + [_dump(row).lower() for row in self.rows.values()]
        texts = [re.sub(r"[a-z]+", lambda m: spell.get(m.group(0), m.group(0)), text) for text in texts]
        out = []
        for item in self.items:
            for need in item.musts:
                if need in item.dropped or ":" not in need:
                    continue
                words = [spell.get(w, w) for w in re.findall(r"[a-z]+", need.split(":", 1)[0].lower()) if len(w) >= 3]
                if words and not any(all(re.search(r"\b" + re.escape(w.rstrip("s")) + r"s?\b", text) for w in words) for text in texts):
                    out.append(need)
        return out

    def unlisted_text(self) -> str:
        musts = self.unlisted_musts()
        if not musts:
            return ""
        self.trace.append({"kind": "unlisted", "musts": musts})
        return ("\n\nREQUIREMENTS NO LISTING MAY SHOW (found by code: no listing seen uses these properties' words in an attribute key, a title"
                " or a description). When no record states one of them under any key or plain synonym either (screen for display, memory for"
                " ram), the store does not list that property and it decides nothing: judge every other requirement. When a record does state"
                " it, judge it as usual:\n" + "\n".join(musts))

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
        self.fill_facts(queue[: 3 if self.max_steps <= 40 else 5], 22)

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
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        guide = PLAN
        if any(re.search(LOOK_KEY, name.lower()) for _, name, _, _ in table):
            # the listings carry a look key besides colour: a hint about looks is weighed against those keys too
            guide = PLAN.replace(LOOKS_KYO, LOOKS_KEYED)
        if self.lists_on:
            guide = guide.replace("Choose up to COUNT, best first.", f"Choose up to {PLAN_NAMES} properties for EACH item, best first, ranking every item on its own.")
        prompt = guide.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 2800 if self.lists_on else 1800) or {}
        ordered: list[tuple[int, str]] = []
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
                    self.hinted.add((number, _words(label)))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        if self.lists_on:
            self.ask_lists(ordered, sent, told)
            count = 0 if self.lists_on else self.msgs_free()
            ordered = [pick for pick in ordered if (pick[0], _words(pick[1])) not in self.named]
        # all messages but the last first; the last re-asks a question whose answer wanted a single question
        # (named with example values this time), else it goes to the next pick
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head]) if count > 0 else []
        if not batch and not sent:
            return ""
        sent += batch
        if batch:
            heard = self.hear(batch)
            self.take_heard(heard, told)
        if batch and count >= 2 and self.msgs_free() > 0:
            # the last message goes to a property not asked yet: asking one again never drew a preference (the shopper
            # answers a property the same way twice; race 10: 0 of 38 repeats), so the planner's next pick, else the
            # most carried key nobody has asked about
            asked = [(n, _words(lab)) for n, lab in head]
            # an item whose answers held nothing yet first: a held-back preference is on one property of one product,
            # so a product that already gave one is the less likely place for the next
            quiet = {n for n in range(1, len(self.items) + 1) if not self.items[n - 1].told}
            rest = [(n, lab) for n, lab in ordered[len(head):] if (n, _words(lab)) not in asked]
            rest.sort(key=lambda pick: pick[0] not in quiet)
            again = next(((n, lab, self.question(n, lab)) for n, lab in rest), None)
            # a device's processor: a shopper often holds it back, and an episode that never names it cannot hear it, so
            # when the listings mostly carry a processor key that no question named, the last message asks it
            chip = self.chip_pick(asked)
            if chip:
                again = (chip[0], chip[1], self.question(chip[0], chip[1]))
                self.trace.append({"kind": "chip_pick", "item": chip[0], "label": chip[1]})
            if again is None:
                spare = self.spare_pick(asked)
                again = (spare[0], spare[1], self.question(spare[0], spare[1])) if spare else None
            if again:
                batch = self.send([again])
                if batch:
                    sent += batch
                    # one answer: the light model reads it
                    self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
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
        return summary

    def ask_lists(self, ordered: list[tuple[int, str]], sent: list[tuple[int, str, str]], told: list[str]) -> None:
        """Spend the free messages on lists: one message names several properties of one item. The shopper answers about
        the first named property it tracks, so a property it has no view on hides whatever stands behind it: the order
        changes between messages. A reply asking for one question at a time turns lists off for the episode."""
        self.lanes = {n: [] for n in range(1, len(self.items) + 1)}
        self.lane_order = []
        for number, label in ordered:
            self.lanes[number].append(label)
            if number not in self.lane_order:
                self.lane_order.append(number)
        self.fill_lanes(self.lanes)
        self.lane_turns = {n: 0 for n in self.lanes}
        self.walk_lanes(sent, told)

    def walk_lanes(self, sent: list[tuple[int, str, str]], told: list[str], most: int = 99) -> None:
        """Send lists off the lanes while messages are free, `most` at the most."""
        lanes, spent, order = self.lanes, self.lane_turns, self.lane_order
        while most > 0 and self.msgs_free() > 0 and not self.finished:
            for number, labels in lanes.items():
                lanes[number] = [label for label in labels if not self.settled(number, label)]
            live = [n for n in lanes if lanes[n]]
            if not live:
                return
            # an item that gave nothing yet first, then the one asked least, then the planner's order
            number = min(live, key=lambda n: (bool(self.items[n - 1].told), spent[n], order.index(n) if n in order else len(order)))
            wide = min(LIST_WIDE, LIST_STEPS[min(spent[number], len(LIST_STEPS) - 1)]) if self.lists_on else 1
            names, text = self.fit_list(number, lanes[number][:wide])
            if not names:
                lanes[number] = []
                continue
            batch = self.send([(number, names[0], text)])
            if not batch:
                return
            most -= 1
            sent += batch
            spent[number] += 1
            self.lists_sent += 1
            self.named.update((number, _words(name)) for name in names)
            before = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if _items(heard.get("needs")):
                # a need decides the order, and the light model sometimes turns one round or keeps a linking word: the heavy one reads it
                heard = self.hear(batch, (HEAVY, LIGHT)) or heard
            if len(names) > 1:
                heard.pop("free", None)  # "no view" answers one property of the list, not all of them
            self.take_heard(heard, told)
            new = [need for item, size in zip(self.items, before) for need in item.told[size:]]
            crowded = any(_count(x) == 1 for x in _items(heard.get("crowded")))
            self.trace.append({"kind": "list_ask", "item": number, "names": names, "new": new, "crowded": crowded})
            if len(names) > 1 and crowded and not new:
                self.lists_on = False
                self.named.difference_update((number, _words(name)) for name in names)
                return
            rest = lanes[number][len(names):]
            if len(names) == 1:
                lanes[number] = rest
            elif new:
                hit = self.answered(names, new)
                lanes[number] = names[hit + 1:] + rest
            else:
                # a miss says nothing about the names: the reply line is the task's own. A name at the head is answered four times in
                # five, one behind it about half the time, so the next list leads with new names and gives those a better place
                coming = LIST_STEPS[min(spent[number], len(LIST_STEPS) - 1)]
                if coming < len(names):
                    # the next message is narrower: the same names from the top, where a shopper who passes over lists can see them
                    lanes[number] = names + rest
                elif len(names) <= 2 and rest:
                    lanes[number] = rest + names    # (narrow already: the next names, in the planner's order)
                elif rest:
                    again = names[:0:-1]
                    lanes[number] = [name for pair in zip(rest + [""] * len(again), again + [""] * len(rest)) for name in pair if name]
                    lanes[number][2:2] = names[:1]  # (one in five stays unanswered at the head: it comes back third, where about half are)
                else:
                    # no new names left: each list moves every name on by a third, so none sits in one place through the messages
                    turn = max(1, len(names) // 3)
                    lanes[number] = names[turn:] + names[:turn]

    def settled(self, number: int, label: str) -> bool:
        """Whether the request or an answer already fixed this property of the item."""
        item = self.items[number - 1]
        words = _stem(label)
        return any(":" in need and (_stem(need.split(":", 1)[0]) == words or (len(_stem(need.split(":", 1)[0])) >= 2 and _stem(need.split(":", 1)[0]) <= words))
                   for need in item.told + item.musts)

    def fill_lanes(self, lanes: dict[int, list[str]]) -> None:
        """Top the planned lists up from the listings' keys, enough for every message an item can get, then order each lane: what the
        story pointed at and the planner's first two lead (its judgement beats a key's count at the head of a list), and behind
        them the most carried keys (a key's count beats its judgement further down)."""
        width = LIST_WIDE * max(1, -(-max(1, self.msg_limit) // max(1, len(lanes))))
        asked = [(n, _words(label)) for n, labels in lanes.items() for label in labels]
        chip = self.chip_pick(asked)
        if chip:
            lanes[chip[0]].insert(1, chip[1])
            asked.append((chip[0], _words(chip[1])))
        for _ in range(width * len(lanes)):
            if all(len(labels) >= width for labels in lanes.values()):
                break
            spare = self.spare_pick(asked)
            if not spare:
                break
            asked.append((spare[0], _words(spare[1])))
            if len(lanes[spare[0]]) < width:
                lanes[spare[0]].append(spare[1])
        for number, labels in lanes.items():
            head = [x for x in labels if (number, _words(x)) in self.hinted]
            head += [x for x in labels[:2] if x not in head]
            tail = [x for x in labels if x not in head]
            tail.sort(key=lambda x: -self.carried(number, x))
            lanes[number] = head + tail

    def carried(self, number: int, label: str) -> int:
        """How many read listings of an item carry the key this label names; 0 for a property no key names."""
        words = {w for w in _stem(self.items[number - 1].shelf or "") | _stem(self.items[number - 1].name) if len(w) >= 4}
        return next((count for shelf, name, count, _ in self.key_table(30) if _words(name) == _words(label)
                     and (len(self.items) == 1 or ({w for w in _stem(shelf) if len(w) >= 4} & words))), 0)

    def late_list(self) -> str:
        """About to buy with a message still unsent: what the interview kept for the ending is free once no ending without a purchase
        is coming. One more list goes out, and the checkout waits for what it brings."""
        if self.late_asked or not self.lanes or self.msg_used >= self.msg_limit or self.left() < 8 or self.clock() > CLOCK_WARN:
            return ""
        if any(p["when"] == "order" and p["messages"] > 0 for p in self.process):
            return ""
        self.late_asked = True
        held, self.reserve = self.reserve, 0
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        self.walk_lanes(sent, told, 1)
        self.reserve = held
        self.trace.append({"kind": "late_list", "sent": len(sent), "told": told})
        if not told:
            return ""
        self.version += 1
        return ("Not run yet (no step used). Kyo used a spare message: " + " ".join(f"Q: {q} A: {a}" for _, q, a in sent)
                + "\nNew firm requirements: " + "; ".join(told) + ". Apply them to the cart, then check out.")

    def fit_list(self, number: int, names: list[str]) -> tuple[list[str], str]:
        """The longest leading part of `names` whose question the message tool takes whole, with that question."""
        said = re.sub(r"(?i)\band\b", "or", self.items[number - 1].name)      # (a name joining nouns with "and" draws one-at-a-time replies)
        for size in range(len(names), 0, -1):
            part = names[:size]
            text = self.question(number, part[0]) if size == 1 else f"What {', '.join(part[:-1])} or {part[-1]} would you like for the {said}?"
            args = self.message_args(text)
            if args and text in args.values():
                return part, text
        return [], ""

    def answered(self, names: list[str], new: list[str]) -> int:
        """The place in `names` of the property a new need answers; 0 when the answer used other words for it."""
        said = set().union(*[_stem(need.split(":")[0]) for need in new]) if new else set()
        for place, name in enumerate(names):
            if _stem(name) & said:
                return place
        return 0

    def chip_pick(self, asked: list[tuple[int, str]]) -> tuple[int, str] | None:
        """The processor key of an item whose read listings mostly carry one, when no question named a processor yet and
        the item's requirements do not settle it."""
        chip_word = r"\b(processor|cpu|chip)\b"
        if any(re.search(chip_word, w) for _, w in asked):
            return None
        table = self.key_table()
        for number, item in enumerate(self.items, 1):
            words = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            rows = [(name, carried) for label, name, carried, _ in table
                    if len(self.items) == 1 or ({w for w in _stem(label) if len(w) >= 4} & words)]
            if not rows or re.search(chip_word, " ".join(item.musts + item.told).lower()):
                continue
            most = max(c for _, c in rows)
            chip = next(((name, c) for name, c in rows if re.search(chip_word, name)), None)
            if chip and most >= 3 and chip[1] * 2 >= most:
                return number, chip[0]
        return None

    def spare_pick(self, asked: list[tuple[int, str]]) -> tuple[int, str] | None:
        """A key from the listings' table for a message the plan left unused: an item asked about least first, its most
        carried key that no question named and the request does not settle (purchase variants and logistics left out)."""
        counts = {n: sum(1 for m, _ in asked if m == n) for n in range(1, len(self.items) + 1)}
        for number in sorted(counts, key=lambda n: counts[n]):
            item = self.items[number - 1]
            settled = " ".join(item.musts + item.told).lower()
            words = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            for label, name, carried, _ in self.key_table(30):
                if len(self.items) > 1 and not ({w for w in _stem(label) if len(w) >= 4} & words):
                    continue
                # an identifier is left out, but the vehicle or device a part is made to fit is a property
                fitted = re.search(r"\b(vehicle|car|motorcycle|truck|van|device)\b", name)
                if re.search(r"\b(options?|series|warranty|condition|sku|code|id)\b", name) or (re.search(r"\bmodel\b", name) and not fitted) or carried < 2:
                    continue
                if any(n == number and (w == name or w in name or name in w) for n, w in asked) or name in settled:
                    continue
                return number, name
        return None

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
            said, notes = self.reply_parts(answer)
            words = self.spoken(said)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if notes and not self.finished:
                # a notice the store sent with the answer (a new budget, a dropped need) is read as a note, so it is
                # neither taken for the answer nor lost
                meaning = self.take_note(notes)
                self.trace.append({"kind": "reply_note", "lines": notes, "meaning": meaning})
                if meaning:
                    self.queued.append(f"(sent with a reply) {meaning}")
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def reply_parts(self, observation: Any) -> tuple[Any, list[str]]:
        """A shopper's reply can carry the store's notices beside the answer, as a list of lines under a key of their own:
        returns the answer part and those lines (not yet read)."""
        if not isinstance(observation, dict):
            return observation, []
        notes = [str(line) for value in observation.values() if isinstance(value, list)
                 for line in value if isinstance(line, str) and len(line.split()) >= 3]
        if not notes:
            return observation, []
        said = {key: value for key, value in observation.items() if not isinstance(value, list)}
        return (said or observation), [line for line in notes if line not in self.heard]

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

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.ask_json(prompt, list(models), 1500) or {}
        # a sum the shopper wrote in cents stays in cents (the store's prices are in the same unit): 1500 cents is 1500
        said, budget = _cents_said(" ".join(a for _, _, a in sent)), got.get("budget")
        if said and isinstance(budget, (int, float)) and not isinstance(budget, bool) and abs(budget * 100 - said) < 0.5:
            got["budget"] = said
        return got

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
        answered = False
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
            answered = True
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
        if (not answered and stage == "order" and any(p["messages"] > 0 for p in steps) and self.roles.get("ask")
                and self.msg_used < self.msg_limit and self.left() >= 2):
            # no model to word the step: a message the code writes, naming the final cart, still comes before the order
            text = self.cart_message()
            if text and self.send([(1, "", text)], 1):
                self.gate_msgs += 1
                said.append(f"Q: {text}")
                self.trace.append({"kind": "gate_code_message", "text": text[:200]})
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def cart_message(self) -> str:
        """The cart in words for the shopper: each line's listing name, id, units and price, and the total when known."""
        counts = {i: n for i, n in self.store_counts().items() if n > 0} or dict(self.cart_lines)
        parts, total = [], 0.0
        for ident, units in counts.items():
            row = self.rows.get(ident) or {}
            name = next((str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"title|headline", k.lower())), "") \
                or next((str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"name|label", k.lower()) and not re.search(r"sold|seller|shop|brand", k.lower())), "")
            price = self.code_price(ident)
            total += (price or 0) * units
            parts.append(f"{name[:80] + ' ' if name else ''}({ident}) x{units}" + (f" at {price:g} each" if price is not None else ""))
        if not parts:
            return ""
        return "Before I place the order, here is what I picked: " + "; ".join(parts) + (f". Total {total:g}." if total else ".") + " Please confirm."

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
        said = _cents_said(" ".join(lines))
        if said and isinstance(budget, (int, float)) and not isinstance(budget, bool) and abs(budget * 100 - said) < 0.5:
            budget = said       # written in cents, kept in cents: "1500 cents" is 1500, not 15.0
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

    def restated(self, block: dict[str, Any]) -> str:
        """A record's figures in the units the requirements in force are written in, where they differ ("folded length 20.5 in = 52.07 cm")."""
        self.learn_units(block)
        found: list[str] = []
        for key, value in block.items():
            for unit, number in _amounts(value):
                for (have, want), size in self.factors.items():
                    if have == unit and size:
                        found.append(f"{key} {number:g} {unit} = {number * size:.4g} {want}")
        return "; ".join(list(dict.fromkeys(found))[:8])

    def learn_units(self, block: dict[str, Any]) -> None:
        """The sizes of the unit pairs this record meets, where a key of it is one a requirement in force talks about and the two write
        that figure in different units. Worked out once a pair, for this task only."""
        pairs: set[tuple[str, str]] = set()
        for _, need in self.needs_now():
            wanted = {u for u, _ in _amounts(need) if u}
            if not wanted:
                continue
            prop, low = _stem(need.split(":", 1)[0]), need.lower()
            for key, value in block.items():
                if _names(prop, _stem(key)) or str(key).replace("_", " ").lower() in low:
                    pairs.update((u, w) for u, _ in _amounts(value) if u for w in wanted if w != u)
        fresh = sorted(p for p in pairs if p not in self.factors)[:10]
        if not fresh or self.cost > SPEND_HARD or self.clock() > CLOCK_WARN + 30:
            return
        got = self.ask_json(UNITS_ASK + "\n\n" + "\n".join(f"{a} -> {b}" for a, b in fresh), [LIGHT, HEAVY], 200) or {}
        sizes = _items(got.get("factors"))
        for number, pair in enumerate(fresh):
            size = sizes[number] if number < len(sizes) else None
            self.factors[pair] = float(size) if isinstance(size, (int, float)) and not isinstance(size, bool) and size > 0 else None

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
                hits = [f"{k}={str(v)[:50]}" for k, v in _own_keys(prop, block).items()]
                tag = f"(item {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            same = self.restated(block)
            if same:
                parts.append("in the requirements' units: " + same)
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        # (code's figure comparisons go to the cart check only, where the pick is decided: the turn model's view stays as it was)
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    def limit_mark(self, need: str, block: dict[str, Any]) -> str:
        """' [meets at least 18.5 inches]' or ' [fails ...]' for a numeric limit whose figure the listing's matching keys
        state in comparable units; '' otherwise."""
        limit = _limit(need)
        if not limit:
            return ""
        prop = need.split(":", 1)[0].strip()
        self.learn_units(block)
        verdict = _limit_verdict(limit, list(_own_keys(prop, block).values()), self.factors)
        return f" [{verdict} {need.split(':', 1)[-1].strip()}]" if verdict else ""

    def other_values(self, ids: list[str]) -> str:
        """Cart listings whose own key for a required property (a stated or a heard one, not a limit or a value read off
        another page) holds none of the wanted value's words."""
        lines = []
        for ident in ids:
            block = _fact_block(self.full.get(ident) or {}) or {}
            if not block:
                continue
            for _, need in self.needs_now():
                if ":" not in need or _limit(need) or any(need == ref for _, ref in self.refs):
                    continue
                prop, want = (x.strip().lower() for x in need.split(":", 1))
                if re.match(r"(anything|not |no |unlike|other than|except|either|any )", want) or " or " in want:
                    continue
                keys = {k: str(v).lower() for k, v in _own_keys(prop, block).items()}
                words = _stem(want)
                if keys and words and not any(words & _stem(v) for v in keys.values()):
                    lines.append(f"{ident}: {prop} wanted {want!r}; the listing says " + ", ".join(f"{k}={v[:40]!r}" for k, v in list(keys.items())[:2]))
        return "\n".join(lines)

    def limit_lines(self, ids: list[str]) -> str:
        """Kyo's own comparisons of the numeric limits in force with the figures the listings' attributes state."""
        lines = []
        for ident in ids:
            block = _fact_block(self.full.get(ident) or {}) or {}
            for _, need in self.needs_now():
                mark = self.limit_mark(need, block) if block else ""
                if mark:
                    lines.append(f"{ident}: {need.split(':', 1)[0].strip()}{mark}")
        return "\n".join(lines)

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
            lines.append(_dump(_clip(record, 90))[:520])
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
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + self.unlisted_text()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        limits = self.limit_lines(ids)
        if limits:
            prompt += "\n\nFIGURES COMPARED BY CODE (numeric limits in force against the attributes, in the same unit; trust these):\n" + limits
        fitted = self.fit_lines(ids)
        if fitted:
            prompt += "\n\nFIT RULES COMPARED BY CODE (each part's Fits line against the device's model number and other names; trust these):\n" + fitted
        others = self.other_values(ids)
        if others:
            prompt += ("\n\nKEYS STATING ANOTHER VALUE (found by code: the listing's own key for a required property holds none of the"
                       " required value's words; a plain synonym or spelling can still mean the same, so judge each):\n" + others)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 8)
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
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def resolve_refs(self) -> bool:
        """Once, when the loop stopped: the chain model reads the pages already opened (those first) for each requirement
        whose value is on another listing and has none yet; returns whether a value was found."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if not pending or getattr(self, "refs_resolved", False):
            return False
        self.refs_resolved = True
        opened = [i for i in reversed(self.seen_order) if i in self.opened and (self.full.get(i) or self.rows.get(i))]
        rows = [i for i in reversed(self.seen_order) if i not in self.opened and self.rows.get(i)]
        records = ["OPENED " + _dump(_clip(self.full.get(i) or self.rows.get(i), 300))[:1400] for i in opened[:40]]
        records += ["ROW " + _dump(_clip(self.rows.get(i), 80))[:300] for i in rows[: max(0, 60 - len(records))]]
        if not opened:
            return False
        prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending) + "\n\nRECORDS (opened pages first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
        found = False
        for chain in _items(got.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
            ids = [str(link.get("id") or "").strip() for link in _items(chain.get("links")) if isinstance(link, dict)]
            if need and chain.get("done") is True and ids and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                self.ref_values[need] = str(chain["value"]).strip()
                self.version += 1
                found = True
        self.trace.append({"kind": "resolve_refs", "pending": pending, "values": dict(self.ref_values)})
        return found

    def recheck_lines(self) -> None:
        """A carted line that a newly read value clearly rules out (by code) is swapped for the read listing of that item
        that meets the most requirements with none broken (the cheapest among equals)."""
        cart = self.roles.get("cart")
        if not cart:
            return
        plan = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if not self.sold_out(i)}
        swaps = {}
        for ident in list(plan):
            number = self.item_of(ident)
            if number is None or self.fit_score(ident, self.items[number - 1]) is not None:
                continue
            item = self.items[number - 1]
            ranked = []
            for other in self.full:
                if other == ident or self.sold_out(other) or not self.code_buyable(other) or self.item_of(other) != number:
                    continue
                score = self.fit_score(other, item)
                if score is not None:
                    price = self.code_price(other)
                    ranked.append((-score, price if price is not None else float("inf"), other))
            if ranked:
                swaps[ident] = sorted(ranked)[0][2]
        if not swaps:
            return
        for old, new in swaps.items():
            plan[new] = plan.pop(old)
        if self.budget is not None:
            prices = [self.code_price(i) for i in plan]
            if any(p is None for p in prices) or sum(p * plan[i] for p, i in zip(prices, plan)) > self.budget:
                return
        calls = self.cart_calls(plan, list(swaps))
        if not calls or len(calls) + 1 > self.left():
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
            self.absorb(reply.get("observation"))
        self.trace.append({"kind": "recheck_lines", "swaps": swaps})

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
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    same = self.restated(_fact_block(record) or {})
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400] + (f" [same figures: {same}]" if same else ""))
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
        # (a hold sends the turn model searching: it must still have the steps for that)
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 2 * len(self.items) + 7
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
        # a line last seen before it was held is stale too: holding it can be what changed it
        stale = [i for i in self.cart_ids() if i not in self.gone
                 and (self.offer_seen.get(i, -99) < since or self.offer_seen.get(i, -99) < self.held_at.get(i, -99))]
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

    def store_counts(self) -> dict[str, int]:
        """Units per listing as the store's newest cart reply lists them (empty when that reply lists no lines with counts)."""
        found: dict[str, int] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(value, dict):
                ident = next((v for v in value.values() if isinstance(v, str) and v in self.rows), None)
                count = next((v for k, v in value.items() if isinstance(v, int) and not isinstance(v, bool) and k != self.price_key
                              and re.search(r"count|unit|qty|quant", k.lower()) and not re.search(r"price|cost|due", k.lower())), None)
                if ident and count is None:
                    # no key says so: the one whole number on the line besides its price
                    rest = [v for k, v in value.items() if isinstance(v, int) and not isinstance(v, bool) and k != self.price_key
                            and not re.search(r"price|cost|total|due", k.lower())]
                    count = rest[0] if len(rest) == 1 else None
                if ident and count is not None:
                    found[ident] = count
                    return
                for inner in value.values():
                    walk(inner, depth + 1)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner, depth + 1)

        walk(self.cart_obs, 0)
        return found

    def composition(self) -> str:
        """What keeps the cart from matching the request line for line, judged by code from the store's own cart reply: an
        item with no line or several listings, a unit count other than its quantity, or a line from another shelf than the
        one the request names the product by. Empty when nothing is clearly off."""
        counts = {i: n for i, n in self.store_counts().items() if n > 0}
        cat_key = self.category_key()
        if not counts or not cat_key:
            return ""
        seen = {str(r.get(cat_key)).strip() for r in self.rows.values() if isinstance(r.get(cat_key), str)}
        shelves = sorted(seen | self.shelf_labels({s.lower() for s in seen}))

        def same(one: str, two: str) -> bool:
            return bool(one and two) and (_stem(one) == _stem(two) or re.sub(r"[^a-z0-9]", "", one.lower()) == re.sub(r"[^a-z0-9]", "", two.lower()))

        shelf_of = {i: str((self.rows.get(i) or {}).get(cat_key) or "").strip() for i in counts}
        # the shelf the request names each product by, when the store has a shelf of that name
        named = {n: next((s for s in shelves if same(s, item.shelf or "")), None) or next((s for s in shelves if same(s, item.name)), None) for n, item in enumerate(self.items, 1)}
        owner: dict[str, int | None] = {}
        for ident in counts:
            by_name = [n for n, s in named.items() if s and same(s, shelf_of[ident])]
            owner[ident] = by_name[0] if by_name else self.item_of(ident)
        stray = [i for i, n in owner.items() if n is None or named.get(n) and not same(named[n], shelf_of[i])]
        problems = []
        for number, item in enumerate(self.items, 1):
            ids = [i for i, n in owner.items() if n == number and i not in stray]
            if not ids and named[number] and stray:
                problems.append(f"item {number} ({item.name}): the store has a shelf named '{named[number]}' for it, but no cart line comes from it ("
                                + ", ".join(f"{i} is from '{shelf_of[i]}'" for i in stray) + f"); take a listing from '{named[number]}' instead")
            elif len(ids) > 1:
                problems.append(f"item {number} ({item.name}) has {len(ids)} different listings in the cart ({', '.join(ids)}); keep only one")
            elif len(ids) == 1 and number in self.caps and (self.code_price(ids[0]) or 0) > self.caps[number]:
                problems.append(f"item {number} ({item.name}): {ids[0]} costs {self.code_price(ids[0]):g} a unit, over the {self.caps[number]:g} the request allows for it;"
                                " take a listing at or under that")
            elif len(ids) == 1 and counts[ids[0]] != item.qty:
                problems.append(f"item {number} ({item.name}) needs {item.qty} units of its listing in the cart and has {counts[ids[0]]}"
                                f" ({ids[0]}); the quantity counts cart units, whatever pack size a title states")
        return "; ".join(problems)

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
        if self.cost > SPEND_SOFT:
            return []
        # the closest listings of the lost line's own item: read ones first, then by requirements stated, then (for a cheapest pick) by price
        kinds = {self.item_of(i) for i in lost}
        cheapest = self.brief.get("pick") == "cheapest"
        pool = sorted(reversed(self.seen_order), key=lambda i: (self.item_of(i) not in kinds, i not in self.full, -self.need_hits(i),
                                                                (self.price_of(i) or 0.0) if cheapest else 0.0))
        records = []
        for ident in pool:
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 12:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (closest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200)
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
        # a second review only after the task changed (a sell-out, a new budget or requirement): the first one is stale then
        if self.cost > SPEND_SOFT or (self.audited and (self.audit_seen == self.version or self.audits >= 2)):
            return ""
        self.audited = True
        self.audit_seen = self.version
        self.audits += 1
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

    def audit_call(self, models: list[str] | None = None) -> dict[str, Any] | None:
        records = []
        for ident in sorted(reversed(self.seen_order), key=lambda i: -self.need_hits(i)):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            same = self.restated(_fact_block(record) or {}) if record else ""
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200] + (f" [same figures: {same}]" if same else "")) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 16:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + self.unlisted_text()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        figures = self.limit_lines([i for i in self.seen_order if i in self.full and not self.sold_out(i)][:16])
        if figures:
            prompt += "\n\nFIGURES COMPARED BY CODE (units converted):\n" + figures
        got = self.ask_json(prompt, models or [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "audit", "got": got})
        self.audit_memo = (self.version, got)
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

    def checkout_lines(self, ids: list[str]) -> str:
        """Set the cart to these listings (one line per item at its quantity, nothing else) and check out, with the steps
        the request asks for at the order; returns what was done, or '' when it cannot be done here."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy:
            return ""
        plan: dict[str, int] = {}
        for ident in ids:
            number = self.item_of(ident)
            if number and not any(self.item_of(p) == number for p in plan):
                plan[ident] = self.items[number - 1].qty
        if len(plan) < len(self.items):
            return ""
        calls = self.cart_calls(plan, [i for i in self.cart_ids() if i not in plan])
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
        go, said = self.gate("order")
        if not go:
            return "Kyo set the cart to the checked lines; " + said
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        reply = self.act(buy, args)
        if reply is None:
            return ""
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return ("Not run (Kyo acted instead): an independent check passed the cart built earlier (" + ", ".join(plan)
                + "), so Kyo set the cart to it and checked out. Store reply: " + self.show(reply.get("observation")))

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

    def settle(self) -> str:
        """A first checkout sets off the store's and the shopper's changes, and answering them takes steps. So once few steps are left, a
        line for each item that shows every requirement in force goes out when the cart check passes, instead of more reading."""
        # (race 16, episodes of up to 40 steps: a first checkout before the 25th call passed about half the time, after the 30th a fifth)
        if self.buys or self.finished or not self.asked or self.settled_at == self.version or self.left() > 11 + 2 * len(self.items):
            return ""
        live = {need for _, need in self.needs_now()}
        if any(need in live and need not in self.ref_values for _, need in self.refs):
            return ""
        self.settled_at = self.version
        if self.brief.get("pick") == "cheapest":
            self.sweep()

        def picks() -> list[str]:
            lines = []
            for number, item in enumerate(self.items, 1):
                shown = [(p, i) for unmet, p, i in self.fitting(number, item) if not unmet and not self.sold_out(i)]
                if shown:
                    lines.append(min(shown)[1])
            return self.fitted(lines)

        lines = picks()
        if len(lines) < len(self.items) or "does NOT fit" in self.fit_lines(lines):
            return ""
        if self.budget is not None and self.brief.get("budget_scope") != "each":
            if sum((self.price_of(i) or 0.0) * self.items[(self.item_of(i) or 1) - 1].qty for i in lines) > self.budget:
                return ""
        self.check_cart(lines, SPEND_HARD)
        if not self.verdicts.get(self.check_key(lines)):
            self.checked.discard(self.check_key(lines))     # (the check at the checkout must still see these lines and hold them)
            return ""
        self.trace.append({"kind": "settle", "lines": lines, "step": self.step})
        return self.checkout_lines(lines)

    def fitted(self, lines: list[str]) -> list[str]:
        """Lines for a code-made order: where a part must fit a device, the cheapest basket in which each does, when the lines break that."""
        if not self.pairs or "does NOT fit" not in self.fit_lines(lines):
            return lines
        basket = self.cheapest_basket()
        return list(basket.values()) if len(basket) >= len(self.items) else lines

    def last_order(self) -> str:
        """Before an ending without a purchase, once for each state of the task: every page read is judged again against what is in
        force now (a requirement may have been dropped since it was read, a figure may need another unit) and what fits is bought. Only
        listings no attribute rules out are bought: a line known to break a requirement cannot pass either."""
        # (race 16: an ending without a purchase was right 14 times in 936, all on one task of 90)
        if self.last_tried == self.version or not self.full or self.left() < len(self.items) + 1:
            return ""
        self.last_tried = self.version
        # (the review is the dearest call of an episode: what it said of this state of the task stands, and a new state gets the light model)
        memo = self.audit_memo
        fits = self.audit_fits(memo[1] if memo and memo[0] == self.version else self.audit_call([LIGHT, HEAVY]))
        picks: dict[int, str] = {}
        for number, item in enumerate(self.items, 1):
            named = [i for i in fits.get(str(number), []) if i in self.full and self.on_item(i, number) and self.fit_score(i, item) is not None]
            pool = [(u, p, i) for u, p, i in self.fitting(number, item) if not self.sold_out(i)]
            if self.brief.get("pick") == "cheapest":
                named.sort(key=lambda i: self.price_of(i) or 0.0)
            pick = named[0] if named else (pool[0][2] if pool else None)
            if pick:
                picks[number] = pick
        if len(picks) < len(self.items):
            return ""
        if self.budget is not None:
            prices = [self.price_of(i) for i in picks.values()]
            units = [self.items[n - 1].qty for n in picks]
            if any(p is None for p in prices) or (self.brief.get("budget_scope") != "each" and sum(p * q for p, q in zip(prices, units)) > self.budget):
                return ""
        lines = self.fitted(list(picks.values()))
        if self.verdicts.get(self.check_key(lines)) is False:
            return ""       # (the check refused this very cart in this state of the task)
        self.trace.append({"kind": "last_order", "picks": lines, "named": bool(fits)})
        return self.checkout_lines(lines)

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
        elif left <= 2 * len(self.items) + 7 and self.buys == 0:
            # what the first checkout sets off takes steps too: reading stops while two rounds of changes can still be answered
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

    def clear_item(self, name: str, args: dict[str, Any]) -> str:
        """An order holds one line an item: a new listing for an item that still has a line takes that line's place, whoever carted it.
        Says why not when too few steps are left for both."""
        if name != self.roles.get("cart") or self.finished:
            return ""
        spec = (self.schemas.get(name) or {}).get("properties") or {}
        taken = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        removal = any(isinstance((spec.get(k) or {}).get("enum"), list) and v in spec[k]["enum"][1:] for k, v in args.items())
        count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        if len(taken) != 1 or removal or count == 0:
            return ""
        number = self.item_of(taken[0])
        old = [i for i in self.cart_ids() if i != taken[0] and number and self.item_of(i) == number]
        calls = self.cart_calls({}, old) if old else None
        if not calls:
            return ""
        if self.left() < len(calls) + 2:
            return f"too few steps are left to replace {', '.join(old)} and check out. Check out the cart as it is, or end the episode."
        for call in calls:
            faults: list[str] = []
            call = _shape(call, self.schemas.get(name), "args", faults)
            reply = self.act(name, call) if not faults else None
            if reply is None or self.finished:
                return ""
            self.note_cart(name, call)
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "clear_item", "old": old, "new": taken[0]})
        return ""

    def code_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing given by a code that its row shows (a part number in the title) where a tool takes its id: the row's id goes out
        instead, so no step is spent on a reference the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None}:
            return args

        def fix(value: Any) -> Any:
            word = value.strip().lower() if isinstance(value, str) else ""
            if len(word) < 4 or value.strip() in self.rows or " " in word:
                return value
            found = re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")
            hits = [i for i, row in self.rows.items() if found.search(_dump(row).lower())]
            return hits[0] if len(hits) == 1 else value

        out = {k: ([fix(x) for x in v] if isinstance(v, list) else fix(v)) for k, v in args.items()}
        if out != args:
            self.trace.append({"kind": "code_ids", "tool": name})
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
        text = _dump(_clip_reply(observation, SHOW_CHARS))
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
            return max(1, min(self.compare_bad - 1, max(self.compare_ok, 1)))
        # two until the store shows more works: a side-by-side view can refuse more without saying why, and each refusal costs a step
        return max(2, self.compare_ok)

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
            args = self.code_ids(name, args)
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            wrong = self.clear_item(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room():
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
                    wider = self.late_list() or self.sweep() or self.wider_hold() or self.page_hold()
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
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the cart against the request line for line, by code, at most twice so a misread cannot stall the order
                    off = self.composition() if self.comp_held < 2 and self.left() > 3 else ""
                    if off:
                        self.comp_held += 1
                        self.trace.append({"kind": "composition", "off": off[:300]})
                        out.append((cid, "Not run yet (no step used). The cart does not match the request line for line: " + off
                                    + ". Fix the cart (one listing per requested item, at its quantity), then check out.\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5:
                        self.last_look()
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
            if self.is_refuse(name, args) and len(self.items) > 1 and not self.several_held:
                # an order of several products: an ending without a purchase has passed on none of them (race 15: 0 of 1300), so with
                # a line for each product the order goes out, whatever the last doubts; only a single product can truly have no fit
                self.several_held = True
                lines = [i for i in self.built if not self.sold_out(i)]
                if len({self.item_of(i) for i in lines} - {None}) < len(self.items):
                    picks = self.code_lines()
                    lines = list(picks) if len(picks) >= len(self.items) else lines
                lines = self.fitted(lines)
                if len({self.item_of(i) for i in lines} - {None}) >= len(self.items):
                    self.trace.append({"kind": "several_held", "lines": lines})
                    done = self.checkout_lines(lines)
                    if done:
                        out.append((cid, done + (" [episode finished]" if self.finished else "")))
                        continue
            if self.is_refuse(name, args) and self.left() > 5 and not self.decline_held:
                # a cart with a line for every item was built and is about to be thrown away: when an independent check
                # passes it, it is checked out instead (an ending without a purchase almost never passed: race 9 none of
                # 247 on tasks someone solved, race 10 18 of 195)
                lines = [i for i in self.built if not self.sold_out(i)]
                if len(lines) < len(self.items):
                    # nothing full was carted (the usual way a solvable task got declined): the code's own picks from the
                    # pages read, for the independent check to judge
                    picks = self.code_lines()
                    lines = list(picks) if len(picks) >= len(self.items) else lines
                if len(lines) >= len(self.items):
                    self.decline_held = True
                    self.fill_facts(lines, 6)
                    self.check_cart(lines, SPEND_HARD)
                    if self.verdicts.get(self.check_key(lines)):
                        # (Kyo acts itself: told to check out instead, the turn model went on searching and ended unbought)
                        self.trace.append({"kind": "decline_held", "lines": lines})
                        done = self.checkout_lines(lines)
                        if done:
                            out.append((cid, done + (" [episode finished]" if self.finished else "")))
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
                done = self.complete_cart() or self.last_order()
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
            rerank = self.rerank_hint()
            if rerank:
                text += "\n" + rerank
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def item_of(self, ident: str) -> int | None:
        """The requested item a listing is a candidate for, by its category label's words."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        if not label:
            return 1 if len(self.items) == 1 else None
        flat = re.sub(r"[^a-z0-9]", "", label.lower())
        for number, item in enumerate(self.items, 1):
            if flat in (re.sub(r"[^a-z0-9]", "", (item.shelf or "").lower()), re.sub(r"[^a-z0-9]", "", item.name.lower())) \
                    or _stem(label) in (_stem(item.shelf or ""), _stem(item.name)):
                return number
        words = {w for w in _stem(label) if len(w) >= 4}
        shared = [len(words & {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        if max(shared) > 0:
            return shared.index(max(shared)) + 1
        return 1 if len(self.items) == 1 else None

    def on_item(self, ident: str, number: int) -> bool:
        """item_of held strictly: once rows of the item's own shelf were seen, a shelf of like words is another shelf."""
        if self.item_of(ident) != number:
            return False
        cat_key = self.category_key()
        flat = lambda text: re.sub(r"[^a-z0-9]", "", str(text or "").lower())      # noqa: E731
        shelf = flat(self.items[number - 1].shelf)
        if not cat_key or not shelf:
            return True
        if len(self.rows) != self.shelf_seen[0]:
            self.shelf_seen = (len(self.rows), {flat(r.get(cat_key)) for r in self.rows.values()})
        return shelf not in self.shelf_seen[1] or flat((self.rows.get(ident) or {}).get(cat_key)) == shelf

    def fit_score(self, ident: str, item: Item) -> float | None:
        """How well a listing's own record meets an item's requirements in force, judged by code: None when a record
        clearly breaks one (a limit the figures fail, or a key for the property that states another value), else the
        number met (a value found only in the title or description counts half)."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        text = _dump({k: v for k, v in record.items() if not isinstance(v, (dict, list))}).lower()
        cap, price = self.caps.get(self.items.index(item) + 1), self.code_price(ident)
        if cap is not None and price is not None and price > cap:
            return None
        score = 0.0
        self.fit_unmet = 0
        for need in item.musts + item.told:
            if need in item.dropped or ":" not in need:
                continue
            prop, value = (x.strip() for x in need.split(":", 1))
            keys = [str(v).lower() for v in _own_keys(prop, block).values()]
            limit = _limit(need)
            if limit and value.lower() in self.labels_of(prop):
                # the required value is one of the store's own labels for that property ("100 and up"): it is met by
                # that label, not by a figure the label would cover
                limit = None
            if limit:
                self.learn_units(block)
                verdict = _limit_verdict(limit, keys, self.factors)
                if verdict == "fails":
                    return None
                score += verdict == "meets"
                # (a limit on the price is met or not by the figures above: no attribute states it)
                self.fit_unmet += verdict != "meets" and not re.search(r"(?i)price|cost", prop)
                continue
            if any(need == ref for _, ref in self.refs) and need not in self.ref_values:
                continue        # (its value is on another page: the code cannot judge it)
            want = self.ref_values.get(need, value).lower()
            if re.match(r"(anything other than|not |no |unlike|other than|except)", want):
                continue
            if _conditional(want) and not re.search(r"except (when|if)|unless", want):
                continue        # (it applies only with a certain other line: the checks read it against the cart)
            want = re.split(r",?\s+(?:except (?:when|if)|unless)\b", want)[0]
            options = [o.strip() for o in re.split(r"\s+or\s+|\s*,\s*", want) if o.strip()]
            named = set(re.findall(r"[a-z0-9]+", prop.lower()))
            options += [" ".join(rest) for rest in ([w for w in re.findall(r"[a-z0-9.]+", o) if w not in named] for o in options) if rest]
            words = [w for w in re.findall(r"[a-z0-9]+", want) if len(w) >= 2]
            if keys:
                if any(_phrase_in(o, k) for o in options for k in keys):
                    score += 1
                elif any({w for w in words if len(w) >= 4} & set(re.findall(r"[a-z0-9]+", k)) for k in keys):
                    score += 0.5     # a word in common: it may say the same thing, which the checks judge
                    self.fit_unmet += 1
                else:
                    return None
            elif words and any(_phrase_in(o, text) for o in options):
                score += 0.5
            else:
                self.fit_unmet += 1
        return score

    def labels_of(self, prop: str) -> set[str]:
        """The values the listings read in full give for a property (lowercase): tells a store's own label ("100 and up")
        apart from a limit to compute."""
        out: set[str] = set()
        for record in self.full.values():
            for key, value in (_fact_block(record) or {}).items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool) and _names(_stem(prop), _stem(key)):
                    out.add(str(value).strip().lower())
        return out

    def code_lines(self) -> dict[str, int]:
        """One listing per item, judged by code from the pages read: the buyable listing meeting the most requirements in
        force with none clearly broken (the cheapest among equals), at the item's quantity; an item with none is left out."""
        lines: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            ranked = []
            for ident in self.full:
                if self.sold_out(ident) or not self.code_buyable(ident) or self.item_of(ident) != number:
                    continue
                score = self.fit_score(ident, item)
                if score is not None:
                    price = self.code_price(ident)
                    ranked.append((-score, price if price is not None else float("inf"), ident))
            if ranked:
                lines[sorted(ranked)[0][2]] = item.qty
        return lines

    def rerank_hint(self) -> str:
        """Under a cheapest pick, once after each change of the budget or the requirements: the pages already read that
        still fit by code, cheapest first, so the cart is weighed against the whole fitting set again."""
        if self.brief.get("pick") != "cheapest" or not self.notes or getattr(self, "reranked_at", -1) == self.version:
            return ""
        self.reranked_at = self.version
        rows = []
        for number, item in enumerate(self.items, 1):
            fits = []
            for ident in self.full:
                if self.item_of(ident) == number and self.buyable(ident) and not self.sold_out(ident) and self.price_of(ident) is not None:
                    score = self.fit_score(ident, item)
                    if score is not None:
                        fits.append((self.price_of(ident), ident))
            if len(fits) >= 2:
                rows.append(f"item {number}: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(fits)[:5]))
        if not rows:
            return ""
        return ("[Kyo] After the change, pages already read that no attribute rules out, cheapest first (check each against every"
                " requirement in force before taking it): " + " | ".join(rows))

    def fallback_cart(self) -> str:
        """The model loop ended (no credit, failed calls, the clock, the turn cap) with no line carted for some item: Kyo
        picks, from the listings whose attributes it read, the buyable one that meets the most requirements by code for
        each such item (the cheapest among equals), carts it inside the budget and checks out. No model call."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.fell_back or self.finished or not cart or not buy or self.left() < 2:
            return ""
        self.fell_back = True
        have = {self.item_of(i) for i in self.cart_ids() if not self.sold_out(i)}
        plan: dict[str, int] = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if not self.sold_out(i)}
        picks = {i: n for i, n in self.code_lines().items() if self.item_of(i) not in have}
        if len(have - {None}) + len(picks) < len(self.items):
            return ""
        plan.update(picks)
        if self.budget is not None:
            # under a budget, no blind buy: every line's price must be known and the total inside it
            prices = [self.code_price(i) for i in plan]
            if any(p is None for p in prices) or sum(p * plan[i] for p, i in zip(prices, plan)) > self.budget:
                return ""
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if not calls or len(calls) + 1 > self.left():
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
            self.absorb(reply.get("observation"))
        self.trace.append({"kind": "fallback_cart", "lines": plan, "drop": drop})
        return "carted " + _dump(plan)

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # the store's own newest cart reply listing no lines at all: an order would only be refused
        obs = self.cart_obs
        emptied = isinstance(obs, dict) and any(isinstance(v, list) for v in obs.values()) and not any(v for v in obs.values() if isinstance(v, list))
        if buy and not emptied and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += self.read_shops()
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            # a requirement whose value is on another listing, still unread when the loop stopped: one reading of the pages
            # already opened, and a carted line that value rules out is swapped for a read listing that meets it
            if not self.broke and not self.no_credit and self.left() > 3 and self.resolve_refs():
                self.recheck_lines()
            # an item with no cart line when the loop stopped: a code pick rather than an ending that throws the episode away
            if not self.broke and len({self.item_of(i) for i in self.cart_ids() if not self.sold_out(i)} - {None}) < len(self.items):
                self.fallback_cart()
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.clock() < CLOCK_STOP:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.clock() < CLOCK_STOP):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + self.read_shops() + self.scan() + "\n" + self.status()})
                continue
            settled = self.settle()
            if settled:
                self.chat.append({"role": "user", "content": settled + "\n" + self.status()})
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
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + self.read_shops() + self.scan() + "\n" + self.status()})
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
