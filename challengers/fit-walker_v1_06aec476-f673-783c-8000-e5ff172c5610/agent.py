"""Wren: buys for one shopper inside an ORO store world.

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
SPEND_SOFT = 0.050
SPEND_HARD = 0.064
CHECK_HEAVY_BELOW = 0.025  # spend under which the final cart check uses the heavy model first
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Wren reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Wren reads before a cheapest checkout
WALK_PAGES = 2              # pages of a category listed under a cart line's price that Wren reads before a cheapest checkout
WALK_LOW_PAGES = 3          # pages it reads under a lowered price ceiling
WALK_CALLS = 5              # store calls one cart line's price walk may make
DECIDE_RECORDS = 40         # listing records the last-resort buyer is shown (spread over the requested kinds)
WALK_KEEP = 12              # steps beyond the closing ones that a price walk always leaves
CHASE_LINKS = 7             # listings followed along one requirement's chain of pages
CHASE_LOOKS = 3             # store searches per link before the chase gives that link up
CHASE_KEEP = 9              # steps a chase always leaves to the rest of the episode
CHASE_TOKENS = 2400         # reply room of a chase call: the light model's thinking counts against it
CHASE_SPEND = 0.015         # spend all chases of an episode may take (light model)
CHASE_CALLS = 20            # model calls all chases of an episode may make (each takes seconds of the episode's time)
WRAP_SECONDS = 220.0        # the agent's own turns stop here (an episode has 300 s), leaving time to check out
SECOND_ROUND_BY = 200.0     # a second round after a refused checkout starts only before this
HARD_STOP = 268.0           # no model call starts after this
CHASE_MARGIN = 45.0         # chases stop this long before the agent's turns do
TIME_WARN = 40.0            # the agent is told to check out this long before its turns stop
FIT_BATCH = 8               # listings one fit reading takes (light model)
FIT_CALLS = 6               # fit readings an episode may make
FIT_ROWS_EACH = 10          # newest unopened search rows of each item a fit reading takes
FIT_SPEND = 0.006           # spend all fit readings of an episode may take (booked apart, like the chases)
FIT_TEXT = 600              # characters of one listing's own words a fit reading sees
PART_STEPS = 6              # store calls the checkout's walk to continuation pages may make
CODE_SHAPE = r"\b[A-Za-z]{1,6}\d{0,3}[-\s]?\d{2,6}[A-Za-z0-9]*\b|\b[A-Za-z]{1,6}[-\s]?\d{1,4}x{1,3}\b"  # a model, series or part code
FIT_WORDS = r"\b(?:fits?|compatible|goes with|works with|for use with|made for|model)\b"  # words that say what a listing fits or name a model
ASK_BACKUP = 6              # an item's most carried keys that stand behind the model's picks
ASK_PRIOR = 18.0            # weight (of 100) of the most carried key when the model ranked nothing for an item
ASK_PRIOR_RANKED = 0.25     # share of that weight it adds to a candidate the model ranked
ASK_UNSEEN = 0.15           # chance kept for properties no candidate names
ASK_ODDS_ONE = 0.75         # chance a lone requested item carries an unwritten preference, when the model gives none
ASK_ODDS_MANY = 0.6         # the same for each of several requested items
ASK_AGAIN_ONE = 0.55        # an item's chance of a further preference after one was found: one item requested
ASK_AGAIN_MANY = 0.35       # the same with several items requested
ASK_FLOOR = 0.01            # below this chance a message is not worth sending
ASK_SPREAD = 0.6            # with several items, the weight of each further question on an item already asked
WIDE_NAMES = 6              # properties one question about one product names while the shopper takes several at once

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Wren, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Wren knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Wren's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Wren carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Wren's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, British or American spelling, a short code of one or two letters in front), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("dark sienna" meets sienna, "extra narrow cut" meets narrow). A value with different words does not count (an abbreviation or code is not the word it stands for, and "sienna brown" is not "rust brown"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- The property's own keys decide: a key named like the requirement's property (under that name or a plain synonym, in any spelling, a shortened form, or with the product's name in front). A listing often carries several own keys for one property; the requirement is met when at least one of them states the value, even if another says something else, and it fails only when every own key states something else, whatever other keys or the title say. A key for a different property, even a related one (a rim colour is not the bowl's colour), decides nothing about the requirement. Wren adds [evidence] lines after opened listings with the keys whose names match a requirement, marking each key that states the wanted value.
- When the listing has no own key for that property, a key with words of its own added that names the property decides in the same way: met when one of them states the value, failed when they all say something else, whatever the title says. Only when no key names the property at all does the wanted value stated under another attribute (for example a list of features that includes it) or in the title or description count. A title or value naming several values of one property (two audiences, several colours) states each of them, so it meets a requirement for any one of them. A number of another kind does not count (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- Own keys include the product's name in front ("kite length" on a kite, "kettle colour" on a kettle). A key with words of its own added ("item material", "fit of the coat") is not an own key: it decides only when the listing has no own key for the property.
- A requirement still holds when the shopper says they will check it themselves; the ledger's list is the one in force.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X that another shop's listing gives"). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy. A line on a page saying that some other part or listing also fits something is about that other part, not about the listing whose page shows it; a line sending you to another part's page for the rest of a list means this listing's list goes on there.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Wren reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better. Between fitting listings you already have, take one whose title also names the required values.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A condition the request marks as one the shopper means to verify personally ("(I'll confirm that one on my end)") is a must, even a use or occasion said that way ("they're for a harvest fair (I'll make sure of it)" gives "occasion: harvest fair"). A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value comes from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


ASK_RANK = """You pick the product properties a shopping agent will ask a shopper about, one property per message. Apart from what the request states, the shopper may hold a few firm preferences they never wrote down. Each is about one property of one requested product, and a preference nobody asks about cannot be met. The shopper reveals a preference only to a question that names that very property; a question about any other property gets nothing useful and uses up one of very few messages. Rank the candidates by how likely each one is to be such a preference.

Where they hide:
1. Properties the request points at without settling. For a product people wear or use on their body, its audience key is a top candidate whenever the request leaves it open, and the first one when the story says someone else will use or wear it: gender first (even when the story hints at it), then an age group or the like; when it dwells on how the product looks or is seen, the look key that most of its listings carry (colour, pattern, finish or shape, whichever the request leaves open); when it is about what the product is made of or holds, the material or contents key; where it must fit, its size or dimension key; what it must work with, its connection or compatibility key. A request that offers an open choice for a property (one thing or another, any size) points at that property too.
2. Otherwise the product's main choice dimensions: the properties a buyer of that kind of product settles first (what kind or style it is, its cut or form, how big it is or how much it holds, what it is made of, how it looks) among the keys most of its listings carry. Keys seen on only one or two listings rarely matter.
What the request itself asks you to check with the shopper (ASK FIRST, when listed) comes before everything else.

Skip a property the request already settles for that product (a requirement names it); a key whose values merely repeat a required value is not settled. Skip identifiers, model numbers, purchase options or variant lists, condition, warranty, shipping and price. Name each candidate by that product's own key from ATTRIBUTE KEYS SEEN in plain words, in the spelling most of its listings carry; a property no listed key names may be given by its plain name. With several products, rank each on its own; several unwritten preferences in one order usually belong to different products.

Weight every candidate from 0 to 100 (your chance that it is the preference, if the product has one) and give each product the chance that it carries any unwritten preference at all.

Return ONE JSON object and nothing else:
{"items": [{"item": product number, "chance": 0-100, "ranked": [{"key": "...", "weight": 0-100}]}]}
List at least COUNT and at most 8 candidates per product, best first."""


LINK_PICK = """You are tracing a chain of store listings for a shopper's requirement. The requirement on one product takes its value from another listing, and a listing's page can hand that property on to a further listing. The link to find now is the description under LINK: it is ONE listing, and every detail the description gives (the shop or seller, the kind of product, each attribute value) must hold for it. A page can mention similar versions under other listings: such a listing is the link only when every detail matches. When LINK is the requirement itself, the link is the listing the requirement describes, not the one that listing's page may point on to. A listing on the chain carries a note of its own from the seller about it; among close candidates, prefer the one whose page has such a note.
You get the requirement the chain is for, the link's description, and records of listings seen so far: search rows, and their attributes and page when they were read.
When the description singles out one of several matching listings by a measure (the one ranking highest or lowest in some attribute), compare that attribute across all the matching records before you pick.
Return ONE JSON object and nothing else: {"pick": the id of the record that is the described listing, or "" when no record is, "read": up to 3 ids of records that might be it but whose deciding attributes are not shown yet, "page": up to 3 ids of such records whose page should be opened instead (a page often describes its own listing in words), "search": when no record is or might be it, the words to search the store with to find it (the seller's name with the kind of product or one distinctive detail, at most 6 words), else "", "seller": the shop or seller name the description gives, else ""}"""

LINK_READ = """You are tracing where a shopper's requirement gets its value. The requirement names a property of a product and takes the property's value from another listing; that listing's page may state the value, or send the reader on to a further listing for it, which it then describes. The listings along the way can be of any kind of product, unlike the product the requirement is for: what counts is the property the requirement names.
You get the requirement, the chain followed so far (each link reached by the description the previous page gave) and the page of the listing just opened, the last link (with its attributes when they were read). A remark that similar versions exist under other listings sends the reader nowhere. A listing on the chain carries a note of its own from the seller; when this page has no such note and its attributes do not settle the property, it is a look-alike: give neither "value" nor "next". When the requirement wants a value that must match or fit the listing's value and the page names that matching value, the value is the matching one (a page naming walnut as the match for its oak finish gives walnut); when the page lists such matches for several values instead, take the listing's own value for the property, from the page or its attributes, and give its match.
Return ONE JSON object and nothing else: {"next": when this page sends the requirement's property on to another listing, that listing's description copied from the page (seller, kind of product and every detail given), else "", "search": with "next", the words to search the store with to find it (the seller's name with the kind of product or one distinctive detail, at most 6 words), else "", "seller": with "next", the shop or seller name its description gives, else "", "value": when the page, or its attributes, states the property's value and sends it nowhere else, that value as stated with its unit, else "", "rule": with "value", the requirement restated with that value in place of the pointer, keeping the requirement's own comparison word for word (equal to, near, at most, unlike, and so on; a rule that admits only one value keeps admitting only that one); for a matching value, "value" is already the value the product must have, so the rule names that value plainly, else ""}"""


BEST = """A shopping agent is about to end without buying anything. Such an ending earns nothing unless the store truly has no listing that can serve the request, so buying the closest listing is worth more than leaving empty-handed. For each requested item choose, among the listing records shown, the one that comes closest to the requirements in force: one that meets them all when there is one, else the one meeting the most of them (the kind of product and the shopper's own stated and answered needs first), in stock and inside the budget, following the pick rule. Leave an item out only when no record is of its kind at all, or every record of its kind is out of stock or breaks the budget. A line already in the cart stays the item's choice unless another record clearly fits better or the latest check rejected it.
Return ONE JSON object and nothing else: {"lines": [{"item": item number, "id": the chosen listing's id}], "missing": [numbers of the items left out]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the answer takes the value from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "single": [question numbers], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value; so does an answer that refuses every value except one ("nothing but V", "anything else than V is a no"): the need is V.
- "single": the numbers of the questions whose answer asks for the questions to come one at a time or separately (too many at once), without stating any value. An answer that did not follow the message, or wants it simpler, shorter or said again, is not "single"."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "dark sienna" for sienna, units converted, limits include the edge, a range must lie wholly inside); the property's own keys decide (a key naming the same property in any spelling, a shortened form, or with the product's name in front, as "kite length" on a kite): when a listing has several own keys for the property, it meets the requirement if at least one of them states the value, even if another says something else, and it fails only when every own key says something else; a key for a different property, even a related one (a rim colour is not the bowl's colour), decides nothing; when the listing has no own key for that property, a key with words of its own added that names it ("item material") decides in the same way, and only when no key names the property does the wanted value stated under another attribute or in the title or description count; a title or value naming several values of one property (two audiences, several colours) states each of them (a number of another kind does not count: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. The REQUIREMENTS IN FORCE are the full list: a requirement of the request missing there was dropped, and one the shopper says they will check personally must still be met. When a listing's text says which other models, series or parts it is compatible with (in its description, page or title), every other cart line it relates to must be among them, read off that line's own records (a model or series code often sits in the title); a line saying that some other part also fits something is about that other part, not about the listing whose page shows it, and a line sending the reader to another part's page for the rest means the list goes on there. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the property's own keys decide: a key naming that same property in any spelling, a shortened form, or with the product's name in front, as "kite length" on a kite (one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "dark sienna" for sienna, limits include the edge); when a listing has several own keys for the property, it meets the requirement if at least one of them states the value, even if another says something else, and fails only when every own key says something else; a key for a different property, even a related one (a rim colour is not the bowl's colour), decides nothing; when no own key exists, a key with words of its own added that names the property ("item material") decides in the same way, and only when no key names the property does the wanted value stated under another attribute or in the title or description count; a title or value naming several values of one property (two audiences, several colours) states each of them (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; the requirements in force are the full list (one the request states but they leave out was dropped), and one the shopper says they will check personally must still be met; out-of-stock listings and listings that break the budget do not fit."""

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

FIT_READ = """A shopping agent must buy several products for one order, and some of them may have to work together (a device and a part or accessory for it). Below are listings the agent has seen, with their own words. For each listing, report only what its own words say about ITSELF:
"own": the codes that name this listing itself: its model, series or part code (often in the title, or after words such as model, ref, series, part or item no.), never a code its words name as another part ("more on part P" names P, another part);
"fits": the codes of other models this listing says it goes with, works with, fits, suits or is made for, copied as written (a range such as "TX-39xx" or "every TX-39 model" stays one entry, written as the listing writes it);
"not": codes the listing says it does NOT fit, such as the exceptions to a range;
"more": part codes whose page this listing says carries the rest of its list ("more on part P", "see part P for the others");
"others": sentences about ANOTHER part's fit, as {"part": that part's code, "fits": the codes it is said to fit}. A sentence such as "part P also fits M" speaks about part P, never about the listing that shows it, so it goes here and never under "fits".
Never take codes that only describe a different listing. Leave a list empty when the words do not say it.
Return ONE JSON object and nothing else: {"listings": [{"id": "...", "own": [...], "fits": [...], "not": [...], "more": [...], "others": [{"part": "...", "fits": [...]}]}]}"""


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
    """The same word, or two long words one dropped letter apart (one spelling of a word and another)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) != 1:
        return False
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def _covers(part: frozenset[str], whole: frozenset[str]) -> bool:
    return all(any(_near(w, x) for x in whole) for w in part)


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return _covers(prop, key) or (len(key) >= 2 and _covers(key, prop))


def _value_words(need: str) -> tuple[list[str], set[str]] | None:
    """A requirement's value as (its words, the words a negation applies to); None for a number, a limit, a list of
    alternatives or a value read from another listing, which no single word can stand for."""
    value = need.split(":", 1)[1].strip().lower() if ":" in need else ""
    if not value or re.search(r"\d", value) or re.search(
            r"\b(?:least|most|under|over|below|above|between|than|or|nor|either|neither|unlike|same|matching|match|as)\b", value):
        return None
    words = re.findall(r"[a-z]+", value)
    # a negating word on its own (not the front of a joined word such as "non-woven") negates the next two words
    stops = [len(re.findall(r"[a-z]+", value[:m.start()])) for m in re.finditer(r"\b(?:not|no|never|without|except|but|non)\b(?!-)", value)]
    negated = {words[i] for at in stops for i in range(at + 1, min(len(words), at + 3))}
    return words, negated


def _stated_part(words: frozenset[str], text: Any) -> str | None:
    """The element of a key's value (a comma, slash or "and" list) that holds every word of a plain wanted value, singular
    or plural, without negating one of them ("no X", "non-X", "X-free"); None when no element does."""
    if not words:
        return None
    for part in re.split(r"\s*(?:[,/;|&+]|\band\b|\bor\b)\s*", str(text).lower()):
        said = _words(" ".join(re.findall(r"[a-z]+", part)))
        if not said or not words <= _stem(said):
            continue
        negated = False
        # a negating word that is itself part of the wanted value ("non-woven") negates nothing
        nots = "|".join(w for w in ("not", "no", "without", "never", "non") if w not in words)
        for word in words:
            if word in ("no", "non", "not", "free"):
                continue
            stem = re.escape(word)
            if ((nots and re.search(r"\b(?:" + nots + r")\s+(?:[a-z]+\s+)?" + stem, said))
                    or ("free" not in words and re.search(r"\b" + stem + r"s?\s+free\b", said))):
                negated = True
                break
        if not negated:
            return part.strip()
    return None


def _states(words: frozenset[str], text: Any) -> bool:
    return _stated_part(words, text) is not None


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


class Wren:
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
        # the episode's own clock: model calls take seconds, and an episode that runs out of time ends with nothing
        self.started = time.monotonic()
        self.wrap_at = WRAP_SECONDS
        self.fit_shown = ""
        self.ask_failed = False
        self.one_at_a_time = False   # the shopper asked for one question per message: no more wide questions
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
        self.chased: set[str] = set()
        self.decided = 0
        self.decided_at = -9
        self.decided_state: tuple[Any, Any] = (None, None)
        self.page_seen: dict[str, tuple[str, Any]] = {}
        self.search_form: dict[str, Any] = {}
        self.chase_tries: dict[str, int] = {}
        self.check_fail: dict[str, Any] = {}
        self.link_pages: dict[str, Any] = {}
        self.chase_calls = 0
        self.chase_cost = 0.0
        self.fit_info: dict[str, dict[str, Any]] = {}   # what a fit reading found in each listing's own words
        self.fit_len: dict[str, int] = {}               # how many characters of its words that reading saw
        self.part_tries: dict[str, int] = {}           # store calls the checkout walk spent on each part code
        self.part_pages: list[str] = []                     # continuation pages the walk opened
        self.fit_calls = 0
        self.fit_cost = 0.0
        self.chase_seen: set[str] = set()
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
        # the search that lists one category under a maximum price: None until looked up, False when the store has none
        self.walk_tool: Any = None
        self.walked: set[str] = set()
        self.walk_seen: dict[str, Any] = {}
        self.walk_only: set[str] = set()
        self.records_cache: dict[Any, list[dict[str, Any]]] = {}
        self.wider_held = False
        self.paged_obs: list[Any] = []

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def fit_text(self, ident: str) -> str:
        """A listing's own words for a fit reading: its title and every text field but the id and category."""
        record = self.rows.get(ident) or {}
        cat_key = self.category_key()
        fields = [(k, v.strip()) for k, v in record.items() if k not in (self.id_key, cat_key) and isinstance(v, str) and v.strip()]
        parts = []
        # short fields first (a title, a seller's note); a long text keeps only its sentences that carry a code or say
        # what something fits
        for key, text in sorted(fields, key=lambda kv: len(kv[1])):
            if len(text) > 160:
                text = " ".join(x for x in re.split(r"(?<=[.;!?])\s+|\n", text)
                                if re.search(CODE_SHAPE, x) or re.search(FIT_WORDS, x, re.I))[:300]
                if not text:
                    continue
            parts.append(f"{key}: {text}")
        block = _fact_block(record) or {}
        for key, value in block.items():
            if isinstance(value, str) and re.search(CODE_SHAPE, value):
                parts.append(f"{key}: {value}")
        return " | ".join(parts)[:FIT_TEXT]

    def part_targets(self) -> set[str]:
        """Code keys of the parts that read listings send the reader to for the rest of their fit lists."""
        return {self.code_key(p) for info in self.fit_info.values() for p in info.get("more") or [] if self.code_key(p)}

    def fit_read(self, first: list[str] | None = None) -> None:
        """With several items: read what listings say they are and what they fit (light model, in batches), once each:
        listings of a requested kind whose words speak of fit or name a model, any listing a read list continues at
        (whatever its kind), and the listings in `first` (a cart's lines, pages just walked) ahead of the rest. The last
        reading of the episode waits for a call that names its listings."""
        if len(self.items) < 2 or self.finished or self.no_credit or self.broke:
            return
        # the shopper's answers come first: they say which listings are still candidates, and the reading takes time
        if not self.asked and self.roles.get("ask") and self.msgs_free() > 0:
            return
        cat_key = self.category_key()
        lead = set(first or [])
        targets = self.part_targets()
        fresh = []
        rows_left: dict[int, int] = {}
        for ident in reversed(self.seen_order):
            # a listing read from its search row is read again once its page added words of its own
            if ident in self.fit_info and len(self.fit_text(ident)) <= self.fit_len.get(ident, 0) + 40:
                continue
            record = self.rows.get(ident) or {}
            label = str(record.get(cat_key) or "") if cat_key else ""
            number = self.owner_of(label) if label else None
            text = self.fit_text(ident)
            is_target = bool(targets) and any(t and t in self.code_key(text[:200]) for t in targets)
            if not number and not is_target and ident not in lead:
                continue
            if not re.search(CODE_SHAPE, text):
                continue
            if ident not in lead and not is_target and not re.search(FIT_WORDS, text, re.I):
                continue
            # every opened listing, and the newest search rows of each item
            if ident not in self.full and ident not in lead and not is_target and number:
                if rows_left.get(number, FIT_ROWS_EACH) <= 0:
                    continue
                rows_left[number] = rows_left.get(number, FIT_ROWS_EACH) - 1
            fresh.append((ident, label, text))
        fresh.sort(key=lambda x: (0 if x[0] in lead else 1, 0 if x[0] in self.full else 1))
        budget = FIT_CALLS if lead else FIT_CALLS - 1
        halved = False
        while fresh and self.fit_calls < budget and self.fit_cost < FIT_SPEND and self.elapsed() < self.wrap_at - (10 if lead else CHASE_MARGIN) \
                and self.calls + 6 <= ALL_CALLS - (10 + 2 * len(self.items)):
            size = FIT_BATCH // 2 if halved else FIT_BATCH
            batch, fresh = fresh[:size], fresh[size:]
            prompt = FIT_READ + "\n\nLISTINGS:\n" + "\n".join(f"- id {i} (category {lab!r}): {t}" for i, lab, t in batch)
            before = self.cost
            self.fit_calls += 1
            got = self.ask_json(prompt, [LIGHT], 3000) or {}
            self.fit_cost += self.cost - before
            if not isinstance(got.get("listings"), list):
                # no readable reply: the first half is tried again on its own once, the rest waits for a later reading
                if len(batch) > 2 and not halved:
                    halved = True
                    fresh = batch + fresh
                    continue
                break
            wanted = {i for i, _, _ in batch}
            for entry in _items(got.get("listings")):
                if not (isinstance(entry, dict) and str(entry.get("id") or "") in wanted):
                    continue
                info = {k: [str(x).strip() for x in _items(entry.get(k)) if str(x).strip()][:12] for k in ("own", "fits", "not", "more")}
                info["others"] = [(str(o.get("part")).strip(), [str(x).strip() for x in _items(o.get("fits")) if str(x).strip()][:12])
                                  for o in _items(entry.get("others")) if isinstance(o, dict) and str(o.get("part") or "").strip()][:6]
                # a code the listing names as another part (its list goes on there, or a sentence is about it) is not its own
                elsewhere = {self.code_key(c) for c in info["more"]} | {self.code_key(p) for p, _ in info["others"]}
                info["own"] = [c for c in info["own"] if self.code_key(c) not in elsewhere]
                ident = str(entry["id"])
                self.fit_info[ident] = info
                self.fit_len[ident] = len(next(t for i, _, t in batch if i == ident))
            # a listing the reply left out stays unread
        self.trace.append({"kind": "fit", "read": len(self.fit_info), "calls": self.fit_calls})

    @staticmethod
    def code_key(code: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(code).lower())

    def code_fits(self, pattern: str, code: str) -> bool:
        """A listing's code against what another listing says it fits: the same code, a series named by its front
        ("TA3" for "TA3-48213"), a range written with x's ("TX-39xx"), or a family a phrase names ("every TX-39 model")."""
        raw = str(pattern)
        # the code inside a phrase such as "every TX-39 model" or "VIT-39xx models"
        token = re.search(r"[A-Za-z]{1,6}[-\s]?\d{1,6}[xX]*(?![A-Za-z0-9])", raw)
        core = token.group(0) if token else raw
        want, have = self.code_key(core), self.code_key(code)
        if not want or not have:
            return False
        if want == have:
            return True
        front = core.strip().upper()
        if re.fullmatch(r"[A-Z]+[-\s]?\d+", front) and re.match(re.escape(front) + r"[-\s/]", str(code).strip().upper()):
            return True
        if re.search(r"\dx+$", want):
            shape = re.sub(r"x+$", lambda m: r"[0-9a-z]{%d}" % len(m.group(0)), want)
            if re.fullmatch(shape, have):
                return True
        # a family named by its front part ("every TX-39 model"): letters and digits, shorter than the code
        return bool(re.fullmatch(r"[a-z]+\d{2,}", want)) and len(want) < len(have) and have.startswith(want) \
            and bool(re.search(r"\b(?:every|all|any|series|range|family)\b", raw, re.I))

    def fit_links(self) -> list[tuple[int, str, int, str, bool | None]]:
        """(item a, listing a, item b, listing b, verdict) for every pair of listings of different items where a's words
        list what it fits and b carries a code of that list's family. Verdict: True (b is in a's list), False (b is not,
        and a's list is complete), None (b is not in what is known of a's list, which goes on at a part not read yet)."""
        cat_key = self.category_key()
        owner: dict[str, int] = {}
        for ident in self.fit_info:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            number = self.owner_of(label) if label else None
            if number:
                owner[ident] = number
        # what any page says about a part ("part P also fits M") belongs to the listing that part code names
        told: dict[str, list[str]] = {}
        for info in self.fit_info.values():
            for part, codes in info.get("others") or []:
                told.setdefault(self.code_key(part), []).extend(codes)
        read_codes = {self.code_key(c) for info in self.fit_info.values() for c in info.get("own") or []}
        links = []
        for a, info in self.fit_info.items():
            fits = list(info.get("fits") or [])
            for code in info.get("own") or []:
                fits += told.get(self.code_key(code), [])
            if not fits or a not in owner:
                continue
            open_end = any(self.code_key(p) not in read_codes for p in info.get("more") or [])
            families = {re.match(r"[a-z]*", self.code_key(p)).group(0) for p in fits} - {""}
            for b, other in self.fit_info.items():
                if b == a or b not in owner or owner[b] == owner[a] or not other.get("own"):
                    continue
                # a fit list speaks only of codes of its own family (the same leading letters): a part named by a code of
                # another family is no device the list is about
                if not any(re.match(r"[a-z]*", self.code_key(c)).group(0) in families for c in other["own"]):
                    continue
                good = any(self.code_fits(p, c) for p in fits for c in other["own"])
                bad = any(self.code_fits(p, c) for p in info.get("not") or [] for c in other["own"])
                verdict: bool | None = good and not bad
                if not verdict and not bad and open_end:
                    verdict = None
                links.append((owner[a], a, owner[b], b, verdict))
        return links

    def fit_digest(self) -> str:
        """With several items: what the listings seen say they fit, read from their own words, so that lines the request
        ties together can be chosen to fit each other. Empty when nothing of the kind was seen."""
        if len(self.items) < 2:
            return ""
        self.fit_read()
        links = self.fit_links()
        if not links:
            return ""
        usable = {i for i in self.fit_info if not self.sold_out(i) and self.buyable(i)}
        lines = ["[fit board, read from the listings' own words (codes in brackets); it matters for the lines the request says must"
                 " work together, a part or accessory with the device or model it is for]"]
        pairs: dict[tuple[int, str, int], list[str]] = {}
        for a_item, a, b_item, b, verdict in links:
            if a in usable and b in usable and verdict:
                pairs.setdefault((a_item, a, b_item), []).append(b)
        for (a_item, a, b_item), hosts in sorted(pairs.items())[:14]:
            own = ", ".join((self.fit_info.get(a) or {}).get("own") or [])
            lines.append(f"item {a_item} listing {a}{' [' + own + ']' if own else ''} is listed for item {b_item} listing(s): "
                         + ", ".join(h + " [" + ", ".join((self.fit_info.get(h) or {}).get("own") or []) + "]" for h in hosts[:5]))
        complete = {(x, y) for x, y, _, _, v in links if v is False}
        lonely = sorted({(a_item, a) for a_item, a, _, _, v in links if a in usable and v is False} - {(x, y) for x, y, _ in pairs})
        for a_item, a in [p for p in lonely if p in complete][:6]:
            want = ", ".join(((self.fit_info.get(a) or {}).get("fits") or [])[:6])
            lines.append(f"item {a_item} listing {a} is listed for none of the other items' listings seen so far (its list: {want})")
        read_codes = {self.code_key(c) for info in self.fit_info.values() for c in info.get("own") or []}
        for a, info in self.fit_info.items():
            gone = [p for p in info.get("more") or [] if self.code_key(p) not in read_codes]
            if gone and a in usable:
                lines.append(f"listing {a}'s list goes on at part {', '.join(gone[:2])}: until that part's page is read, its list is not"
                             " complete (search that code and open the page)")
        if len(lines) > 1 and self.brief.get("pick") == "cheapest":
            lines.append("for the cheapest order, compare whole baskets: each device with the cheapest parts listed for it, then the lowest total")
        return "\n".join(lines) if len(lines) > 1 else ""

    def fit_lines(self, ids: list[str]) -> str:
        """For the cart check: what the cart's listings say they fit, for the pairs whose codes relate; data for the
        lines the request ties together, not a verdict on the others."""
        if len(self.items) < 2 or len(ids) < 2:
            return ""
        self.fit_read(first=ids)
        wanted = set(ids)
        out = []
        for a_item, a, b_item, b, verdict in self.fit_links():
            if a in wanted and b in wanted:
                info, other = self.fit_info.get(a) or {}, self.fit_info.get(b) or {}
                said = {True: "in that list", False: "not in that list", None: "not in what is known of that list (it goes on at a part not read yet)"}[verdict]
                out.append(f"{a} (item {a_item}) lists: {', '.join(info.get('fits') or []) or 'codes credited from other pages'}"
                           + (f"; not: {', '.join(info.get('not') or [])}" if info.get("not") else "")
                           + f" | {b} (item {b_item}) is {', '.join(other.get('own') or [])}: {said}")
        if out:
            out.insert(0, "(this matters only for lines the request says must fit each other)")
        return "\n".join(out[:12])

    def whole_label(self, number: int) -> str:
        """The one store category label seen whose words are exactly the words of the item's name, when another label
        seen holds only part of them (a request for "kettle or teapot" beside categories "Teapots" and "Kettles & Teapots")."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        want = {w for w in _stem(self.items[number - 1].name) if len(w) >= 3 and w not in ("and", "for", "the", "with")}
        if not want:
            return ""
        labels = {str(r.get(cat_key)).strip() for r in self.rows.values() if isinstance(r.get(cat_key), str) and r.get(cat_key).strip()}
        words = lambda lab: {w for w in _stem(lab) if len(w) >= 3 and w not in ("and", "for", "the", "with")}
        full = [lab for lab in labels if words(lab) == want]
        part = [lab for lab in labels if (want & words(lab)) and words(lab) != want]
        return full[0] if len(full) == 1 and part else ""

    def holding(self) -> bool:
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        return bool(self.cart_ids()) if fresh else bool(self.cart_lines)

    def partial(self) -> bool:
        """Several items requested and the cart holds lines for only some of them (read by the lines' category labels).
        Items of one kind are left out of this reading: the buyer could not tell their lines apart."""
        if len(self.items) < 2:
            return False
        kinds = [_words(item.shelf or item.name) for item in self.items]
        if len(set(kinds)) < len(kinds):
            return False
        ids = self.cart_ids()
        cat_key = self.category_key()
        if not ids or not cat_key or len(ids) >= len(self.items):
            return False
        owners = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in ids} - {None}
        return len(owners) < len(self.items)

    def out_of_room(self) -> bool:
        """No order yet, the cart empty or missing a requested item, and just enough steps left for the buyer to cart
        each item and check out: the agent's own turns stop so that the finish still has the steps for it."""
        return (self.buys == 0 and (not self.holding() or self.partial()) and bool(self.roles.get("cart")) and bool(self.roles.get("buy"))
                and bool(self.roles.get("refuse")) and len(self.items) + 2 <= self.left() <= len(self.items) + 3)

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps due right before an order take stay back."""
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
        # near the episode's time limit no model call starts: the finish then buys a held cart without one
        if self.elapsed() >= HARD_STOP:
            return None
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
        # room for the light model's thinking: a reply cut at its limit sends the brief to the heavy model
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
            # messages stay back only for the steps due right before an order: the ending without a purchase is the
            # rare finish, and a message held back for it is a question the interviewer can never ask
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
            else:
                whole = self.whole_label(number)
                if whole:
                    head += f" (the store category whose label names it as a whole: {whole!r}; buy it only from there)"
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
            when = {"choose": "before choosing", "order": "at the order (Wren does it when you check out)", "end": "at an ending without a purchase (Wren does it then)"}[entry["when"]]
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
        text = f"[Wren] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.buys == 0 and self.elapsed() >= self.wrap_at - TIME_WARN:
            text += " TIME IS NEARLY UP: cart the best listing for each item now and check out; an episode that runs out of time earns nothing."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        digest = self.fit_digest()
        # shown again only when it holds something new
        if digest and digest != self.fit_shown:
            self.fit_shown = digest
            text += "\n" + digest
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
                # one value on every listing that carries the key narrows nothing
                if len(values) >= 4 and len(set(values)) == 1:
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Wren reads a short
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
            return (f"\n[Wren] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
        return ("\n[Wren read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Wren read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Wren] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

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
            if isinstance(seen, dict) and not seen.get("error"):
                self.page_seen[ident] = (tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def ceiling_tool(self) -> tuple[str, str, str, dict[str, Any]] | None:
        """The store's search that lists one category under a maximum price, found by its parameters' names: (tool,
        category parameter, price ceiling parameter, fixed arguments such as a filter mode). None when there is none."""
        if self.walk_tool is not None:
            return self.walk_tool or None
        self.walk_tool = False
        skip = {self.roles.get(r) for r in ("cart", "cart_view", "buy", "refuse", "ask", "open", "offer", "compare")} - {None}
        best: tuple[int, tuple[str, str, str, dict[str, Any]]] | None = None
        for tool in self.tools:
            name = tool["function"]["name"]
            if name in skip:
                continue
            schema = self.schemas.get(name) or {}
            props = schema.get("properties") or {}
            cat = cap = None
            for key, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                said = _words(key)
                about = said + " " + str(spec.get("description") or "").lower()[:200]
                kinds = _kinds(spec)
                described = str(spec.get("description") or "").lower()[:200]
                if (cap is None and ("number" in kinds or "integer" in kinds)
                        and re.search(r"\b(?:cap|ceiling|max|maximum|upper|top|under|below|limit|budget)\b", said)
                        and (re.search(r"\b(?:price|cost|spend|budget|cap|ceiling|money|pay)\b", said)
                             or re.search(r"\b(?:price|cost|spend|budget|money|pay)\b", described))
                        and not re.search(r"\b(?:min|minimum|floor|lower|from|row|rows|page|offset|start|count|size|results?|items?|shipping|delivery)\b", said)
                        and not re.search(r"\b(?:how many|number of|per page|page size|listings|rows|results|returned|shown)\b", described)):
                    cap = key
                elif ("string" in kinds and re.search(r"\b(?:section|category|categories|department|aisle|shelf|group|label|kind)\b", said)
                        and not re.search(r"\b(?:brand|maker|manufacturer|color|audience|age|gender|size|style|material|shop|seller|vendor|condition|tag|mode|search)\b", said)):
                    # a plain category word beats a looser one (group, label, kind)
                    plain = bool(re.search(r"\b(?:section|category|categories|department|aisle|shelf)\b", said))
                    if cat is None or (plain and not re.search(r"\b(?:section|category|categories|department|aisle|shelf)\b", _words(cat))):
                        cat = key
            if not cap or not cat:
                continue
            fixed: dict[str, Any] = {}
            ok = True
            for key, spec in props.items():
                if key in (cap, cat) or not isinstance(spec, dict):
                    continue
                enum = spec.get("enum") if isinstance(spec.get("enum"), list) else None
                mode = next((v for v in enum or [] if isinstance(v, str)
                             and re.search(r"field|criteri|filter|browse|categor|section|shelf|structur|attribut|range|screen|sift", v.lower())), None)
                if mode is not None and (key in (schema.get("required") or []) or re.search(r"\b(?:mode|using|by|kind|type|search)\b", _words(key))):
                    fixed[key] = mode
                elif key in (schema.get("required") or []):
                    ok = False
                    break
            if ok:
                score = len(schema.get("required") or [])
                if best is None or score < best[0]:
                    best = (score, (name, cat, cap, fixed))
        if best:
            self.walk_tool = best[1]
            self.trace.append({"kind": "walk_tool", "tool": best[1]})
        return self.walk_tool or None

    def walk_room(self) -> bool:
        return not self.finished and self.left() - self.tail() - len(self.items) >= WALK_KEEP

    def walk_list(self, label: str, ceiling: float, pages: int) -> tuple[int, int, bool] | None:
        """One listing of a category under a price ceiling with the store's ceiling search, `pages` pages at most (a
        step each; a page listed before is read again from its record). Returns (listings matched, rows on a page,
        whether the rows come cheapest first); None when the search failed or ignored the ceiling, which also turns the
        walk off for the episode."""
        name, cat_arg, cap_arg, fixed = self.walk_tool
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(cap_arg) or {}
        value: Any = int(ceiling) if "integer" in _kinds(spec) or float(ceiling).is_integer() else round(ceiling, 2)
        faults: list[str] = []
        call = _shape({**fixed, cat_arg: label, cap_arg: value}, self.schemas.get(name), "args", faults)
        if faults:
            return None
        count, shown, ordered = 0, 0, False
        for page in range(pages):
            mark = name + _dump(call)
            seen = self.walk_seen.get(mark)
            if seen is None:
                if not self.walk_room():
                    break
                known = set(self.rows)
                reply = self.act(name, call)
                if reply is None or self.finished:
                    return None
                seen = reply.get("observation")
                self.observe(name, seen)
                self.read_cache[mark] = (self.version, self.step, self.show(seen))
                self.walk_seen[mark] = seen
                # rows only the walk brought in stand behind the agent's own in the record lists, as a chase's do
                fresh_ids = [i for i in self.seen_order if i not in known]
                self.chase_seen.update(fresh_ids)
                self.walk_only.update(fresh_ids)
                rows = self.records_in(seen)
                prices = [q for q in (self.price_of(i) for i, _ in rows) if q is not None]
                if (isinstance(seen, dict) and seen.get("error")) or (rows and sum(1 for q in prices if q > value + 1e-9) * 2 > len(rows)):
                    # an error, or a search that does not apply the ceiling in this mode: no more walks this episode
                    self.walk_tool = False
                    self.trace.append({"kind": "walk", "label": label, "off": _clip(seen, 40)})
                    return None
            rows = self.records_in(seen)
            plan = self.paging(name, seen)
            if page == 0:
                shown = len(rows)
                count = plan[0] if plan else len(rows)
                ordered = self.price_sorted(seen)
            if not plan or plan[3] is None or plan[0] <= plan[1]:
                break
            faults = []
            call = _shape({**call, plan[2]: plan[3]}, self.schemas.get(name), "args", faults)
            if faults:
                break
        self.trace.append({"kind": "walk", "label": label, "ceiling": value, "count": count, "pages": pages})
        return count, shown, ordered

    def price_walk(self) -> None:
        """Before a cheapest checkout (and right after a cheapest cart add): Wren lists each cart line's category up to
        that line's price with the store's price-ceiling search, a few steps and no model turn, so cheaper listings the
        agent's own searches never showed come into view; cheaper_rows then reads the likeliest of them. When far more
        listings sit under that price than a few pages show and rows come in no price order, it probes lower ceilings,
        guided by the match counts the store reports, until the cheapest part of the category fits in a few pages."""
        if self.brief.get("pick") != "cheapest" or not self.walk_room() or not self.ceiling_tool():
            return
        cat_key = self.category_key()
        if not cat_key:
            return
        name, _, cap_arg, _ = self.walk_tool
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(cap_arg) or {}
        whole = "integer" in _kinds(spec)
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip()
            # once per category and price; a listing selling out since then (often the cheaper one the walk found)
            # makes the cheapest part of the category worth listing again
            mark = f"{label.lower()}|{price}|{len(set(self.gone) | self.unavailable)}"
            if price is None or price <= 0 or not label or mark in self.walked or not self.walk_tool:
                continue
            if not self.walk_room():
                return
            self.walked.add(mark)
            begin = self.step
            # the first listing goes up to the line's own price, so the line itself must be in it: an empty one means
            # the search reads the category some other way, and the walk stops
            got = self.walk_list(label, float(int(price)) if whole else price, 1)
            if not got:
                continue
            count, shown, ordered = got
            if count == 0:
                self.walk_tool = False
                self.trace.append({"kind": "walk", "label": label, "off": "the line's own category listed nothing"})
                return
            if count <= shown:
                continue
            if ordered or count <= WALK_PAGES * shown:
                # cheapest first, or a short rest: read on
                self.walk_list(label, float(int(price)) if whole else price, WALK_PAGES)
                continue
            # far more listings than a few pages show, in no price order: search the ceiling under which about two
            # pages of the category's cheapest listings stay, interpolating with the category's prices seen so far
            want = WALK_PAGES * shown
            known = sorted(q for q in (self.price_of(i) for i in self.seen_order
                                       if str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == label.lower())
                           if q is not None and q < price)
            low, high = (0.0, 0), (float(price), count)
            while self.step - begin < WALK_CALLS and self.walk_room() and self.walk_tool:
                (c_low, n_low), (c_high, n_high) = low, high
                share = max(0.0, min(1.0, (want - n_low) / max(1, n_high - n_low)))
                inside = [q for q in known if c_low < q < c_high]
                guess = inside[min(len(inside) - 1, int(share * len(inside)))] if len(inside) >= 3 else c_low + (c_high - c_low) * max(0.15, min(0.85, share))
                guess = float(int(guess)) if whole or float(price).is_integer() and float(guess) >= 1 else round(guess, 2)
                if guess <= c_low or guess >= c_high:
                    break
                probe = self.walk_list(label, guess, 1)
                if not probe:
                    break
                many = probe[0]
                if many > WALK_LOW_PAGES * shown:
                    high = (guess, many)
                    continue
                if many >= want // 2 or self.step - begin >= WALK_CALLS - 1:
                    if many > probe[1]:
                        # read on through this range, inside the walk's calls for this line
                        self.walk_list(label, guess, max(1, min(WALK_LOW_PAGES, WALK_CALLS - (self.step - begin) + 1)))
                    break
                low = (guess, many)

    def walk_now(self) -> str:
        """Right after a line is carted under a cheapest pick, while steps are still plenty: the price walk and a read of
        the likeliest cheaper rows, shown with the cart's reply so the agent can swap before it checks out."""
        if self.brief.get("pick") != "cheapest" or not self.walk_room():
            return ""
        steps, before = self.step, len(self.seen_order)
        self.price_walk()
        if self.step == steps or self.finished:
            return ""
        read = self.cheaper_rows() if self.walk_room() else []
        text = (f"\n[Wren listed each cart line's category below that line's price ({self.step - steps} steps, no turn of yours): "
                f"{len(self.seen_order) - before} listings not seen before]")
        if read:
            text += "\n[Wren read the attributes of the likeliest cheaper ones; swap one in if it meets every requirement]\n" + "\n".join(read)
        return text

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few: rows naming the item's wanted values first, then the cheapest) so the last check
        can weigh them against the cart's choice."""
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
            number = self.owner_of(kind)
            # the item's requirements in force (dropped ones left out), as wanted words: the ones its listings' values
            # use when there are such listings, else the value's own longer words
            wanted = []
            for n, need in self.needs_now():
                if n != number:
                    continue
                words = self.wanted_of(number, need)
                parsed = _value_words(need)
                if words is None and parsed:
                    words = frozenset(w for w in _stem(" ".join(parsed[0])) if len(w) > 2 and w not in parsed[1])
                if words:
                    wanted.append(words)
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]

            def named(ident: str) -> int:
                text = " ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str))[:1500]
                said = _stem(" ".join(re.findall(r"[a-z]+", text.lower())))
                return sum(len(w & said) for w in wanted)

            pool.sort(key=lambda i: (-named(i), self.price_of(i) or 0.0))
            picks += [i for i in pool[:2] if i not in picks]
        read = self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)
        # listings read for the check are candidates now, whoever brought them in
        self.chase_seen.difference_update(i for i in picks if i in self.full)
        self.walk_only.difference_update(i for i in picks if i in self.full)
        return read

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Wren's own call and read any note in it; returns what the note changed."""
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

    def owner_of(self, label: str) -> int | None:
        """The requested item whose kind a category label names; with one item, that item."""
        name = _words(label)
        long_label = {w for w in _stem(label) if len(w) >= 4}
        best: tuple[int, int] | None = None
        for number, item in enumerate(self.items, 1):
            if name and name in (_words(item.shelf or ""), _words(item.name)):
                return number
            shared = len(long_label & {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4})
            if shared and (best is None or shared > best[0]):
                best = (shared, number)
        if best:
            return best[1]
        return 1 if len(self.items) == 1 else None

    def settled_for(self, number: int, key: str, strict: bool = False) -> bool:
        """Whether a requirement the request states for this item already names this property. Strict: only the same
        words, or the same words with the product's own name or a short tag added (a "colour" must settles "kettle
        colour" but not "lid colour")."""
        want = _stem(key)
        item = self.items[number - 1]
        naming = _stem(item.name) | _stem(item.shelf or "")
        for must in item.musts:
            prop = _stem(must.split(":", 1)[0])
            if not prop:
                continue
            if strict:
                small, big = (prop, want) if len(prop) <= len(want) else (want, prop)
                if small <= big and all(len(w) <= 2 or w in naming for w in big - small):
                    return True
            elif _names(prop, want) or _names(want, prop):
                return True
        return False

    def ask_lanes(self, count: int) -> tuple[dict[int, list[list[Any]]], dict[int, float]]:
        """Per item: candidate properties to ask, each [label, chance it is an unwritten preference given that the
        item has one], best first; and per item the chance that it carries an unwritten preference at all.
        The heavy model reads the request's story for clues; the keys most of an item's listings carry back it up."""
        table = self.key_table()
        settled = [f"item {n}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing") for n, item in enumerate(self.items, 1)]
        listing: list[str] = []
        last = None
        own: dict[int, list[tuple[str, int]]] = {n: [] for n in range(1, len(self.items) + 1)}
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
            number = self.owner_of(label)
            if number and not self.settled_for(number, name):
                own[number].append((name, carried))
        names_each = WIDE_NAMES if len(self.items) > 1 else 0
        prompt = ASK_RANK.replace("COUNT", str(max(2, count, names_each))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        asked_first = [w for w in (_stem(x) - {"the", "and", "for", "with", "what", "which"} for x in first) if w]
        many = len(self.items) > 1
        lanes: dict[int, list[list[Any]]] = {}
        odds: dict[int, float] = {}
        given: dict[int, dict[str, Any]] = {}
        for entry in _items(plan.get("items")):
            if isinstance(entry, dict):
                number = _count(entry.get("item"))
                if number and 1 <= number <= len(self.items):
                    given.setdefault(number, entry)
        for number in range(1, len(self.items) + 1):
            entry = given.get(number) or {}
            # keys that name one property with the product's own name or a short tag in front ("kettle colour",
            # "xy colour") go under the plainest name, so a question asks it plainly and the property counts once
            item = self.items[number - 1]
            naming = _stem(item.name) | _stem(item.shelf or "")

            def qualifies(plain: str, key: str) -> bool:
                """`key` is `plain` with only qualifying words added (the product's own name, a short tag)."""
                extra = _stem(key) - _stem(plain)
                return bool(_stem(plain)) and _stem(plain) <= _stem(key) and all(len(w) <= 2 or w in naming for w in extra)

            keys: list[tuple[str, int]] = []
            for key, carried in sorted(own[number], key=lambda kc: (len(_stem(kc[0])), -kc[1])):
                home = next((i for i, (plain, _) in enumerate(keys) if qualifies(plain, key)), None)
                if home is None:
                    keys.append((key, carried))
                else:
                    keys[home] = (keys[home][0], max(keys[home][1], carried))
            most = max([c for _, c in keys] or [1])

            def spelled(text: str) -> str:
                """The item's key a property the model named stands for: the same words, or a plainer name of it."""
                want = _stem(text)
                for key, _ in keys:
                    if _stem(key) == want:
                        return key
                for key, _ in keys:
                    if qualifies(key, text):
                        return key
                return _words(text)

            weight: dict[str, float] = {}
            for rank, pick in enumerate(_items(entry.get("ranked"))):
                text = pick.get("key") if isinstance(pick, dict) else pick
                if not isinstance(text, str):
                    continue
                label = re.split(r"\s+(?:and|or)\s+|[/,;(]", text)[0].strip().rstrip("?.")
                if not label or len(label) > 40 or len(label.split()) > 5 or not re.search(r"[A-Za-z]", label):
                    continue
                # what the request asks to be checked with the shopper is asked whatever the filters say
                first_hand = any(_stem(label) & a for a in asked_first)
                if not first_hand and self.settled_for(number, label, strict=True):
                    continue
                label = spelled(label)
                raw = pick.get("weight") if isinstance(pick, dict) else None
                value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else max(5.0, 60.0 - 10 * rank)
                if 0 < value <= 1:
                    value *= 100.0
                if first_hand:
                    value = 100.0
                weight[label] = max(weight.get(label, 0.0), min(100.0, max(1.0, value)))
            # the keys most listings of the item carry back the model's picks up as a tie-breaker (its reading of the
            # story comes first); they stand in as candidates only when the model ranked nothing for the item
            ranked_any = bool(weight)
            for rank, (key, carried) in enumerate(sorted(keys, key=lambda kc: -kc[1])[:ASK_BACKUP]):
                base = ASK_PRIOR * (carried / most) * (1.0 - 0.08 * rank)
                if key in weight:
                    weight[key] += base * ASK_PRIOR_RANKED
                elif not ranked_any:
                    weight[key] = base
            total = sum(weight.values())
            ranked = sorted(weight.items(), key=lambda kv: -kv[1])
            # some mass stays on properties nobody named
            lanes[number] = [[label, (value / total) * (1.0 - ASK_UNSEEN) if total else 0.0] for label, value in ranked]
            raw = entry.get("chance")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                chance = float(raw) if 0 < raw <= 1 else float(raw) / 100.0
            else:
                chance = ASK_ODDS_MANY if many else ASK_ODDS_ONE
            low, high = (0.45, 0.85) if many else (0.4, 0.95)
            odds[number] = min(high, max(low, chance))
        self.trace.append({"kind": "plan", "model": {n: [[str((p.get("key") if isinstance(p, dict) else p) or "")[:30], (p.get("weight") if isinstance(p, dict) else None)]
                                                     for p in _items((given.get(n) or {}).get("ranked"))[:6]] for n in given},
                           "chance": {n: (given.get(n) or {}).get("chance") for n in given},
                           "lanes": {n: [[a, round(b, 3)] for a, b in v[:6]] for n, v in lanes.items()}, "odds": {n: round(v, 2) for n, v in odds.items()}})
        return lanes, odds

    def next_pick(self, lanes: dict[int, list[list[Any]]], odds: dict[int, float], tried: dict[int, set[str]]) -> tuple[int, str] | None:
        """The untried property most likely to be an unwritten preference. With several items, questions spread over
        them: unwritten preferences in such orders usually sit on different items, so an item already asked counts for less."""
        best: tuple[float, int, str] | None = None
        many = len(lanes) > 1
        for number, lane in lanes.items():
            spread = ASK_SPREAD ** len(tried[number]) if many else 1.0
            for label, share in lane:
                if _words(label) in tried[number] or share <= 0:
                    continue
                score = odds.get(number, 0.5) * share * spread
                if best is None or score > best[0]:
                    best = (score, number, label)
        if best is None or best[0] < ASK_FLOOR:
            return None
        return best[1], best[2]

    def missed(self, lanes: dict[int, list[list[Any]]], odds: dict[int, float], number: int, label: str) -> None:
        """A question on this property brought no preference: it is not the one, and the item's odds drop accordingly."""
        lane = lanes.get(number) or []
        share = next((s for lab, s in lane if _words(lab) == _words(label)), 0.0)
        before = odds.get(number, 0.5)
        if 0 < share < 1:
            odds[number] = before * (1 - share) / max(1e-6, 1 - before * share)
            for entry in lane:
                if _words(entry[0]) == _words(label):
                    entry[1] = 0.0
                else:
                    entry[1] = entry[1] / (1 - share)

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        lanes, odds = self.ask_lanes(count)
        tried: dict[int, set[str]] = {n: set() for n in lanes}
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        many = len(self.items) > 1
        # one question per message, each chosen after the last answer was read
        while self.msgs_free() > 0 and not self.finished and self.left() > 6:
            pick = self.next_pick(lanes, odds, tried)
            if pick is None:
                break
            number, label = pick
            names = [label]
            if many and not self.one_at_a_time:
                names = self.wide_names(lanes, tried, number, label)
            for name in names:
                tried[number].add(_words(name))
            text = self.question(number, label, None) if len(names) == 1 else self.wide_question(number, names)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            sent += batch
            if self.ask_failed:
                break
            before = {n: len(item.told) for n, item in enumerate(self.items, 1)}
            reading = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(reading, told)
            fresh = [n for n, count in before.items() if len(self.items[n - 1].told) > count]
            if len(names) > 1:
                if _items(reading.get("single")) and not fresh:
                    # the shopper takes one property per message: none of the names was answered, so none is ruled out
                    self.one_at_a_time = True
                    for name in names:
                        tried[number].discard(_words(name))
                    continue
                if fresh:
                    # the answer is about the first named property that is a wish: the ones named before it are not
                    heard = [self.bare_words(str(n).split(":", 1)[0]) for n in self.items[number - 1].told[before[number]:]] if number in fresh else []
                    first = next((i for i, name in enumerate(names) if self.bare_words(name) in heard), None)
                    for name in names[:first] if first is not None else []:
                        self.missed(lanes, odds, number, name)
                    for name in names[(first + 1) if first is not None else len(names):]:
                        tried[number].discard(_words(name))
                    for n in fresh:
                        odds[n] = odds.get(n, 0.5) * ASK_AGAIN_MANY
                    continue
                # no wish heard: only the first name is ruled out (a wish named later can sit behind a property the
                # shopper does not mind); the others stay available, a little less likely
                self.missed(lanes, odds, number, names[0])
                for name in names[1:]:
                    tried[number].discard(_words(name))
                    for entry in lanes.get(number) or []:
                        if _words(entry[0]) == _words(name):
                            entry[1] *= 0.5
                # a puzzled answer is no sign of a shopper who takes one question at a time: where several products are
                # asked about, a question that names none of the wishes is answered that way too
                continue
            for n in fresh:
                # a second unwritten preference on the same item is the rarer case
                odds[n] = odds.get(n, 0.5) * (ASK_AGAIN_MANY if many else ASK_AGAIN_ONE)
            if fresh:
                continue
            # any other answer, a complaint about the question's form included, is a miss: the same property asked
            # again in other words gets the same kind of answer, so the next message goes to another property
            self.missed(lanes, odds, number, label)
        found = self.chase_refs()
        if told:
            self.version += 1
        if not sent:
            rest = ("Values read by following the pages these requirements point at: " + " | ".join(found)) if found else ""
            if self.queued:
                rest += ("\n" if rest else "") + "[notes seen meanwhile] " + " | ".join(self.queued)
                self.queued = []
            return rest
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Wren's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick, together with the request's own requirements."
        if told:
            summary += (" When no listing you have opened meets them all, search with each new value and the product's name, and also list the item's"
                        " category (with the price limit when the store takes one) and page through it, reading the rows' words for the values"
                        " before opening the likeliest listings; a listing that drops an earlier requirement does not fit.")
        if found:
            summary += "\nValues read by following the pages these requirements point at (every page on the way is opened): " + " | ".join(found)
        pointed = [need for _, need in self.refs if any(need in t for t in told) and need not in self.ref_values]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def part_holder(self, key: str, skip: str = "") -> str | None:
        """The listing a part code names (never `skip`): one whose title carries the code, else one a fit reading found
        it on as its own code."""
        for ident in self.seen_order:
            if ident == skip:
                continue
            record = self.rows.get(ident) or {}
            title = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"title|name|heading|head", k, re.I))
            if key and key in self.code_key(title):
                return ident
        for ident in self.seen_order:
            if ident != skip and any(self.code_key(c) == key for c in (self.fit_info.get(ident) or {}).get("own") or []):
                return ident
        return None

    def walk_parts(self) -> list[str]:
        """Before a checkout with several items: open the page of every part that a cart line's own words send the reader
        to for the rest of its fit list (that page tells what else the line fits, and the order counts only when it was
        opened), finding each part by its code; returns the pages opened (also the ones opened before)."""
        if len(self.items) < 2:
            return []
        search = self.roles.get("search")
        page = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(page)
        if not search or not page or not target:
            return []
        self.fit_read(first=self.cart_ids())
        calls = 0
        opened: list[str] = []
        for line in self.cart_ids():
            info = self.fit_info.get(line) or {}
            mine = {self.code_key(c) for c in info.get("own") or []}
            for part in (info.get("more") or [])[:2]:
                key = self.code_key(part)
                if not key or key in mine or self.part_tries.get(key, 0) >= 2:
                    continue
                if self.finished or calls >= PART_STEPS or self.left() <= self.tail() + 3 or self.elapsed() >= HARD_STOP - 25:
                    break
                holder = self.part_holder(key, skip=line)
                if holder is None:
                    args = self.phrase_args(search, part)
                    if args is None:
                        continue
                    self.part_tries[key] = self.part_tries.get(key, 0) + 1
                    calls += 1
                    reply = self.act(search, args)
                    if reply is None:
                        continue
                    self.observe(search, reply.get("observation"))
                    holder = self.part_holder(key, skip=line)
                if holder is None:
                    continue
                if holder in self.opened:
                    opened.append(holder)
                    continue
                if calls >= PART_STEPS or self.left() <= self.tail() + 2:
                    break
                self.part_tries[key] = self.part_tries.get(key, 0) + 1
                calls += 1
                if self.open_link(page, target, holder):
                    opened.append(holder)
        self.part_pages = list(dict.fromkeys(self.part_pages + opened))
        if opened and self.elapsed() < self.wrap_at - 10:
            self.fit_read(first=self.cart_ids() + opened)
        if opened:
            self.trace.append({"kind": "parts", "opened": opened, "calls": calls})
        return opened

    def ask_name(self, number: int) -> str:
        """The words a question uses for an item. With several items: the store's label for its category, written as the
        store writes it, when the brief gave that label, the store shows that very category and no other requested item
        carries it too; in every other case (one item included) the item's name as the shopper gave it."""
        wanted = self.items[number - 1]
        shelf = (wanted.shelf or "").strip()
        if len(self.items) < 2 or not shelf:
            return wanted.name
        mine = _words(shelf)
        cat_key = self.category_key()
        shown = {_words(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in self.seen_order} if cat_key else set()
        taken = [o for k, o in enumerate(self.items, 1) if k != number and o.shelf and _words(o.shelf) == mine]
        return shelf.lower() if mine in shown and not taken else wanted.name

    def wide_names(self, lanes: dict[int, list[list[Any]]], tried: dict[int, set[str]], number: int, label: str) -> list[str]:
        """The pick and the item's next likeliest untried properties, likeliest first, WIDE_NAMES at most."""
        names = [label]
        for name, share in lanes.get(number) or []:
            if len(names) >= WIDE_NAMES:
                break
            if share > 0 and _words(name) not in tried[number] and all(_words(name) != _words(n) for n in names):
                names.append(name)
        # a short list is topped up with the keys most of the product's listings carry
        for label_of, key, _, _ in self.key_table():
            if len(names) >= WIDE_NAMES:
                break
            if self.owner_of(label_of) != number or self.settled_for(number, key) or _words(key) in tried[number]:
                continue
            if all(_words(key) != _words(n) for n in names):
                names.append(key)
        return names

    def wide_question(self, number: int, names: list[str]) -> str:
        """One message naming several properties of one product, the likeliest first."""
        plain = []
        for name in names:
            words = _words(name).split()
            while len(words) > 1 and len(words[0]) <= 2:
                words = words[1:]
            plain.append(" ".join(words) or name)
        listed = ", ".join(plain[:-1]) + " or " + plain[-1] if len(plain) > 1 else plain[0]
        return f"For the {self.ask_name(number)}, which {listed} would you like?"

    @staticmethod
    def bare_words(label: str) -> frozenset[str]:
        """The words that tell one property from another: a store's one- or two-letter tags are left out."""
        return frozenset(re.findall(r"[a-z0-9]{3,}", _words(label)))

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        # a store's short key codes in front of the property ("xq pattern") are no words a shopper would use
        words = _words(label).split()
        while len(words) > 1 and len(words[0]) <= 2:
            words = words[1:]
        text = f"What {' '.join(words) or label} would you like for the {self.ask_name(number)}"
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
                # the store turned the message down (no messages left, a malformed call): no more questions
                self.ask_failed = True
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
                return False, ("Before going on, Wren did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Wren: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, Wren did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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

    def item_records(self, number: int | None) -> list[dict[str, Any]]:
        """The attribute blocks of the opened listings of one requested item's kind (all of them with one item)."""
        mark = (number, len(self.full), self.version)
        cached = self.records_cache.get(mark)
        if cached is not None:
            return cached
        cat_key = self.category_key()
        blocks = []
        for ident, record in self.full.items():
            if len(self.items) > 1 and number and cat_key and self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or "")) != number:
                continue
            block = _fact_block(record)
            if block:
                blocks.append(block)
        self.records_cache[mark] = blocks
        return blocks

    def wanted_of(self, number: int | None, need: str) -> frozenset[str] | None:
        """The words of a requirement's wanted value that this item's listings use for that property, so the framing
        words of an answer fall away (no listing's value carries them). None for a number, a limit, alternatives, an
        exclusion of a value the listings use, or a property no opened listing states."""
        parsed = _value_words(need)
        if not parsed:
            return None
        words, negated = parsed
        prop = _stem(need.split(":", 1)[0])
        said: set[str] = set()
        for block in self.item_records(number):
            for key, value in block.items():
                if _names(prop, _stem(key)):
                    said |= _stem(" ".join(re.findall(r"[a-z]+", str(value).lower())))
        if not said:
            return None
        one = lambda w: next(iter(_stem(w)), w)
        if any(any(_near(one(w), x) for x in said) for w in negated):
            return None
        keep = [w for w in words if w not in negated and any(_near(one(w), x) for x in said)]
        return _stem(" ".join(keep)) if keep else None

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
        cat_key = self.category_key()
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                stem = _stem(prop)
                found = [(k, v) for k, v in block.items() if _names(stem, _stem(k))]
                if not found:
                    continue
                wanted = self.wanted_of(number, need)
                # an own key: the property's words, maybe with the product's own name in front (short code prefixes
                # aside); a key naming another part decides nothing; a plural or options key lists variants; any other
                # added words make a wider key
                item = self.items[number - 1] if 0 < number <= len(self.items) else None
                head = re.split(r"\b(?:with|for|of|in|on|from|that|which)\b", _words(item.name))[0].split()[-1:] if item and _words(item.name) else []
                product = _stem(" ".join(head) + " " + (item.shelf if item else "")) | _stem(str((self.rows.get(ident) or {}).get(cat_key or "") or ""))
                read = []
                for key, value in found:
                    extra = {w for w in _stem(key) if len(w) > 2 and not any(_near(w, p) for p in stem)} - product
                    said = _words(key).split()
                    # a key named by the plural of the property lists the variants on offer
                    listing = any(w + "s" in said or w + "es" in said for w in _words(prop).split())
                    kind = "wider" if extra else ("list" if listing else "own")
                    read.append((kind, key, value, _stated_part(wanted, value) if wanted else None))
                if not read:
                    continue
                rank = {"own": 0, "list": 1, "wider": 2}
                read.sort(key=lambda r: (r[3] is None, rank[r[0]]))
                hits = [f"{key}=" + (f"{part[:60]} [states it]" if part is not None else str(value)[:50])
                        + ("" if kind == "own" else (" (a list of variants)" if kind == "list" else " (a wider key)")) for kind, key, value, part in read[:4]]
                tag = f"(item {number}) " if many else ""
                verdict = ""
                own = [r for r in read if r[0] == "own"]
                if any(r[3] is not None for r in own):
                    verdict = " => met (an own key states the value; another own key saying something else does not undo it)"
                parts.append(f"{tag}{prop} -> " + ", ".join(hits) + verdict)
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
        return "lines by Wren's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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
        order = self.newest()
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first; a chase's own rows last
            order.sort(key=lambda i: (i not in self.full, i in self.chase_seen, self.price_of(i) if self.price_of(i) is not None else float("inf")))
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
        if key in self.checked or self.spent() > limit or self.cost > SPEND_HARD + 0.01 or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        pages = [w for w in self.part_pages if w not in ids][:4]
        if pages:
            prompt += "\n\nCONTINUATION PAGES OPENED (each carries the rest of a cart line's list of what it fits):\n" + "\n".join(f"{i}: " + self.record_of(i) for i in pages)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        fit = self.fit_lines(ids)
        if fit:
            prompt += "\n\nFIT READING (what each cart listing's own words say it is and what it fits):\n" + fit
        # the heavy model checks while the episode is cheap; a costly episode (many items, long searches) checks with the
        # light one first, so a run of episodes stays inside its spending cap
        got = self.ask_json(prompt, [HEAVY, LIGHT] if self.cost - self.fit_cost < CHECK_HEAVY_BELOW else [LIGHT, HEAVY], 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        self.check_fail = {"ids": list(ids), "problems": problems, "better": better, "version": self.version}
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Wren opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        self.chase_refs()
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
            order = list(reversed(self.seen_order))
            # rows only a price walk brought in come after the rest, so they cannot crowd out the chain's pages
            for ident in [i for i in order if i not in self.walk_only] + [i for i in order if i in self.walk_only]:
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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 4000) or {}
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
            text += "[Wren opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def phrase_args(self, tool: str | None, text: str) -> dict[str, Any] | None:
        """Arguments for a search: the text in the field the loop model searched with (its mode choices kept), else
        in the field named like a query box, else the required text field, else the first other text field."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        # a page token or cursor is a string too
        free = [k for k in texts if not re.search(r"token|cursor|page|offset|after|next", k, re.I)]
        form = self.search_form.get(tool or "") or {}
        boxy = [k for k in free if re.search(r"query|^q$|_q$|^q_|text|phrase|keyword|search|term|words|find|needle", k, re.I)]
        target = form.get("box") if form.get("box") in free else None
        target = next((k for k in required if k in free), None) or target or (boxy[0] if boxy else None) or (free[0] if free else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key, value in (form.get("modes") or {}).items():
            if key in props and key != target and key in required:
                args[key] = value
        for key in required:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def chase_refs(self) -> list[str]:
        """Follow every live requirement whose value is on another listing and not read yet; returns what was read."""
        found: list[str] = []
        live = {need for _, need in self.needs_now()}
        for number, need in list(self.refs):
            if need in live and need not in self.ref_values and need not in self.chased and self.chase_room():
                known = set(self.seen_order)
                stepped = self.step
                got = self.chase(number, need)
                self.chase_tries[need] = self.chase_tries.get(need, 0) + 1
                if not got and self.step == stepped and self.chase_tries[need] < 2:
                    self.chased.discard(need)
                # rows this chase alone brought in go last in the review windows, whichever way it ended
                self.chase_seen.update(i for i in self.seen_order if i not in known)
                if got:
                    found.append(got)
        return found

    def chase_floor(self) -> int:
        """Steps a chase always leaves: enough to find, cart and check out every item after it."""
        return max(CHASE_KEEP, 6 * len(self.items) + 4, self.max_steps // 3)

    def chase_ask(self, prompt: str) -> dict[str, Any]:
        """A chase call on the light model; a reply that is not JSON gets one longer retry. Its spend is booked apart."""
        before = self.cost
        got = self.ask_json(prompt, [LIGHT], CHASE_TOKENS)
        if got is None and not self.no_credit and self.calls < ALL_CALLS - 4:
            got = self.ask_json(prompt, [LIGHT], 4000)
        self.chase_cost += self.cost - before
        return got or {}

    def chase_room(self) -> bool:
        # model calls stay in reserve for the checkout check, the process gate, the buyer and a few loop turns
        return (not self.finished and not self.broke and not self.no_credit and self.left() > self.chase_floor()
                and self.chase_cost < CHASE_SPEND and self.cost < SPEND_HARD + 0.01 and self.elapsed() < self.wrap_at - CHASE_MARGIN
                and self.calls + 6 <= ALL_CALLS - (10 + 2 * len(self.items)) and self.chase_calls < CHASE_CALLS)

    def chase(self, number: int, need: str) -> str:
        """One requirement's chain: find each described listing, open its page, read the value or the next link."""
        search = self.roles.get("search")
        page = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(page)
        if not search or not page or not target:
            return ""
        self.chased.add(need)
        start = self.step
        budget = max(8, self.max_steps // 3)
        known = set(self.seen_order)
        describe = need
        words = ""
        path: list[str] = []
        steps: list[str] = []
        dead: set[str] = set()
        looks = reads = backs = 0
        fresh: list[str] = []
        tried: set[str] = set()
        seller = ""
        while self.chase_room() and len(path) < CHASE_LINKS and self.step - start < budget:
            # a short task does not wait long for the first link
            if not path and self.step - start >= max(6, budget // 2):
                break
            self.chase_seen.update(i for i in self.seen_order if i not in known)
            known = set(self.seen_order)
            if words:
                looks += 1
                tried.add(words.lower())
                args = self.phrase_args(search, words)
                words = ""
                if args is None:
                    break
                reply = self.act(search, args)
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(search, seen)
                fresh = [i for i, _ in self.records_in(seen)]
                continue
            self.chase_calls += 1
            got = self.chase_ask(LINK_PICK + "\n\nREQUIREMENT:\n" + need + "\n\nLINK:\n" + describe + "\n\nRECORDS:\n"
                                 + self.chase_pool(describe, path, fresh, dead))
            pick = str(got.get("pick") or "").strip()
            seller = seller or str(got.get("seller") or "").strip()
            if pick not in self.rows or pick in path or pick in dead:
                pick = ""
            if not pick:
                unread = [i for i in (str(x).strip() for x in _items(got.get("read"))) if i in self.rows and i not in self.full and i not in dead][:2]
                closed = [i for i in (str(x).strip() for x in _items(got.get("page"))) if i in self.rows and i not in self.link_pages and i not in dead][:2]
                room = budget - (self.step - start)
                unread = unread[:max(0, min(2, room))]
                closed = closed[:max(0, min(2, room - len(unread)))]
                if (unread or closed) and reads < 3:
                    reads += 1
                    if unread:
                        self.fill_facts(unread, self.chase_floor())
                    for ident in closed:
                        if not self.chase_room() or not self.open_link(page, target, ident):
                            break
                    continue
                if looks >= CHASE_LOOKS:
                    break
                options = [str(got.get("search") or "").strip(), seller] + ([" ".join(describe.split()[:6])] if path else [])
                words = next((w for w in options if w and w.lower() not in tried), "")
                if not words:
                    break
                continue
            shown = self.page_seen.get(pick)
            if pick not in self.link_pages and shown and shown[0] == page:
                self.link_pages[pick] = shown[1]
            if pick not in self.link_pages and not self.open_link(page, target, pick):
                break
            path.append(pick)
            steps.append(f"{len(steps) + 1}. {describe if steps else 'the listing the requirement itself describes'} -> listing {pick}")
            got = self.link_read(need, steps, pick)
            if (not str(got.get("value") or "").strip() and not str(got.get("next") or "").strip() and pick not in self.full
                    and self.chase_calls < CHASE_CALLS):
                # the page sends nowhere and states nothing: its attributes may hold the value
                self.fill_facts([pick], self.chase_floor())
                if pick in self.full:
                    got = self.link_read(need, steps, pick)
            value = str(got.get("value") or "").strip()
            onward = str(got.get("next") or "").strip()
            self.trace.append({"kind": "chase", "need": need[:120], "link": pick, "value": value[:80], "next": onward[:120]})
            if value and not onward:
                rule = str(got.get("rule") or "").strip()
                self.ref_values[need] = value + (f" (so: {rule})" if rule else "")
                self.version += 1
                return f"{need} -> {self.ref_values[need]} (pages opened: {' -> '.join(path)})"
            if not onward:
                # a look-alike: this listing's page neither states the property nor sends it on; try another match
                dead.add(pick)
                path.pop()
                steps.pop()
                backs += 1
                if backs > 2:
                    break
                continue
            self.chase_seen.update(i for i in self.seen_order if i not in known)
            known = set(self.seen_order)
            describe = onward
            words = str(got.get("search") or "").strip()
            seller = str(got.get("seller") or "").strip()
            looks = reads = 0
            fresh = []
            tried = set()
        return ""

    def open_link(self, page: str, target: str, ident: str) -> bool:
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(page), "args", faults)
        if faults:
            return False
        reply = self.act(page, args)
        if reply is None or self.finished:
            return False
        seen = reply.get("observation")
        self.observe(page, seen)
        self.note_view(page, seen)
        self.link_pages[ident] = seen
        return True

    def link_read(self, need: str, steps: list[str], ident: str) -> dict[str, Any]:
        self.chase_calls += 1
        record = _dump(_clip(self.link_pages.get(ident), 1200))[:4500]
        facts = _fact_block(self.full.get(ident) or {})
        if facts:
            record += "\nATTRIBUTES: " + _dump(_clip(facts, 200))[:1500]
        return self.chase_ask(LINK_READ + "\n\nREQUIREMENT:\n" + need + "\n\nCHAIN FOLLOWED SO FAR:\n" + "\n".join(steps)
                              + "\n\nPAGE OF " + ident + " (the last link):\n" + record)

    def chase_pool(self, describe: str, path: list[str], fresh: list[str], dead: set[str]) -> str:
        """Records for the link picker: the latest search's rows first, then rows sharing words with the description."""
        want = {w for w in _stem(describe) if len(w) >= 4}
        scored = []
        for ident in self.seen_order:
            if ident in path or ident in dead:
                continue
            row = self.rows.get(ident) or {}
            text = _dump(_clip(row, 120)).lower()
            hits = sum(1 for w in want if w in text)
            if ident in fresh or hits:
                scored.append((0 if ident in fresh else 1, -hits, ident))
        scored.sort()
        lines = []
        for _, _, ident in scored[:18]:
            line = ident + ": " + _dump(_clip(self.rows.get(ident), 120))[:480]
            facts = _fact_block(self.full.get(ident) or {})
            if facts:
                line += " | ATTRIBUTES " + _dump(_clip(facts, 80))[:900]
            if ident in self.link_pages:
                line += " | PAGE " + _dump(_clip(self.link_pages[ident], 500))[:1300]
            lines.append(line)
        return "\n".join(lines) or "(no records yet)"

    def newest(self) -> list[str]:
        """Listings newest first, the ones a chase alone brought in after the rest."""
        order = list(reversed(self.seen_order))
        return [i for i in order if i not in self.chase_seen] + [i for i in order if i in self.chase_seen]

    def spent(self) -> float:
        """Spend outside the chases and fit readings, which the checks' spend gates go by."""
        return self.cost - self.chase_cost - self.fit_cost

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
        text = f"[Wren] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        if self.spent() > SPEND_SOFT or self.cost > SPEND_HARD + 0.01:
            return []
        records = []
        for ident in self.newest():
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
        # a replacement is bought nearly unchecked: the heavy model picks it while the episode is cheap
        order = [HEAVY, LIGHT] if self.cost < CHECK_HEAVY_BELOW else ([LIGHT] if self.cost >= SPEND_HARD else [LIGHT, HEAVY])
        got = self.ask_json(prompt, order, 4000)
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
        if self.audited or self.spent() > SPEND_SOFT or self.cost > SPEND_HARD + 0.01:
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
        records = []
        for ident in self.newest():
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
        got = self.ask_json(prompt, [LIGHT] if self.cost >= SPEND_HARD else [LIGHT, HEAVY], 4000)
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
        """The agent ends without buying although the independent check confirmed a reviewed listing: Wren carts the reviewed
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
        return ("Not run (Wren acted instead): the independent check confirmed " + self.confirmed + ", so Wren set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def decisive(self) -> str:
        """Before an ending without a purchase: cart the closest listing for each item and check out, unless some item
        has no listing of its kind in stock inside the budget (then the ending goes ahead)."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        # a second try only when the requirements or the budget changed since the first (an ending refused by a change),
        # or the episode moved on
        state = (self.version, self.budget)
        if self.decided >= 2 or (self.decided and state == self.decided_state and self.step <= self.decided_at + 1):
            return ""
        if not cart or not buy or self.finished or self.left() < len(self.items) + 2 or self.cost > SPEND_HARD + 0.02:
            return ""
        self.decided += 1
        self.decided_at = self.step
        self.decided_state = state
        records = []
        limit = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        # candidates: the cart's lines first, then each category's newest listings (opened ones before rows), so that
        # every requested kind is shown even when later searches filled the list with one kind
        pool = [i for i in self.newest() if not self.sold_out(i) and self.buyable(i)
                and not (limit is not None and (self.price_of(i) or 0) > limit)]
        cat_key = self.category_key()
        per_kind = max(12, DECIDE_RECORDS // max(1, len(self.items)))
        chosen_ids = [i for i in self.cart_ids() if i in pool]
        counts: dict[str, int] = {}
        for opened in (True, False):
            for ident in pool:
                if ident in chosen_ids or (ident in self.full) != opened:
                    continue
                kind = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
                if counts.get(kind, 0) >= per_kind:
                    continue
                counts[kind] = counts.get(kind, 0) + 1
                chosen_ids.append(ident)
                if len(chosen_ids) >= DECIDE_RECORDS:
                    break
            if len(chosen_ids) >= DECIDE_RECORDS:
                break
        for ident in chosen_ids:
            record = self.full.get(ident)
            records.append(("OPENED " if record else "ROW ") + ident + ": " + _dump(_clip(record or self.rows.get(ident), 120))[:800])
        if not records:
            return ""
        sold = [i for i in self.cart_ids() if self.sold_out(i)]
        prompt = BEST + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nCART NOW:\n" + self.cart_text()
        if sold:
            prompt += "\nSOLD OUT in the cart (these lines must be replaced): " + ", ".join(sold)
        if self.notes:
            prompt += "\n\nSTORE NOTES SEEN:\n" + " | ".join(self.notes[-4:])
        if self.check_fail and self.check_fail.get("version") == self.version:
            # only the lines a problem names (or that a better listing replaces) count as rejected; a problem naming no
            # line (an item without a line) leaves the lines that are there in place
            named = [i for i in self.check_fail["ids"] if any(i in p for p in self.check_fail["problems"])]
            named = named or (list(self.check_fail["ids"]) if self.check_fail["better"] else [])
            prompt += (("\n\nTHE LATEST CHECK REJECTED THESE LINES: " + ", ".join(named) + " (" if named else "\n\nTHE LATEST CHECK FOUND (")
                       + "; ".join(self.check_fail["problems"]) + ")" + (("; it named as better: " + ", ".join(self.check_fail["better"])) if self.check_fail["better"] else ""))
        prompt += "\n\nLISTING RECORDS (in stock, newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT] if self.cost >= SPEND_HARD else [LIGHT, HEAVY], 4000) or {}
        self.trace.append({"kind": "decisive", "got": _clip(got, 120)})
        if [x for x in _items(got.get("missing")) if _count(x)]:
            return ""
        chosen: dict[int, str] = {}
        for entry in _items(got.get("lines")):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            number = _count(entry.get("item"))
            if (ident in self.rows and self.buyable(ident) and not self.sold_out(ident) and number and 1 <= number <= len(self.items)
                    and number not in chosen and ident not in chosen.values()):
                chosen[number] = ident
        if len(chosen) < len(self.items):
            return ""
        plan = {ident: self.items[number - 1].qty for number, ident in chosen.items()}
        cat_key = self.category_key()
        kinds = [str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() for i in plan] if cat_key else []
        if len(plan) > 1 and kinds and all(kinds) and len(set(kinds)) < len(kinds):
            return ""
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
            prices = {i: self.price_of(i) for i in plan}
            # with the store's price field known, a planned listing showing no price lets the ending stand
            if self.price_key and any(p is None for p in prices.values()):
                return ""
            if any(p is None for p in prices.values()):
                prices = {i: p or 0 for i, p in prices.items()}
            each = self.brief.get("budget_scope") == "each"
            over = any(prices[i] > self.budget for i in plan) if each else sum(prices[i] * plan[i] for i in plan) > self.budget
            if over:
                return ""
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or (not calls and not self.cart_ids()):
            return ""
        if len(calls) + self.tail() > self.left():
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
            self.cart_said = _dump(reply.get("observation"))
            done.append(_dump(args))
        self.recent = []
        lost = [i for i in plan if self.sold_out(i)]
        if not lost and (self.version, self.budget) != self.decided_state:
            # a change read on these cart calls (budget, requirements): the plan is stale, so no checkout yet
            self.decided_at = self.step
            return ("Not run (Wren acted instead): Wren set the cart (" + "; ".join(done) + "), but the store's reply changed the terms"
                    + (f" (budget now {self.budget})" if self.budget is not None else "") + ": re-check the lines against the ledger, fix them, then check out.\nCart now: " + self.cart_text())
        if lost:
            self.decided_at = self.step
            return ("Not run (Wren acted instead): Wren set the cart (" + "; ".join(done) + "), but " + ", ".join(lost)
                    + " sold out meanwhile: replace that line with the next closest listing, then check out.\nCart now: " + self.cart_text())
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults or self.left() < 1:
            return ""
        self.walk_parts()
        go, said = self.gate("order")
        if not go:
            return ("Not run (Wren acted instead): ending empty-handed earns nothing while listings can serve, so Wren set the cart ("
                    + "; ".join(done) + "). Before the checkout: " + said + "\nCart now: " + self.cart_text())
        reply = self.act(buy, args)
        self.trace.append({"kind": "decisive_buy", "lines": plan, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        self.decided_at = self.step
        extra = ""
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        if not reply.get("finished") and not self.finished:
            # the same reading of a refused checkout as the agent's own: a lone line named sold out, and replacements
            if self.note_sold:
                extra += self.find_sold()
            hint = self.swap_hint()
            extra += ("\n" + hint) if hint else ""
        return ("Not run (Wren acted instead): ending empty-handed earns nothing while listings can serve, so Wren carted the closest listing for each item ("
                + "; ".join(done) + ") and checked out. Store reply: " + self.show(reply.get("observation")) + extra)

    def cart_fits(self) -> bool:
        """No line known sold out, and the known prices fit the budget."""
        ids = self.cart_ids()
        if any(self.sold_out(i) for i in ids):
            return False
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and ids:
            prices = {i: self.price_of(i) for i in ids}
            if all(p is not None for p in prices.values()):
                each = self.brief.get("budget_scope") == "each"
                lines = self.cart_lines or {i: 1 for i in ids}
                return all(prices[i] <= self.budget for i in ids) if each else sum(prices[i] * lines.get(i, 1) for i in ids) <= self.budget
        return True

    def buyer_finish(self) -> None:
        """The last ending's buyer: plan and check out; after its own refused checkout it re-plans once under the new
        terms; a cart it filled but did not check out goes through only when no line is sold out and it fits the budget."""
        buys = self.buys
        done = self.decisive()
        if not done or self.finished:
            return
        refused = self.buys > buys
        if (refused or (self.version, self.budget) != self.decided_state) and self.left() >= len(self.items) + 2:
            buys = self.buys
            done = self.decisive()
            if self.finished:
                return
            refused = self.buys > buys if done else refused
        if self.cart_lines and self.cart_fits() and self.left() >= 1 and not refused:
            self.buy_now()

    def buy_now(self) -> bool:
        """A forced checkout of the cart as it stands (the last steps of an episode); True when the store took it."""
        buy = self.roles.get("buy")
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults) if buy else {}
        if not buy or faults or self.finished or self.left() <= 0:
            return False
        self.walk_parts()
        if self.left() >= 2:
            self.gate("order")
        if self.finished or self.left() <= 0:
            return False
        reply = self.act(buy, args)
        self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
        return bool(self.finished)

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
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Wren opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Wren's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Wren] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Wren] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
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
                    self.walk_parts()
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    self.price_walk()
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Wren read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Wren re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
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
                done = self.complete_cart() or self.decisive()
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
                    trimmed = "\n[Wren] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Wren] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            # rows the agent's own calls show are no longer the chase's alone
            self.chase_seen.difference_update(i for i, _ in self.records_in(observation))
            self.walk_only.difference_update(i for i, _ in self.records_in(observation))
            slot = self.single_target(name)
            if slot and isinstance(args.get(slot), str) and isinstance(observation, dict) and not observation.get("error"):
                self.page_seen[args[slot].strip()] = (name, observation)
            # A: the price and stock keys come from the first search result, whatever the pick rule
            if name == self.roles.get("search") and self.price_key is None and not ending:
                self.read_shape(name, observation)
            # the field and mode choices of a search of the model's own that found rows (the chase searches the same way)
            if name == self.roles.get("search") and self.records_in(observation):
                spec = (self.schemas.get(name) or {}).get("properties") or {}
                words = [k for k, v in args.items() if isinstance(v, str) and isinstance(spec.get(k), dict) and "enum" not in spec[k]
                         and not re.search(r"token|cursor|page|offset|after|next", k, re.I) and len(v.split()) >= 1]
                modes = {k: v for k, v in args.items() if isinstance(spec.get(k), dict) and isinstance(spec[k].get("enum"), list) and v in spec[k]["enum"]}
                need = (self.schemas.get(name) or {}).get("required") or []
                box = next((k for k in words if k in need), None) or next(
                    (k for k in words if re.search(r"query|^q$|_q$|^q_|text|phrase|keyword|search|term|words|find|needle", k, re.I)), None)
                if box:
                    self.search_form[name] = {"box": box, "modes": {k: v for k, v in modes.items() if k in need}}
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
                    text += "\n[Wren] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            if name == self.roles.get("cart") and not ending and not self.finished and self.cart_ids():
                text += self.walk_now()
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Wren re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Wren's own calls] " + " | ".join(self.queued)
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
        if refuse and buy and not holding and not (self.cart_touched and not fresh) and self.left() >= len(self.items) + 2:
            if again:
                done = self.decisive()
                if done:
                    self.trace.append({"kind": "forced_decisive", "reply": done[:200]})
                    if self.finished or self.left() <= 0:
                        return ""
                    return done
            else:
                self.buyer_finish()
                if self.finished or self.left() <= 0:
                    return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if not again and holding and buy and refuse and self.decided < 2 and self.left() >= len(self.items) + 2 and not self.cart_fits():
            # a held cart with a sold-out line or over the budget: the buyer re-plans before any forced checkout
            self.buyer_finish()
            if self.finished or self.left() <= 0:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if holding and buy and refuse and self.decided < 2 and self.left() >= len(self.items) + 2 and self.partial():
            # a cart missing a requested item never passes: the buyer adds the closest listing for each missing item
            if again:
                done = self.decisive()
                if done:
                    self.trace.append({"kind": "forced_decisive", "partial": True, "reply": done[:200]})
                    if self.finished or self.left() <= 0:
                        return ""
                    return done
            else:
                self.buyer_finish()
            if self.finished or self.left() <= 0:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                self.walk_parts()
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
                if reply is not None and not self.finished:
                    self.observe(buy, reply.get("observation"))
                    if not again and not self.cart_lines and self.left() >= len(self.items) + 2:
                        # an emptied cart was checked out: the buyer fills it, then it goes through
                        self.buyer_finish()
                        if self.finished or self.left() <= 0:
                            return ""
        if refuse and not self.finished and self.left() > 0:
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
                    if not again and self.left() >= len(self.items) + 2:
                        # the last ending was refused by a change: the buyer takes its try under the new terms
                        self.buyer_finish()
                        if self.finished:
                            return ""
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
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.elapsed() < SECOND_ROUND_BY:
                self.wrap_at = SECOND_ROUND_BY
                self.chat.append({"role": "user", "content": "[Wren] Wren tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.spent() < spend_cap
               and self.elapsed() < self.wrap_at and not self.out_of_room()):
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
        episode = Wren(problem_data)
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
