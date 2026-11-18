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
LOOP_MAX = 64               # the walker's turn cap grows with the store's step limit up to this
SPEND_SOFT = 0.052
SPEND_HARD = 0.075
SPEND_BY_ITEMS = {1: (0.052, 0.075), 2: (0.045, 0.065)}   # (soft, hard) per task by requested items; more items:
SPEND_MANY = (0.035, 0.055)
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 10             # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3
SWEEP_OPENS = 2             # cheaper listings of a line's department Kyo opens itself per item and sweep (cheapest pick)
SWEEP_ROOM = 10             # free steps a sweep before the order leaves beyond the finish; one after a change leaves 2
DEADLINE = 235              # seconds after the start at which the walker stops and Kyo ends the episode
ASK_UNTIL = 175             # past this the interviewer sends its remaining questions in one burst, read once
ASK_WIDTH = 6               # most properties one message names: race 17's shopper weighed a list's first six places
                            # (about 0.9, 0.8, 0.8, 0.5, 0.5, 0.5); race 18's 1.0, 0.77, 0.37, then 0.22 in lists of 4-6
                            # (0.56 at place 3 in lists of 2-3). Replayed on race 18's single-item episodes, six keys with
                            # the likeliest lead (next_ask) reveal as much as three (57% vs 54%) and far more under race
                            # 17's shopper (68% vs 54%)
PLAN_KEYS = 6               # keys the planner ranks for each item (and its band for hinted moves)
PACK_WIDTH = 6              # properties a message naming several items shares out among them (at most PACK_KEYS each);
                            # race 18: the first item's key at its place 1 drew the need 16 times in 17 in such messages
ASK_BY = 20                 # in a long task the interview starts by this step even when few listings are open yet
                            # (race 17: 51 of 177 episodes first asked at step 25 or later and 1 passed; the champion
                            # asked those same tasks by about step 21 and passed 16 of 102)
LINK_OPENS = 4              # pages one chain link may open before it is given up (PICK's own pick is always read)
ASK_SPARE = 6               # properties one message names when none of the open ones is a good key
# how surely the shopper weighs the property at each place of a list message, between race 17's shopper (0.9, 0.8, 0.8,
# 0.5, 0.5, 0.5) and race 18's (1.0, 0.77, 0.37, 0.22, 0.22, 0.22 in lists of 4-6). A key that drew no need keeps (1 - this)
# of its chance, and next_ask leads every message with its likeliest open key: after a six-key 'no preference' that is the
# list's 4th key (weighed least), then its 3rd and 5th, with never-asked keys from the 4th place on (v5.10 led with the
# list's 2nd key: in race 18 its 2nd message re-asked 1.6 well-weighed keys first and drew a need 8% of the time, against
# 31-37% for the leaders' other leads)
HEARD_AT = (1.0, 0.8, 0.6, 0.35, 0.35, 0.35, 0.3, 0.3)
SINGLE_MIN = 8              # a key with one value on at least this many opened listings goes last (race 14: 0.2% held a
                            # need). Race 18 read 3-6 listings of a shelf before asking, and its needs often sat on a key
                            # those few listings all shared one value of (closure 3 of 3, pack type, pattern): with the bar
                            # at 3 the need's key ranked in the first six 66% of the time, at 8 72%
PRIOR_STEP = 0.75           # each later planner place holds the unwritten need about this share as often as the one before
OPEN_MIN = 0.0              # a key is not named again once its unweighed share is down to this: it led a message
PUZZLE_AFTER = 3            # once this many messages went out, an item that puzzled the shopper gets no more
PACK_MAX = 3                # most items one message names when more items wait for a first question than messages are
                            # left (race 17: 41 of 90 tasks asked for 3 to 5 items with 2 messages; a preference sat on
                            # an item one-item messages never reached in most of them, and a message naming two items drew
                            # it 16 times in 20 when it named its item and property: 14 passes of 82 against 5)
PACK_KEYS = 3               # most properties such a message names for each of its items
READ_SLACK = 20             # an answer is read before the next message only while ASK_UNTIL is this far off
READ_SLOW = 12              # a read slower than this (seconds) ends the reading between messages
SPARSE_KEYS = 4             # an item with fewer keys of its own in the table is asked about the planner's own labels
SPARSE_COUNT = 12           # labels the planner names itself for such an item: enough for every message after a need came
RAW_SHOWN = 2               # raw store replies shown to the planner when an item has few keys
LATE = 250                  # past this no further model call is started
HARD_END = 290              # every call's timeout is cut so that it returns before this
THINK_OFF_OK = {LIGHT}      # models that take thinking off; Sonnet 5.5 answers the switch with HTTP 400
END_CALL = 15               # a model call started past DEADLINE gets at most this many seconds and no second try
TOOL_KEY = " tool"          # in a planned cart call: the store tool to send it to when it is not the cart tool
OPEN_END = 265              # past this no listing is opened: the cart calls and the checkout keep the rest

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). Rows come in the store's own ranking: before rewording a search a third time, read its next page (the store's token or offset). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. When the pages state how values go together ("that X style goes together with Y", a pairing or matching rule) and the requirement asks for a value that goes with, pairs with or is compatible with that listing's, the requirement's value is the paired one (Y), not the listing's own (X); a requirement for the very same value, or one matching it, takes the listing's own value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words: what the accessory says it works with (a family written with x's covers every model code that begins the same way, except the codes it lists), or the part series the device's page says it takes (a part whose own code begins with one of them); when a page leaves the rest of its list to another part's page, open that page. A search by a code can come back empty in some stores: then browse the item's department and compare the codes its rows show. When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
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
- A change's wording does not show whether a number went up or down: compare the numbers, as the ledger's [changes so far] does. A higher budget or a dropped requirement can make a listing you turned down fit, and under a cheapest pick a cheaper one may now win; a lower budget or a higher price can push the total over the limit; a sold-out line needs a replacement chosen by the same pick rule.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product, "fits": the number (counting from 1 in this list) of the other requested item that the request says this product has to be compatible with, or fit onto or into (a lens for the camera body also being bought, a lid made for the pot also being bought), else 0; products that are only bought together get 0, "noun": the product's plain name as store titles would print it, in one or two words ("a weather vane for the barn" gives "weather vane"), "find": one entry per must, in the same order: that must's value in the words a listing would print, for a store search ("handle: wrapped in cork" gives "cork handle"), or "" for a limit on a number, an exclusion or a value read from another listing}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"cart_modes": when the "cart" tool takes an argument with a fixed list of values saying whether a call puts units in or takes them out, {"put": the value that puts units in, "take": the value that takes them out}, copied exactly from that list as the tool's description explains them; else null.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper, by ranking the properties to ask about. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property. The agent names several properties of one product in each message and has only a few messages, so your ranking decides which properties get asked at all.

You get the request, the items with what the request already settles for each, and, for each category, the attribute keys seen on its opened listings in their DEFAULT ORDER, the order the store's own records give: keys carried by the most opened listings first (among those, keys taking more different values first); keys few listings carry, and keys with one value on many listings, last. A key whose few listings all share one value can still hold a preference: the shopper may want another value. Each key shows how many opened listings carry it, how many different values it takes, and some of those values. Unwritten preferences sit mostly on the keys near the top of that order.

For each item, list its category's keys in the default order, changed only like this:
- Leave out a key the request already settles for that item, a second name for a key already listed, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics.
- Move a key up when the request's story points at an unwritten preference on it without giving its value (a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use it, or what is inside it), and only when that key is among the first BAND keys of its category; give such a key its hint. A key further down keeps its place, hint or not.
- What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any) goes first for its item, named by the property it is about.
- A property that no listed key names is seldom a preference: add one only for such a hint, after that item's listed keys.
- An item marked "few store keys" still gets SPARSE_N properties, all named by you: the ones a shopper of that product most likely holds an unwritten preference on, best first, in the store's own words when STORE REPLIES below show them, else in plain product language. Leave out what the request settles; each label one property, as below.
"label" is the key's plain name: all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Give each item up to COUNT keys (SPARSE_N for an item marked "few store keys"), best first: the agent names an item's keys in the order you give, and a shopper answers about the first property a message names almost always, about the second and third less surely and about later ones seldom, so the item's first key is the one most likely to hold an unwritten preference, and its first three are the three likeliest. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false, "find": the wanted value in the answer's words as a listing would print it, for a store search (for example mottled lichen); for a limit on a number, that figure with its unit as a listing prints it (for example 9 rungs); an empty string when the need is an exclusion or a pointer to another listing}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things together or wants one question at a time; not answers that only say they did not follow it], "lost": [numbers of the questions whose answer only says the shopper did not follow or understand it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- A question may name several properties of one product. Its answer is about the property the answer itself names: put the need under that property, never under the first one the question listed. An answer that minds none of them, or only did not follow, states no need on any of them.
- A question may also ask about several products at once, each with its own properties: put each need under the item whose product the answer names (or whose properties it answers).
- An answer that turns down every value of a property but one means the property must be that value ("any awning cloth except hessian gets sent back" is "awning cloth: hessian").
- A wanted value written as a word such as none or nil is a literal value a listing prints, not the absence of a requirement ("the embossing has to read nil" is "embossing: nil").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about. An answer that only says they did not follow the question goes under "lost", not "free" or "crowded".
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole, and when another department label of this store (listed below) names the requested item more exactly than the line's own label does, the line is in the wrong department (a label that joins several kinds names each of them as exactly as a label of one kind); a rule of being unlike another listing's value counts as met unless the records show that same value; when the request says one requested item must fit another (an accessory for a device), the records must link the two cart lines: the accessory's compatible list, the size or model it names, or codes from either side (the accessory's list or code family covers the device's model code, or the device's record names a part series that the accessory's own part code begins with). Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first], "keep": [{"item": item number, "id": listing id} for each requested item whose cart line is the right listing for it (it meets every requirement of that item), at most one per item; a line that is right except for its quantity is still the right listing]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge; when two keys name one property, one spelt with a space and one with an underscore, and their values differ, the requirement is met when either value meets it); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a list of the models a listing works with is met by any model it includes, and a family written with x's covers every model code that begins the same way except the ones it lists; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the value the requirement asks for: the property's value as stated on that last page, or, when the requirement asks for a value that goes with, pairs with or is compatible with it (not the very same or a matching one) and a page states which value goes together with it ("that X style goes together with Y"), that paired value; else ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

PRICES = """A shopping agent must find the cheapest listing of one kind in an online store. From the store's tools below, name the one tool that lists listings priced at or below a highest price the agent passes, and its parameters, by their exact names. Return ONE JSON object and nothing else: {"tool": that tool's name, or null when no tool takes a highest price, "cap": its parameter for the highest price, "shelf": its parameter that takes a category or department label, or null, "words": its parameter that takes free search words, or null}"""

PAGE_ARG = """Below are a store tool's description and parameters. Which parameter asks for a later part of its results (an offset or first row, a page number, or a continuation token)? Return ONE JSON object and nothing else: {"page": that parameter's exact name, or null when none does}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


LINK = """You read a chain link. The text below is either a shopper's requirement whose value is to be taken from another listing, or a listing page's words that send a property on to a further listing. Return ONE JSON object and nothing else: {"seller": the one shop or brand name that sells the listing it points at, as written (without "'s"), "kind": the kind of product that listing is, in the text's words, "with": [each property the text uses to pick that listing out from the shop's others, as "property value"], "values": [the value alone of each "with" entry, as the text writes it], "property": the property whose value is wanted, "relation": "same" (the very same as, matching, like), "compatible" (something compatible with, goes with, pairs with), "different" (anything different from, unlike, not the same as), "at least" (at that or above, no less than), "at most" (at that or below, no more than), "within" (within some amount either way of it)}, and with "within" also "margin": that amount with its unit. When the text points at no further listing, return {"seller": ""}."""

PICK = """You pick the listing a chain link describes and read one property from it. LINK names the listing (its shop, its kind of product, and properties that pick it out from the shop's other listings). RECORDS are listings the agent has seen: OPENED means its page was read in full, ROW means only a search result row was seen. Return ONE JSON object and nothing else: {"id": the id of the OPENED record that is the listing LINK describes (its shop and kind match and none of its stated properties contradicts LINK), else "", "open": [ids of ROW records from that shop and of that kind that could be it, most likely first, at most 3], "value": that listing's value for PROPERTY as its page states it (its attribute for that property, or its own words), or "" when the page does not state it or sends it elsewhere, "next": when that listing's page sends PROPERTY on to a further listing ("For PROPERTY, look to / go by / refer to / use / take it from ...", "its PROPERTY is to be copied off / lifted from the ... listing", "PROPERTY is inherited from the ... model", "the ... item is the yardstick for PROPERTY"), that sentence copied exactly, else "", "pairs": a rule on that page saying which values of PROPERTY go together or pair with which, copied exactly, else ""}. A page's words about a different property do not send PROPERTY anywhere. When the page sends PROPERTY elsewhere, give "next" even if its own attributes list a value for PROPERTY."""

PAIRING = """You apply a pairing rule. A shopper wants a value of PROPERTY that is compatible with (goes with, pairs with) VALUE, and the store's pages state the RULES below. Return ONE JSON object and nothing else: {"value": the value the rules pair with VALUE (when a rule pairs VALUE with itself, VALUE), or "" when no rule covers VALUE}."""


NEAR = """You name the closest listings before a shopping agent gives up. The agent found no listing that surely meets every requirement and is about to end without buying, which earns nothing: the store almost always has a listing that is right, and the agent may have misread one. For each requested item, pick the OPENED listing of that kind of product that comes closest (for an item the request says must fit another, one whose codes link it to the listing you pick for that other item): in stock and inside the budget, with the fewest requirements its records clearly contradict, then the most its records confirm. CART NOW and AGENT'S PICKS show the listings the agent itself chose: keep such a listing for its item unless its records clearly contradict a requirement that another opened listing of that kind meets. Reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"items": {"item number": {"id": listing id, or "" only when no opened listing of that kind is in stock inside the budget, "others": [ids of the next closest opened listings of that kind, closest first, at most 2], "breaks": [requirements its records clearly contradict], "unknown": [requirements no record of it states]}}, "reason": "one sentence"}"""

FITREAD = """You read compatibility codes for a shopping agent. The request says one requested item must fit another, and the lines below are the listings' own sentences that carry model, part or series codes. Decide for each CANDIDATE whether it fits the DEVICE. Codes match whatever their spaces, dashes and case. A family written with x's (QZ-71xx) covers every code that begins with what comes before the x's, except the codes it lists as exceptions. A listing's other names (sold as, also known as) are its codes too. When the device lists the part series it takes, a part whose own part code begins with one of them fits it (series KM2 takes part KM2-5521). A line on another listing's page about one of these codes counts as if it stood on that listing's own page. A pair whose lines state nothing linking or excluding the two codes is unknown, never not; so is a pair whose list is left to another part's page that is not shown.
Return ONE JSON object and nothing else: {"fits": [candidate ids that fit the device], "not": [candidate ids the lines rule out], "unknown": [candidate ids the lines cannot settle], "why": {"candidate id": "the codes that decide, at most 12 words"}, "read": [part codes whose page would settle an unknown candidate, at most 2]}"""

# code shapes, by structure only: a model or part code is capitals then three or more digits, or a series then a number
# (AB-1234, AB1234, AB1-23456); a series is capitals with one to three digits (AB1); a family ends in x's (AB-12xx)
STRICT_CODE = re.compile(r"\b[A-Z]{2,5}(?:[- ]?\d{3,6}(?:-\d{2,6})?|\d{1,3}-\d{3,6})\b")
SERIES_CODE = re.compile(r"\b[A-Z]{2,5}\d{1,3}\b")
FAMILY_CODE = re.compile(r"\b[A-Z]{2,5}[- ]?\d{1,4}x{1,3}\b")


_SHORT_CODE = re.compile(r"\b([a-z]{2,6})[- ]?(\d{1,5})\b")
_NOT_PLAIN = re.compile(r"\b(?:not|no|except|other than|anything but|unlike|at least|at most|under|over|up to|below|above|between|within|same|matching)\b")
# a requirement's own wording that makes a code a limit or one of several choices (CL30 or lower, DDR4 or newer, IP67+)
_LIMIT_OR = re.compile(r"\bor\b|\+|[<>≤≥]|\b(?:lower|less|fewer|higher|greater|more|newer|older|later|earlier|faster|slower|better|max|min|maximum|minimum|least|most)\b")


# a code as a record prints it: letters, then a figure after an optional space, dash or colon (eu:28), and an optional
# short range to a second figure (size 25-36, US7-US11); a timing chain such as CL30-38-38-96 is not read as a range
_SEEN_CODE = re.compile(r"\b([a-z]{2,6})\s?[-:]?\s?(\d{1,5})\b(?:\s*(?:-|~|to)\s*(?:[a-z]{2,6}\s?[-:]?\s?)?(\d{1,5})\b(?!\s*-\s*\d))?")


def _code_clash(value: Any, text: Any) -> bool:
    """A wanted value that is one short code of 2-6 letters and a figure (cl30, ddr4) which the record never prints, while
    the record prints the same letters with other figures only (CL32, DDR5): the record contradicts the value. False for a
    longer value, an exclusion or limit, a record that prints the code (alone, in a range such as EU 39-42, or as eu:42),
    a family written with x's that covers it (LIT-68xx), or no code of those letters (race 18: a CL32 kit bought after the
    shopper held firm on cas latency cl30, twice, where declining was the right finish)."""
    want = str(value or "").strip().lower()
    if not want or len(want.split()) > 3 or _NOT_PLAIN.search(want):
        return False
    codes = _SHORT_CODE.findall(want)
    if len(codes) != 1 or "".join(codes[0]) != re.sub(r"[- ]", "", want):
        return False
    letters, figure = codes[0]
    # the record as words: escaped line breaks and tabs of a dumped record, symbols such as (TM) and (R), and unicode dashes
    low = re.sub(r"\\[ntr]", " ", str(text or ""))
    low = re.sub("[‐-―−]", "-", re.sub("[™®©]", " ", low)).lower()
    if any(figure.startswith(p) for p in re.findall(r"\b" + re.escape(letters) + r"[- ]?(\d{0,4})x{1,3}\b", low)):
        # a family written with x's covers the code; the codes it lists beside it are its exceptions, not this listing's
        return False
    seen: set[int] = set()
    for l, a, b in _SEEN_CODE.findall(low):
        if l != letters:
            continue
        lo = int(a)
        seen.add(lo)
        if b and lo < int(b) <= lo + 60:
            seen.update(range(lo, int(b) + 1))
    # a list of bare figures after a code counts each of them (DDR4/5, EU 41, 42, 43)
    for l, _, rest in re.findall(r"\b([a-z]{2,6})\s?[-:]?\s?(\d{1,5})((?:\s*[/,&]\s*\d{1,5}\b)+)", low):
        if l == letters:
            seen.update(int(x) for x in re.findall(r"\d{1,5}", rest))
    return bool(seen) and int(figure) not in seen


def _code_set(text: Any, strict: bool = False) -> set[str]:
    """The codes a text names, without spaces or dashes and in capitals: model and part codes, and unless `strict` also
    families and series (a series inside a part code is not counted apart)."""
    text = str(text)
    found = STRICT_CODE.findall(text)
    if not strict:
        rest = STRICT_CODE.sub(" ", text)
        found += FAMILY_CODE.findall(rest) + SERIES_CODE.findall(FAMILY_CODE.sub(" ", rest))
    return {re.sub(r"[\s-]", "", c).upper() for c in found}


# Each review prompt's role, read from the opening words of its prompt constant, and the quota pool whose heavy-model calls
# per task it draws on. A role added later appends its (role, prompt, pool) here and its pool to SONNET_QUOTA.
PROMPT_ROLES: list[tuple[str, str, str]] = [
    ("brief", BRIEF, "brief"), ("plan", PLAN, "plan"), ("hear", HEAR, "hear"), ("note", NOTE, "note"), ("check", CHECK, "check"),
    ("audit", AUDIT, "decline"), ("unmix", UNMIX, "unmix"), ("trail", TRAIL, "trail"), ("swap", SWAP, "swap"),
    ("shape", SHAPE, "shape"), ("link", LINK, "link"), ("pick", PICK, "pick"), ("pair", PAIRING, "pair"), ("near", NEAR, "decline")]
# heavy-model replies per task by pool (0: never the heavy model); the decline path shares one: the audit, the checks of
# carts code builds and the closest-listing review
SONNET_QUOTA = {"brief": 1, "plan": 1, "hear": 1, "check": 3, "decline": 1, "swap": 1, "gate": 1, "shape": 1, "link": 1,
                "pick": 1, "other": 1, "note": 0, "unmix": 0, "trail": 0, "pair": 0}
PROMPT_ROLES.append(("prices", PRICES, "prices"))
SONNET_QUOTA["prices"] = 0
PROMPT_ROLES.append(("fitread", FITREAD, "fitread"))
SONNET_QUOTA["fitread"] = 0
SONNET_QUOTA["cheaper"] = 1     # the cheaper swap's check of its planned cart (Kyo.pool_as), apart from the decline path
_GATE_MARK = GATE.split("STAGE", 1)[1][:40]


def _role_of(prompt: str) -> str:
    """The review role of a model prompt: the prompt constant it starts with; a check of a cart code plans is
    'check_lead', the process gate (its head is rewritten per stage) 'gate', anything else 'other'."""
    for role, head, _ in PROMPT_ROLES:
        if prompt.startswith(head[:48]):
            return "check_lead" if role == "check" and "\n\nCART (planned" in prompt else role
    return "gate" if _GATE_MARK in prompt[:300] else "other"


def _pool_of(role: str) -> str:
    """The quota pool a role draws on: a check of a cart code plans is on the decline path."""
    if role == "check_lead":
        return "decline"
    return next((pool for name, _, pool in PROMPT_ROLES if name == role), role)


def _one_edit(a: str, b: str) -> bool:
    """Whether two different strings are one changed, added or dropped character apart."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def _common_prefix(words: list[str]) -> str:
    """The longest start that every string shares."""
    if not words:
        return ""
    first, last = min(words), max(words)
    size = 0
    while size < len(first) and first[size] == last[size]:
        size += 1
    return first[:size]


def _query_form(value: Any) -> Any:
    """An argument as two spellings of one search share it: texts lower-cased with single spaces, numbers by value,
    nested lists and objects the same way (the dump that uses this sorts the keys)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return [_query_form(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _query_form(v) for k, v in value.items()}
    return str(value)


def _skips_rows(value: Any) -> bool:
    """A paging value past where results start: a number other than 0 or 1 (the first offset or page). A text token
    is only ever handed out by the store, so it is never treated as a jump."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", value):
        value = float(value)
    return isinstance(value, (int, float)) and value not in (0, 1)


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _merge_spellings(facts: dict[str, Any]) -> dict[str, Any]:
    """An attribute block with each property spelt two ways (a space and an underscore) shown once, under its first
    spelling: the longer value, then the other when the longer one does not contain it."""
    out: dict[str, Any] = {}
    first: dict[str, str] = {}
    for key, value in facts.items():
        name = _words(key)
        if name not in first:
            first[name] = key
            out[key] = value
            continue
        have = out[first[name]]
        a = have if isinstance(have, str) else ", ".join(map(str, have)) if isinstance(have, list) else str(have)
        b = value if isinstance(value, str) else ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        if a.strip().lower() == b.strip().lower():
            continue
        longer, other = (a, b) if len(a) >= len(b) else (b, a)
        if other.strip().lower() in longer.lower():
            # a long value is clipped at 120 characters in the review records: the short one goes first so it survives
            out[first[name]] = f"{other}; {longer}" if len(longer) > 120 else longer
        else:
            out[first[name]] = f"{longer}; {other}"
    return out


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
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", _words(text)))


def _fold(word: str) -> str:
    """A word's singular by its ending: 'boxes', 'benches', 'dishes', 'mattresses' lose 'es', a word in 'ss' stays, else
    a last 's' goes; then '-ve' and '-fe' read as '-f', so 'shelves' and 'shelf', 'drives' and 'drive' fold alike."""
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith(("ve", "fe")):
        return word[:-2] + "f"
    return word


def _toks(text: str, fold: bool = False) -> frozenset[str]:
    """The words of a text (runs of letters and digits), lowercased, a plural 's' cut (with `fold`, by _fold)."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    if fold:
        return frozenset(_fold(w) for w in words)
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


def _near_word(a: str, b: str) -> bool:
    """The same word, or one with a letter more or less (colour and color, flavour and flavor)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def _same_prop(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two property names (as _stem word sets) are the same up to one letter per word."""
    return bool(a) and bool(b) and all(any(_near_word(w, x) for x in b) for w in a) and all(any(_near_word(w, x) for x in a) for w in b)


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _map_spec(spec: Any) -> bool:
    """A schema for a map of listing ids to counts: an object with no fixed properties whose other values are numbers."""
    if not isinstance(spec, dict) or "object" not in _kinds(spec) or spec.get("properties"):
        return False
    extra = spec.get("additionalProperties")
    return isinstance(extra, dict) and bool({"integer", "number"} & set(_kinds(extra)))


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


def _num(value: Any) -> str:
    """A number for the walker and the checks: integers in full, other values with at most two decimals."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _unwrap(value: Any) -> tuple[Any, list[str]]:
    """A reply that carries messages beside its result: {"observation": result, "refusal": [messages]} when the store
    turns an ending down, or a two-key wrapper {any key: result, any key: [messages]} on any call (key names change from
    store to store). Recognised by shape only: one value is a list of sentences, the other is not a list. Returns the
    result and the messages, unwrapping a wrapper inside a wrapper; a plain reply comes back as it is."""
    notes: list[str] = []
    for _ in range(3):
        if not isinstance(value, dict) or len(value) != 2:
            break
        # a list of text records is a result (search rows), not messages
        said = [k for k, v in value.items() if isinstance(v, list) and v
                and all(isinstance(t, str) and len(t.split()) >= 4 and _text_record(t) is None for t in v)]
        if len(said) != 1:
            break
        inner = value[next(k for k in value if k != said[0])]
        if isinstance(inner, list):
            break
        notes += [t for t in value[said[0]] if t not in notes]
        value = inner
    return value, notes


_REC_KEY = r"[a-z][a-z0-9_]{0,39}:(?: |$)"


def _scalar(text: str) -> Any:
    """A text record's value in its plain type: a whole number as int; True, False and None in the spellings Python and
    JSON print them as their values."""
    value = text.strip()
    if re.fullmatch(r"-?(?:0|[1-9]\d{0,17})", value):
        return int(value)
    return next((literal for literal in (True, False, None) if value in (repr(literal), json.dumps(literal))), value)


def _pairs(text: str) -> dict[str, str] | None:
    """A value written as 'k=v' pairs joined by ';' as a dict (two pairs or more); a piece without '=' joins the value
    before it. None for any other value."""
    pieces = [p.strip() for p in text.split(";")]
    if len(pieces) < 2 or "=" not in pieces[0] or sum(1 for p in pieces if "=" in p) < 2:
        return None
    if len(pieces[0].partition("=")[0].strip()) > 60:
        # a value that does not open with a short 'key=' is prose or code holding '=' signs, not an attribute block
        return None
    out: dict[str, str] = {}
    last = None
    for piece in pieces:
        key, _, value = piece.partition("=")
        key = key.strip()
        if "=" in piece and key and len(key) <= 60 and key not in out and not re.search(r"[<>{}]", key):
            out[key] = value.strip()
            last = key
        elif last is not None and piece:
            out[last] += "; " + piece
    return out if len(out) >= 2 else None


def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing written as text, 'key: value' segments joined by one separator, as a dict; None when the text is not
    shaped so. The separator is the first of '|', a newline and ';' that stands right before a 'key: ' token (a short
    lower-case name of letters, digits and underscores; an empty field is 'key:' at the end) at least twice: descriptions
    hold CSS and line breaks more often than '|'. A separator may be followed by one line break before the token
    ("key: value |\nkey: value"). A separator not followed by such a token, or followed by a key the record holds
    already, stays inside the value."""
    if not isinstance(text, str) or not 20 <= len(text) <= 40000:
        return None
    body = text.strip()
    if not re.match(_REC_KEY, body):
        return None
    counts: dict[str, int] = {}
    # the separator is matched first and the blanks before it are cut off afterwards: no backtracking over long blank runs
    for found in re.finditer(r"([|;\n])[ \t]*(?:\n[ \t]*)?(?=" + _REC_KEY + ")", body):
        counts[found.group(1)] = counts.get(found.group(1), 0) + 1
    mark = next((c for c in "|\n;" if counts.get(c, 0) >= 2), None)
    if mark is None:
        return None
    pieces: list[tuple[str, str]] = []
    start, joint = 0, ""
    for found in re.finditer(re.escape(mark) + r"[ \t]*(?:\n[ \t]*)?(?=" + _REC_KEY + ")", body):
        piece = body[start:found.start()]
        kept = piece.rstrip(" \t\r\n")
        pieces.append((joint, kept))
        start, joint = found.end(), piece[len(kept):] + found.group(0)
    pieces.append((joint, body[start:]))
    out: dict[str, str] = {}
    last = ""
    for joint, piece in pieces:
        head = re.match(r"([a-z][a-z0-9_]{0,39}):(?: |$)", piece)
        if head and head.group(1) not in out:
            last = head.group(1)
            out[last] = piece[head.end():]
        elif last:
            out[last] += joint + piece
    if len(out) < 3:
        return None
    record: dict[str, Any] = {}
    for key, value in out.items():
        block = _pairs(value)
        record[key] = block if block is not None else _scalar(value)
    return record


def _read_records(value: Any, depth: int = 0) -> Any:
    """A store reply with its text-written listings read into dicts (_text_record): a list of such strings becomes a list
    of dicts, a dict whose one value is such a string becomes that dict, and such a string itself becomes a dict. A string
    field of a dict with several keys is read only when it holds five keys or more (a record beside a page's other keys)."""
    if depth > 6:
        return value
    if isinstance(value, str):
        got = _text_record(value)
        return value if got is None else got
    if isinstance(value, list):
        return [_read_records(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if len(value) == 1:
            got = _text_record(next(iter(value.values())))
            if got is not None:
                return got
        out = {}
        for k, v in value.items():
            got = _text_record(v) if isinstance(v, str) else None
            out[k] = _read_records(v, depth + 1) if isinstance(v, (list, dict)) else (got if got is not None and len(got) >= 5 else v)
        return out
    return value


def _said(notes: list[str], meaning: str = "") -> str:
    """The text line for messages a reply carried beside its result, and what the note reader made of them."""
    return (("\n[messages that came with this reply] " + " | ".join(notes)) if notes else "") + (f"\n[note read] {meaning}" if meaning else "")


def _fact_key(node: Any) -> Any:
    """The key of a record's attribute block: its first dict field with the most plain values (at least two), else None."""
    if not isinstance(node, dict):
        return None
    best, size = None, 1
    for key, value in node.items():
        if isinstance(value, dict) and len(value) >= 2:
            count = sum(1 for v in value.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
            if count > size:
                best, size = key, count
    return best


def _fact_block(node: Any) -> dict[str, Any] | None:
    key = _fact_key(node)
    if key is None:
        return None
    return {k: v for k, v in node[key].items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.fits = 0           # the number of the requested item this one has to fit (0: none)
        self.noun = ""          # the product's plain name as store titles print it (BRIEF), or ""
        self.find: dict[str, str] = {}   # must -> its value in search words (BRIEF); "" for no value to search by


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
        self.one_key = False    # one property per message (set by a crowded reply to a list)
        self.no_pack = False    # one item per message (set by a crowded reply to a message that named several items)
        self.packed: dict[str, list[int]] = {}  # a message that named several items -> those items
        self.raw_seen: list[str] = []     # the last store replies as the store wrote them, for the planner's wording
        self.mode_moves: dict[tuple[str, str], dict[str, int]] = {}   # cart tool, mode argument -> value -> +1 put / -1 took
        self.mode_probe: tuple[str, dict[str, Any], int, str] | None = None   # a cart call whose reply listed nothing
        self.line_keys: set[str] = set()        # reply keys under which this store listed the cart's lines
        self.empty_unsure = False               # the last cart reply read held an empty container under no such key
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
        self.clash_blocks = 0     # checkouts held back for a cart line whose records print a code another than a requirement's
        self.last_grown: dict[int, list[str]] = {}   # item -> the told requirements the last read added
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.few_rows: list[tuple[frozenset[str], set[str], tuple[str, ...]]] = []   # (query words, row ids, the words in order) of word searches whose hits fit one page
        self.chain_stop = 10 ** 6              # the step a chain being followed stops at (follow_chain sets it)
        self.last_hits: int | None = None      # hits of Kyo's last own search when they all fit the page shown
        self.price_done = -1                   # step of the last re-read made for a recorded price change
        self.price_wait = -1                   # under a cheapest pick: step of a price-only note not judged yet
        self.rose_at: dict[str, int] = {}      # listing id -> step a record showed its price higher than before
        self.link_reads: dict[str, Any] = {}   # chain-link text -> its LINK reading (None when it names no shop)
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
        self.chain_tried: set[str] = set()
        self.chain_paths: dict[str, list[str]] = {}
        self.chain_started = False
        self.widened = False
        self.chain_rules: list[str] = []
        self.trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.calls_made: list[tuple[str, dict[str, Any]]] = []   # store calls that got a reply (which searches started)
        self.page_params: dict[str, str | None] = {}              # a tool's paging parameter read off its schema
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.cart_rule: dict[str, str] = {}
        self.keeps: dict[str, list[Any]] = {}
        self.code_fixes = 0
        self.rescued = False
        self.t0 = time.time()
        self.soft, self.hard = SPEND_SOFT, SPEND_HARD
        self.loop_cap = max(LOOP_CALLS, min(LOOP_MAX, int(self.max_steps * 0.75)))
        self.all_calls = ALL_CALLS + (self.loop_cap - LOOP_CALLS)
        self.empty_holds = 0
        self.last_cart: tuple[str, str, int, int | None] | None = None
        self.plain: set[str] = set()            # models whose route refuses the thinking switch
        self.dead: set[str] = set()             # models the key refuses
        self.denied: dict[str, int] = {}        # key refusals by "model status"
        self.sonnet_used: dict[str, int] = {}   # heavy-model replies by quota pool
        self.spent_by: dict[str, float] = {}    # model spend by role
        self.fails: dict[str, int] = {}         # failed model calls by status
        self.cut_reply = False
        self.answered_by: str | None = None
        self.id_refused: set[str] = set()
        self.trusted = False                    # the cart code sets is a closest-listing pick: no check, one checkout
        self.near_marks: set[str] = set()       # states (cart and near_state) a closest-listing buy already ran in
        self.near_named: dict[str, list[str]] = {}   # NEAR's last ranking per item number, best first
        self.near_empty: set[int] = set()       # items NEAR found no opened in-stock listing for inside the budget
        self.near_breaks: dict[int, list[str]] = {}   # the requirements NEAR's first pick of an item clearly breaks
        self.word_mode: dict[str, Any] = {}     # the mode arguments of the walker's last search with free words and rows
        self.plain_modes: set[tuple[str, str]] = set()   # mode values the walker used for listings without free words
        self.dept_best: dict[tuple[Any, ...], tuple[bool, int]] = {}   # dept_match's best label per item, by rows seen
        self.link_searched: set[str] = set()    # chain-link searches already made
        self.decline_looked = False             # the one search with the values before a closest-listing decline was made
        self.told_opened: dict[str, int] = {}   # rows told_rows opened, by purpose (before a buy, the look before a decline)
        self.near_needs: str | None = None      # the requirements in force at the last NEAR call
        self.near_before: tuple[Any, ...] = ()  # near_state() right before the last checkout
        self.near_rounds = 0
        self.no_fit = False                     # the last closest-listing buy found an item with nothing buyable
        self.call_until = 0.0                   # a model call must return by this time (0: no such end)
        self.walk_caps: tuple[int, float] = (self.loop_cap, SPEND_HARD)
        self.unread: list[str] = []             # messages act() took off replies, not read yet
        self.answer_notes: list[str] = []       # such messages that came with the shopper's answers
        self.changes: list[dict[str, Any]] = [] # {"step", "kind": budget|drop|add|gone|price, ...} in order
        self.free_notes: list[str] = []         # note summaries no recorded change covers
        self.change_due = False                 # a change after_change has not handled yet
        self.change_from = 0                    # the first step whose changes after_change has not handled
        self.window: tuple[float, float] | None = None   # (old, new) after a budget raise
        self.rewiden = False                    # the re-armed wider search names the change
        self.betters: dict[str, list[str]] = {} # check key -> the listings a failed check named better
        self.others_shown: set[int] = set()     # change counts whose first 'any' check showed other listings
        self.swaps = 0
        self.relaxed = -1                       # the task version a loosening created (a drop, a budget that matters)
        self.cap_tool: dict[str, Any] | None = None   # the capped listing tool (PRICES); {} when the store has none
        self.order_seen: dict[str, list[int]] = {}    # search tool -> [results in ascending price order, results with 3+ prices]
        self.searches: list[tuple[str, dict[str, Any], Any]] = []   # (tool, args, plain result) of the last 12 listing calls
        self.swept: set[str] = set()            # reason|cart|version of the sweeps run
        self.sweep_calls: set[str] = set()      # listing calls made (the walker's with rows, the sweeps'), args sorted
        self.swept_words: set[str] = set()      # the worded-value searches the sweeps ran
        self.sweep_mark = 0                     # bumped when a sweep opens a listing: the cart is checked again
        self.cheaper_fits: list[str] = []       # the last cart check's cheaper fits that passed the gates
        self.final_done = False                 # final_review ran
        self.repaired: set[str] = set()         # cart keys whose repeated checkout repair_cart handled
        self.pointed: list[str] = []            # rows AUDIT named to open or a check named better
        self.fit_seen: dict[str, dict[str, Any]] = {}   # FITREAD prompt -> its checked verdict
        self.fit_pairs: dict[str, dict[str, Any]] = {}  # "device|line" -> the newest verdict on that cart pair
        self.fit_told: set[str] = set()                 # pairs whose misfit reading the walker was shown
        self.fit_calls = 0
        self.told_before: dict[int, int] = {}   # item -> told requirements before the interview
        self.find_words: dict[tuple[int, str], str] = {}   # (item, told need) -> its search words (HEAR); "": none
        self.ask_failed = False                 # the store turned a question down or no message could be built
        self.pool_as = ""                       # a quota pool a caller sets for its next reviews, in place of the prompt's
        self.store_lat = 1.0                    # the median of the last five store calls (messages aside), in seconds
        self.lat_recent: list[float] = []       # the last five store calls' durations, messages to the shopper left out
        self.light_times: list[float] = []      # the last five light-model calls' durations (seconds)

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

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

    def tail(self, changed: bool = False) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message.
        `changed`: the cart is about to change, so an approval given for the cart as it is now will be due again."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and (changed or self.gate_due("order")) else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              role: str = "walker") -> dict[str, Any] | None:
        """One model reply, trying `models` in order. The key refusing a model marks it dead: a 401 at once, a 402 or a
        403 the second time in the task (a first 402 is resent once after the shortest wait, since a hold on in-flight
        spend clears as other calls finish; a first 403 moves on to the next model). Once the light and heavy models are
        both dead nothing more is sent. Rate limits and server errors are waited on and resent; a timeout already took
        up to 50 s, so only the walker resends it, and only while more than 90 s remain before DEADLINE. A call started
        past DEADLINE gets one try on the first live model, so the endgame's store calls keep their time."""
        self.cut_reply = False
        late = time.time() - self.t0 > DEADLINE
        for model in models:
            if self.no_credit or self.calls >= self.all_calls or time.time() - self.t0 > LATE:
                return None
            if model in self.dead:
                continue
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, role)
                if message is not None:
                    return message
                if status in (401, 402, 403):
                    key = f"{model} {status}"
                    self.denied[key] = self.denied.get(key, 0) + 1
                    if status == 401 or self.denied[key] > 1:
                        self.dead.add(model)
                        self.no_credit = {LIGHT, HEAVY} <= self.dead
                        break
                    if late or status == 403 or self.waited + WAITS[0] > WAIT_TOTAL or time.time() - self.t0 + WAITS[0] > LATE:
                        break
                    self.waited += WAITS[0]
                    time.sleep(WAITS[0] * (0.75 + 0.25 * random.random()))
                    continue
                timeout = status in (None, 408)
                passing = status in (425, 429) or (isinstance(status, int) and status >= 500) or (
                    timeout and role == "walker" and DEADLINE - (time.time() - self.t0) > 90)
                if late or not passing or tries >= (1 if timeout else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                if time.time() - self.t0 + pause > LATE:
                    break
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if self.no_credit or late:
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   role: str = "walker") -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content).
        Thinking is switched off only for a short JSON answer from a model that takes the switch (THINK_OFF_OK); other
        JSON calls get room for thinking and the answer. A 400 to a call carrying the switch is sent once more without
        it, and the switch is never sent to that model again. self.cut_reply says whether the reply hit its cap."""
        self.cut_reply = False
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens}
        if not model.startswith("anthropic/"):
            body["temperature"] = 0
        elapsed = time.time() - self.t0
        # every call returns by LATE + END_CALL, so the store calls after it keep their time before HARD_END
        cap = min(50, int(LATE + END_CALL - elapsed), END_CALL if elapsed > DEADLINE else 50)
        if self.call_until:
            # a set end (a walker turn near the deadline): the call, its resends and the spare model all stop by then
            cap = min(cap, int(self.call_until - time.time()))
        _llm.timeout = max(5, cap)
        if model.startswith("anthropic/") and not tools:
            if max_tokens < 900 and model in THINK_OFF_OK and model not in self.plain:
                # a short JSON answer: a model that thinks first can spend the whole cap on thinking and return nothing
                body["reasoning"] = {"enabled": False}
                body["max_tokens"] = max(max_tokens, 1600)
            else:
                # thinking stays on: room for it and the JSON (the model does not see the cap; unused tokens cost nothing)
                body["max_tokens"] = max(max_tokens, 3000)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        called = time.time()
        result = _llm.post_verbose("/inference/chat/completions", body)
        if model == LIGHT:
            self.light_times = (self.light_times + [time.time() - called])[-5:]
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self.cost += float(cost)
                self.spent_by[role] = self.spent_by.get(role, 0.0) + float(cost)
            choices = data["choices"]
            choice = choices[0] if isinstance(choices, list) and isinstance(choices[0], dict) else {}
            self.cut_reply = choice.get("finish_reason") == "length"
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.fails["0"] = self.fails.get("0", 0) + 1
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0, "cut": self.cut_reply})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.fails[str(status)] = self.fails.get(str(status), 0) + 1
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        if status == 400 and "reasoning" in body and model not in self.plain and time.time() - self.t0 <= DEADLINE:
            self.plain.add(model)
            return self.call_model(model, messages, max_tokens, tools, role)
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """A review prompt's JSON reply, trying `models` in order. The heavy model is left out once the prompt's quota pool
        has had its heavy replies for this task (SONNET_QUOTA); a light reply cut at its cap is tried once more on the
        light model with more room before any dearer one. self.answered_by names the model whose reply is returned.
        Past DEADLINE-10 s the heavy model is left out (its end cap would time it out); past the hard spend cap only the
        light model answers. A heavy reply counts against its quota whenever it was billed, an empty one too."""
        role = self.pool_as or _role_of(prompt)
        self.answered_by = None
        late = time.time() - self.t0 > DEADLINE
        if HEAVY in models and (self.sonnet_left(role) <= 0 or time.time() - self.t0 > DEADLINE - 10):
            models = [m for m in models if m != HEAVY] or [LIGHT]
        if self.cost > self.hard:
            models = [LIGHT]
        for model in models:
            billed = self.cost
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, role=role)
            if model == HEAVY and (message is not None or self.cost > billed):
                pool = _pool_of(role)
                self.sonnet_used[pool] = self.sonnet_used.get(pool, 0) + 1
            got, answered = _json_obj((message or {}).get("content")), model
            if got is None and self.cut_reply and model != HEAVY and not late:
                message = self.think([{"role": "user", "content": prompt}], [LIGHT], 4000, role=role)
                got, answered = _json_obj((message or {}).get("content")), LIGHT
            if got is not None:
                self.answered_by = answered
                return got
            if late or self.broke or self.no_credit or self.calls >= self.all_calls:
                return None
        return None

    def sonnet_left(self, role: str) -> int:
        """Heavy-model replies the role's quota pool may still have in this task."""
        pool = _pool_of(role)
        return SONNET_QUOTA.get(pool, 1) - self.sonnet_used.get(pool, 0)

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if time.time() - self.t0 > HARD_END - 5:
            self.trace.append({"kind": "env_skip", "name": name})
            return None
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        if time.time() - self.t0 > DEADLINE and hasattr(_env, "max_retries"):
            # past DEADLINE one attempt per call: the client's own resends could take the checkout's time
            _env.max_retries = 1
        refused = False
        for _ in range(4):
            if time.time() - self.t0 > HARD_END - 5:
                self.trace.append({"kind": "env_skip", "name": name})
                return None
            _env.timeout = max(8, min(95, int(HARD_END + 8 - (time.time() - self.t0))))
            began = time.time()
            result = _env.post_verbose("/environment/call", payload)
            if name != self.roles.get("ask"):
                # the median of the last five calls: a uniformly slow store counts, one slow call does not; a message's
                # reply is written by the shopper and says nothing about the cart calls and the checkout
                self.lat_recent = (self.lat_recent + [time.time() - began])[-5:]
                self.store_lat = max(1.0, sorted(self.lat_recent)[len(self.lat_recent) // 2])
            if isinstance(result.data, dict):
                data = result.data
                inner, said = _unwrap(data.get("observation"))
                if said:
                    data = {**data, "observation": inner, "said": said}
                    self.unread += [t for t in said if t not in self.unread]
                if name != self.roles.get("ask"):
                    # listings the store wrote as 'key: value' text are read into records once, here, for every reader
                    raw = data.get("observation")
                    data = {**data, "observation": _read_records(raw)}
                    if name not in {self.roles.get(job) for job in ("cart", "cart_view", "buy", "refuse")}:
                        self.raw_seen = (self.raw_seen + [_dump(_clip(raw, 1200))[:1500]])[-RAW_SHOWN:]
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                self.calls_made.append((name, dict(args)))
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that); a reply that neither names
                    # the listing nor shows a record (an error code, "too many items") opened nothing
                    shown = _dump(seen)
                    record = isinstance(seen, dict) and len(seen) >= 3
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value])
                                           if isinstance(v, str) and v.strip() and (record or v.strip() in shown))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            # the episode counts as over only when the last failure was a 409 naming no step (set below)
            refused = False
            if status == 409:
                # a step conflict, read by shape: a body naming one step number other than the one sent names the step the
                # platform expects; one naming none refused a used step (the next is tried) or an episode that is over
                named = {int(n) for n in re.findall(r"(?<![\w.-])\d{1,6}(?![\w.])", body)} - {payload["step"]}
                if len(named) == 1:
                    self.step = named.pop() - 1
                    payload["step"] = self.step + 1
                    continue
                if not named:
                    refused = True
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        if refused:
            # no step was accepted: the episode is over (finished, quarantined or sealed)
            self.finished = True
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
        named = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        if not self.no_credit and (not _items(brief.get("items")) or not (isinstance(named.get("buy"), str) and named["buy"] in self.schemas)):
            # no item, or no buy role naming one of the store's tools (a list or a made-up name is dropped below)
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
        for key, role in (("buy_args", "buy"), ("refuse_args", "refuse")):
            # a key the tool does not define, or a value its schema refuses, would make every forced ending fail its shape
            props = (self.schemas.get(self.roles.get(role) or "") or {}).get("properties") or {}
            kept = {}
            for name, value in brief[key].items():
                faults: list[str] = []
                value = _shape(value, props.get(name), name, faults) if name in props else None
                if name in props and not faults:
                    kept[name] = value
            brief[key] = kept
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
            self.items[-1].fits = _count(entry.get("fits")) or 0
            noun = entry.get("noun")
            self.items[-1].noun = noun.strip() if isinstance(noun, str) and 0 < len(noun.split()) <= 3 else ""
            finds = entry.get("find")
            if isinstance(finds, list) and len(finds) == len(musts):
                # one entry per must or none at all: a misaligned list could pair a value with the wrong must
                self.items[-1].find = {m: (f.strip() if isinstance(f, str) else "") for m, f in zip(musts, finds)}
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for number, item in enumerate(self.items, 1):
            if not 1 <= item.fits <= len(self.items) or item.fits == number:
                item.fits = 0
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
            # the messages the steps at the order take: Kyo buys the closest listings rather than end without a purchase,
            # so a message kept for an ending step would be a question lost
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
            got = self.ask_json(prompt, [LIGHT], 500) or {}
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
            if item.fits:
                text += f"; must fit item {item.fits} (judged on both pages by a compatible list, the size or model named, or codes)"
            # a property told or settled as a requirement is not also one the shopper does not mind
            free = [p for p in item.free if not self.settled_for(number, p)]
            if free:
                text += "; shopper does not mind: " + ", ".join(free)
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
        elif self.cart_short() and self.running_out():
            text += " Time or model budget is nearly used up: cart the closest opened listing of each item now and check out."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        gaps = self.missing_items(pacing=True) if self.buys == 0 and left > self.tail() + 1 else []
        room = left - self.tail()
        if gaps and room <= 4 * len(gaps) + 2:
            text += (f" Item(s) {', '.join(map(str, gaps))} have no cart line yet, with about {max(1, room // len(gaps))} steps left for each:"
                     " cart the best opened listing for each now, then check out.")
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.changes:
            text += "\n[changes so far, already in the ledger] " + " | ".join(self.change_lines()[-4:] + self.free_notes[-2:])
        elif self.notes:
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
                    price = self.price_of(ident)
                    self.rows[ident].update(node)
                    if price is not None and (self.price_of(ident) or 0.0) > price:
                        self.rose_at[ident] = self.step
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
                if not name:
                    continue
                text = str(value).strip().lower()
                if name in done:
                    # the same property spelt another way (a space and an underscore): its value joins the shown values, so
                    # the table shows both spellings' values (the length tests still read the first spelling)
                    slot = group["keys"][name]
                    if text and text not in slot["values"]:
                        slot["values"].append(text)
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
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
                if mean > 60 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                coded = sum(1 for v in values if _id_like(v)) * 2 >= len(values)
                wordy = sum(1 for v in values if len(v.split()) > 5) * 2 >= len(values)
                if len(values) >= 4 and len(set(values)) == len(values) and (coded or wordy):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:20]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def shelf_reads(self) -> list[tuple[int, list[str]]]:
        """Per item: (listings of its own shelf read in full, rows of that shelf not read yet)."""
        cat_key = self.category_key()
        out = []
        for item in self.items:
            shelf, name = _words(item.shelf or ""), _words(item.name)
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            mine = [i for i in self.seen_order if cat_key and (lambda label: _words(label) in (shelf, name) or bool({w for w in _stem(label) if len(w) >= 4} & long))(str(self.rows[i].get(cat_key) or ""))]
            out.append((sum(1 for i in mine if i in self.full), [i for i in mine if i not in self.full and self.buyable(i)]))
        return out

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

    def page_param(self, name: str) -> str | None:
        """The parameter that asks a search tool for a later part of its results: from the tool's result shape, else from
        its schema; None when the schema requires it."""
        arg = (self.shapes.get(name) or {}).get("next_arg") or self.page_params.get(name)
        if not arg or arg in ((self.schemas.get(name) or {}).get("required") or []):
            return None
        return arg

    def read_page_param(self, name: str, args: dict[str, Any]) -> None:
        """A search sent with an optional number past where results start, before its paging parameter is known (an
        offset past the matches comes back with no rows to learn the shape from): one light reading of the schema."""
        if self.page_param(name) or name in self.page_params:
            return
        required = (self.schemas.get(name) or {}).get("required") or []
        if not any(_skips_rows(v) for k, v in args.items() if k not in required):
            return
        self.page_params[name] = None
        props = (self.schemas.get(name) or {}).get("properties") or {}
        about = next((t["function"].get("description", "") for t in self.tools if t["function"].get("name") == name), "")
        got = self.ask_json(PAGE_ARG + "\n\nTOOL: " + name + " | " + str(about)[:600] + " | parameters: " + _dump(self.schemas.get(name) or {})[:3000],
                            [LIGHT, HEAVY], 400) or {}
        if isinstance(got.get("page"), str) and got["page"] in props:
            self.page_params[name] = got["page"]
        self.trace.append({"kind": "page_param", "tool": name, "param": self.page_params[name]})

    def search_key(self, name: str, args: dict[str, Any], param: str) -> str:
        """What a search asks for, its paging value left out, in the form two spellings of one search share."""
        return name + json.dumps({k: _query_form(v) for k, v in args.items() if k != param}, sort_keys=True, ensure_ascii=False, default=str)

    def skipped_start(self, name: str, args: dict[str, Any], upto: int | None = None) -> str | None:
        """The paging parameter of a search's FIRST call that asks for a later part of the results (race 17: the walker
        sent a made-up offset on new searches and never saw the rows before it); None for a search already started, a
        call that is nothing but the paging value, or one at its start. `upto` counts only the calls made before it."""
        param = self.page_param(name)
        if not param or not _skips_rows(args.get(param)) or not any(k != param for k in args):
            return None
        key = self.search_key(name, args, param)
        made = self.calls_made if upto is None else self.calls_made[:upto]
        if any(n == name and self.search_key(n, a, param) == key for n, a in made):
            return None
        return param

    def read_start(self, name: str, args: dict[str, Any], upto: int) -> str:
        """A search that skipped its start on its first call, its paging parameter learned only from the reply: Kyo
        reads the start too (one step), shows it, counts its rows for the cheapest-first note and caches it."""
        param = self.skipped_start(name, args, upto)
        if not param or self.finished or self.left() - self.tail() < 6:
            return ""
        plain = {k: v for k, v in args.items() if k != param}
        reply = self.act(name, plain)
        if reply is None or self.finished:
            return ""
        seen = reply.get("observation")
        self.observe(name, seen)
        self.paged_obs.append(seen)
        shown = self.show(seen)
        self.read_cache[name + _dump(plain)] = (self.version, self.step, shown)
        self.trace.append({"kind": "first_page", "tool": name, "skipped": {param: args.get(param)}})
        return (f"\n[Kyo] That search asked for a later page ({param}={args.get(param)}) on its first call, which leaves the rows before"
                f" it unseen; Kyo read its first page too (one step):\n" + shown)

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
        if isinstance(value, int) and (count < len(rows) or value >= count):
            # the count must cover the rows shown and the next value lie below it: a reading the other way round is
            # swapped for good, one that fits neither way is not used
            if not (value >= len(rows) and count < value):
                return None
            shape["count"], shape["next"] = shape["next"], shape["count"]
            self.trace.append({"kind": "shape_swap", "tool": name})
            count, value = value, count
        elif count < len(rows):
            return None
        return count, len(rows), shape["next_arg"], value

    def one_page(self, name: str, observation: Any) -> int | None:
        """The hit count of a search result whose hits all fit the page shown, else None: by the tool's result shape when
        it was read already (no model call here), else by the reply's own whole numbers (every one at most the rows shown,
        one of them equal to it)."""
        if not isinstance(observation, dict):
            return None
        rows = self.records_in(observation)
        if not rows:
            return None
        if self.shapes.get(name):
            plan = self.paging(name, observation)
            if plan:
                return plan[0] if plan[0] <= plan[1] else None
        numbers = [v for v in observation.values() if isinstance(v, int) and not isinstance(v, bool)]
        return len(rows) if numbers and max(numbers) <= len(rows) and len(rows) in numbers else None

    def note_hits(self, name: str, args: dict[str, Any], observation: Any) -> int | None:
        """A word search whose hits all fit the page shown: its words and row ids are kept. The store's word search also
        matches a listing's page facts, so each such row carries every query word somewhere in its records. Returns the
        hit count, or None."""
        if name != self.roles.get("search") or not isinstance(args, dict):
            return None
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        words = args.get(target) if target else None
        if not isinstance(words, str) or not re.search(r"[A-Za-z]", words) or any(isinstance(args.get(k), str) and args.get(k).strip() for k in texts if k != target):
            return None
        observation = _unwrap(observation)[0]
        hits = self.one_page(name, observation)
        if hits is None:
            return None
        order = tuple(_fold(w) for w in re.findall(r"[a-z0-9]+", words.lower()))
        self.few_rows = (self.few_rows + [(_toks(words, True), {i for i, _ in self.records_in(observation)}, order)])[-24:]
        return hits

    def query_shows(self, ident: str, words: frozenset[str], lead: bool = False) -> bool:
        """Every word of a told value sits in the query of a one-page word search that returned this row (`lead`: the
        value's words are the query's first words; the store drops trailing words first)."""
        return bool(words) and any(words <= (frozenset(order[: len(words)]) if lead else query) and ident in ids
                                   for query, ids, order in self.few_rows)

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
        if pages > 3 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
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
            self.sweep_calls.add(name + json.dumps(call, sort_keys=True, default=str))
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
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({_num(p)})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < (8 if self.rewiden else 14) + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {_num(self.price_of(i))}" for i in self.cart_ids() if self.price_of(i) is not None]
        if self.rewiden:
            self.rewiden = False
            return ("Not run yet (no step used). The task changed (" + "; ".join(self.change_lines()[-2:]) + "), and the pick is the cheapest:"
                    " listings turned down before may fit now. Search each item's department again for buyable rows priced below the cart's line"
                    + (f" ({'; '.join(lines)})" if lines else "") + ", read the cheapest rows that may fit, swap in any that fits, then check out.")
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
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or time.time() - self.t0 > OPEN_END:
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

    def was_opened(self, ident: str) -> bool:
        """The listing's page or attributes were read (a page with too few attributes for a fact block counts too)."""
        return ident in self.opened or ident in self.full

    def label_of(self, ident: str, cat_key: str | None) -> str:
        """The listing's department label, lower-cased; "" without one."""
        return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""

    def worded_values(self, number: int) -> list[str]:
        """The item's requirement values in force that are a few plain words (no figure, no pointer to another listing)."""
        refs = {need for _, need in self.refs}
        values = []
        for n, need in self.needs_now():
            value = need.split(":", 1)[1].strip() if ":" in need else ""
            if n == number and need not in refs and value and len(value.split()) <= 5 and not re.search(r"\d", value):
                values.append(value)
        return values

    def screen(self, ident: str, values: list[str]) -> int:
        """How many of the item's worded values a listing's record states somewhere (title, description, attributes)."""
        text = _toks(_dump(self.full.get(ident) or self.rows.get(ident) or {})[:3000])
        return sum(1 for v in values if _toks(v) and _toks(v) <= text)

    def price_cap_tool(self) -> dict[str, Any] | None:
        """Under a cheapest pick: the store tool that lists listings priced at or below a highest price, with its
        parameters, named once per episode by the light model from the tool schemas (PRICES) and checked against them:
        the cap is a numeric parameter of that tool and not the one that pages its results. None when there is none
        (self.cap_tool is {} then), and while it is not known (too late to ask, or no spend left)."""
        if self.cap_tool is not None:
            return self.cap_tool or None
        # too late for a sweep to follow (sweeps start before DEADLINE-45 s), or no spend left: not asked now
        if self.brief.get("pick") != "cheapest" or time.time() - self.t0 > DEADLINE - 60 or self.cost > self.hard:
            return None
        self.cap_tool = {}
        skip = {self.roles.get(r) for r in ("cart", "cart_view", "buy", "refuse", "ask")} - {None}
        lines = [f"- {t['function']['name']}: {str(t['function'].get('description') or '')[:300]} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in skip]
        got = self.ask_json(PRICES + "\n\nTOOLS:\n" + "\n".join(lines)[:6000], [LIGHT], 400) or {}
        tool = got.get("tool") if isinstance(got.get("tool"), str) and got.get("tool") not in skip else ""
        props = (self.schemas.get(tool) or {}).get("properties") if tool else None
        props = props if isinstance(props, dict) else {}
        cap = got.get("cap") if isinstance(got.get("cap"), str) else ""
        spec = props.get(cap)
        paging = (self.shapes.get(tool) or {}).get("next_arg")
        if not isinstance(spec, dict) or not {"integer", "number"} & set(_kinds(spec)) or cap == paging:
            self.trace.append({"kind": "cap_tool", "got": _clip(got, 80), "ok": False})
            return None
        found: dict[str, Any] = {"tool": tool, "cap": cap, "integer": "number" not in _kinds(spec)}
        for part in ("shelf", "words"):
            arg = got.get(part)
            if isinstance(arg, str) and arg != cap and isinstance(props.get(arg), dict) and "string" in _kinds(props[arg]):
                found[part] = arg
        self.cap_tool = found
        self.trace.append({"kind": "cap_tool", "got": found, "ok": True, "t": round(time.time() - self.t0)})
        return found

    def repeat(self, tool: str, args: dict[str, Any]) -> bool:
        """The same listing call (any argument order) was made already in this episode, by a sweep or by the walker: a
        change of the task does not change what the store lists."""
        return tool + json.dumps(args, sort_keys=True, default=str) in self.sweep_calls

    def sweep_call(self, tool: str, args: dict[str, Any]) -> tuple[Any, int]:
        """One listing call of a sweep. Returns (its plain result or None, steps used)."""
        self.sweep_calls.add(tool + json.dumps(args, sort_keys=True, default=str))
        reply = self.act(tool, args)
        if reply is None:
            return None, 0
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_view(tool, seen)
        seen = _unwrap(seen)[0]
        if self.records_in(seen):
            self.searches = (self.searches + [(tool, dict(args), seen)])[-12:]
        return seen, 1

    def sweep_list(self, number: int, shelf: str, anchor: float, after: bool, room: int) -> tuple[int, str]:
        """List the department `shelf` below `anchor` with the store's own tools in at most 2 of the sweep's `room` steps
        (1 after a change with room 2 or less, so the rest can open a listing): the capped tool
        (its call built from the walker's last call to it; after a change a second call goes below the cheapest row
        the first showed), else the next page of the search that best carries the item's worded values, else one
        search by those values. Returns (steps used, what was done)."""
        used, said = 0, []
        one = after and room <= 2
        room = min(room, 1 if one else 2)
        cap = self.price_cap_tool() if room >= 1 else None
        if cap:
            tool, param = cap["tool"], cap["cap"]
            schema = self.schemas.get(tool) or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            # the walker's last call to this tool gives the modes and other arguments the store needs; its free words,
            # paging value and price are its own (often another item's) and are left out
            last = next((dict(a) for t, a, _ in reversed(self.searches) if t == tool), {})
            base = {k: v for k, v in last.items() if k not in ((self.shapes.get(tool) or {}).get("next_arg"), param, cap.get("words"))
                    and (k in required or isinstance((props.get(k) or {}).get("enum"), list))}
            words = " ".join(self.worded_values(number)[:2] + [self.items[number - 1].name])
            for key, value in list(base.items()):
                if isinstance(value, str) and not isinstance((props.get(key) or {}).get("enum"), list):
                    base[key] = words
            limit = anchor - (1 if cap["integer"] else 0.01)
            for _ in range(2 if after and not one else 1):
                if used >= room or limit <= 0 or self.finished or not self.cap_tool:
                    break
                args: dict[str, Any] = {**base, param: int(limit) if cap["integer"] else round(limit, 2)}
                if cap.get("shelf"):
                    args[cap["shelf"]] = shelf
                if cap.get("words") and (not cap.get("shelf") or cap["words"] in required):
                    args[cap["words"]] = words
                for key in schema.get("required") or []:
                    spec = props.get(key)
                    if key not in args and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                        args[key] = self.mode_for(tool, key)
                faults: list[str] = []
                args = _shape(args, schema, "args", faults)
                if faults:
                    break
                if self.repeat(tool, args) or time.time() - self.t0 > DEADLINE - 30:
                    return used, "; ".join(said)
                cat_key = self.category_key()
                known = any(self.label_of(i, cat_key) == shelf.lower() and (self.price_of(i) if self.price_of(i) is not None else limit + 1) <= limit
                            for i in self.seen_order)
                seen, sent = self.sweep_call(tool, args)
                used += sent
                if seen is None:
                    break
                said.append(f"listed {shelf} at or below {_num(limit)}")
                rows = self.records_in(seen)
                prices = [p for p in (self.price_of(i) for i, _ in rows) if p is not None]
                plan = self.paging(tool, seen)
                if any(p > limit for p in prices) or (self.shapes.get(tool) or {}).get("next_arg") == param or (not rows and known):
                    # not a highest price after all, or a tool that does not list what is known to be there
                    self.cap_tool = {}
                    self.trace.append({"kind": "cap_off", "tool": tool, "rows": len(rows)})
                    break
                if plan and plan[3] is not None and plan[2] != param and plan[0] <= 2 * plan[1] and used < room and not one:
                    # a short rest: its next page too
                    more = _shape({**args, plan[2]: plan[3]}, schema, "args", faults)
                    if not faults and not self.repeat(tool, more):
                        page, sent = self.sweep_call(tool, more)
                        used += sent
                        if page is not None:
                            said.append("and its next page")
                    break
                if not after or not prices or not plan or plan[0] <= plan[1]:
                    break
                limit = min(prices) - (1 if cap["integer"] else 0.01)
            if used:
                return used, "; ".join(said)
        values = self.worded_values(number)
        tool = self.roles.get("search") or ""
        tally = self.order_seen.get(tool, [0, 0])
        if room >= 1 and not (tally[1] >= 2 and tally[0] == tally[1]):
            # a store that lists rows in no price order: the next page of the search that carries the item's values best
            # (a search listing the line's department first); then, before any checkout, its further pages while each
            # shows a cheaper row of the department and steps allow, keeping the finish, one spare, a step per item and
            # the sweep's openings (after a turned-down checkout the rescue reads the newest rows: no deep paging then)
            want = frozenset().union(*[_stem(v) for v in values]) if values else frozenset()
            best = None
            for name, args, seen in reversed(self.searches):
                plan = self.paging(name, seen)
                if not plan or plan[3] is None:
                    continue
                call = {**args, plan[2]: plan[3]}
                if self.repeat(name, call):
                    continue
                lists = any(isinstance(v, str) and v.strip().lower() == shelf.strip().lower() for v in args.values())
                score = (lists, len(want & _stem(" ".join(str(v) for v in args.values() if isinstance(v, str)))))
                if best is None or score > best[0]:
                    best = (score, name, call, args)
            if best:
                faults = []
                call = _shape(best[2], self.schemas.get(best[1]), "args", faults)
                if not faults:
                    seen, sent = self.sweep_call(best[1], call)
                    if sent:
                        pages = 1
                        while (best[0][0] and pages < 6 and not self.buys and any(self.label_of(i, self.category_key()) == shelf.strip().lower()
                                                                and (self.price_of(i) or anchor) < anchor for i, _ in self.records_in(seen)) and not self.finished and time.time() - self.t0 < DEADLINE - 45
                               and self.left() - self.tail() - 1 - len(self.items) - SWEEP_OPENS > 0):
                            plan = self.paging(best[1], seen)
                            if not plan or plan[3] is None:
                                break
                            more = _shape({**call, plan[2]: plan[3]}, self.schemas.get(best[1]), "args", faults)
                            if faults or self.repeat(best[1], more):
                                break
                            seen, sent = self.sweep_call(best[1], more)
                            if not sent or seen is None:
                                break
                            call, pages, used = more, pages + 1, used + 1
                            if not any(self.label_of(i, self.category_key()) == shelf.strip().lower() and (self.price_of(i) or anchor) < anchor
                                       for i, _ in self.records_in(seen)):
                                break           # a page with no cheaper row of the department: deeper pages seldom have one
                        return used + 1, f"read {'the next page' if pages == 1 else f'{pages} more pages'} of {_dump(best[3])[:80]}"
        words = " ".join(values[:2] + [self.items[number - 1].name])
        walked = any(words.strip().lower() == v.strip().lower() for _, a, _ in self.searches for v in a.values() if isinstance(v, str))
        if room >= 1 and tool and values and not walked and words not in self.swept_words:
            # a store that lists rows cheapest first shows the cheapest of the kind on this first page
            self.swept_words.add(words)
            before = self.step
            found = self.chain_search(words)
            if self.step > before:
                self.searches = (self.searches + [(tool, dict(getattr(self, "last_search", None) or {}), getattr(self, "last_seen", None))])[-12:]
                return used + self.step - before, f"searched {words!r}" + ("" if found else " (no rows)")
        return used, "; ".join(said)

    def sweep_ready(self, reason: str = "order") -> bool:
        """Whether price_sweep(reason) would run now (no side effects): a cheapest pick before DEADLINE-45 s, at least 3
        steps free beyond the finish and one per item, no sweep yet for this cart and task version, and for "change" a
        loosening made this task version."""
        if self.brief.get("pick") != "cheapest" or self.finished or time.time() - self.t0 > DEADLINE - 45:
            return False
        if reason == "change" and self.relaxed != self.version:
            return False
        free = self.left() - self.tail() - len(self.items)
        return free >= 3 and f"{reason}|{self.cart_mark()}#{self.version}" not in self.swept

    def price_sweep(self, reason: str) -> str:
        """Under a cheapest pick, before the order ("order") and right after a loosening ("change"): Kyo lists each cart
        line's department below the line's price with the store's own tools and opens the cheapest unread candidates
        (those stating the item's worded values first), so the cart check can judge them. An item with no line yet takes
        the cheapest buyable listing of its department seen. Returns a line for the walker, or ""."""
        if not self.sweep_ready(reason):
            return ""
        self.swept.add(f"{reason}|{self.cart_mark()}#{self.version}")
        cat_key = self.category_key()
        if not cat_key:
            return ""
        after = reason == "change"
        free = self.left() - self.tail() - len(self.items)
        room = free - (2 if after else SWEEP_ROOM)
        lines = self.item_lines()
        cart = set(self.cart_ids())
        held = [i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
        if len(self.items) == 1 and not lines and len(held) == 1:
            lines = {1: held[0]}                # the only line of a one-item request, whatever its label
        done: list[str] = []
        opened: list[str] = []
        for number, item in enumerate(self.items, 1):
            ident = lines.get(number)
            if not ident:
                pool = [i for i in self.seen_order if self.buyable(i) and self.price_of(i) is not None and self.dept_match(number, i, cat_key)]
                ident = min(pool, key=lambda i: self.price_of(i) or 0.0) if pool else None
            anchor = self.price_of(ident) if ident else None
            shelf = str((self.rows.get(ident) or {}).get(cat_key) or "").strip() if ident else ""
            if self.finished or time.time() - self.t0 > DEADLINE - 45:
                break
            if anchor is None or not shelf:
                continue
            used, said = self.sweep_list(number, shelf, anchor, after, room) if room >= 1 else (0, "")
            room -= used
            free -= used
            values = self.worded_values(number)
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            cands = [i for i in self.seen_order if i != ident and i not in cart and self.buyable(i) and self.label_of(i, cat_key) == shelf.lower()
                     and self.price_of(i) is not None and (self.price_of(i) or 0.0) < anchor
                     and (budget is None or (self.price_of(i) or 0.0) * (1 if self.brief.get("budget_scope") == "each" else item.qty) <= budget)]
            cands.sort(key=lambda i: (-self.screen(i, values), self.price_of(i) or 0.0))
            opens = max(0, min(SWEEP_OPENS, free - (1 if after else 3)))
            read = self.fill_facts([i for i in cands if not self.was_opened(i)][:opens], self.tail() + len(self.items) + 1) if opens else []
            free -= len(read)
            got = [t.split(":", 1)[0] for t in read]
            opened += got
            shown = ", ".join(f"{i} ({_num(self.price_of(i))})" for i in sorted(cands, key=lambda i: self.price_of(i) or 0.0)[:6])
            done.append(f"item {number} ({shelf}, line {ident} at {_num(anchor)}): {said or 'the listings seen already'}; cheaper of its department: {shown or 'none'}"
                        + (f"; opened {', '.join(got)}" if got else ""))
            self.trace.append({"kind": "sweep", "reason": reason, "item": number, "anchor": anchor, "shelf": shelf, "used": used,
                               "cands": cands[:8], "opened": got})
        if opened:
            self.sweep_mark += 1
        return ("\n[Kyo looked for cheaper listings of each item's department (one step a call)] " + " | ".join(done)) if done else ""

    def cheaper_block(self, ids: list[str]) -> tuple[str, set[str]]:
        """For a cheapest pick's cart check: per cart line, the buyable listings of its department priced below it (opened
        ones cheapest first, at most 8, then search rows, at most 6) and up to 5 opened ones at or above its price.
        Returns (the text, the ids it showed as opened and cheaper)."""
        cat_key = self.category_key()
        if not cat_key:
            return "", set()
        parts: list[str] = []
        shown: set[str] = set()
        for line in dict.fromkeys(ids):
            price, kind = self.price_of(line), self.label_of(line, cat_key)
            if price is None or not kind:
                continue
            same = sorted((i for i in self.seen_order if i not in ids and self.buyable(i) and self.label_of(i, cat_key) == kind
                           and self.price_of(i) is not None), key=lambda i: self.price_of(i) or 0.0)
            below = [i for i in same if (self.price_of(i) or 0.0) < price]
            opened = [i for i in below if self.was_opened(i)][:8]
            rows = [i for i in below if not self.was_opened(i)][:6]
            above = [i for i in same if (self.price_of(i) or 0.0) >= price and self.was_opened(i)][:5]
            if not (opened or rows or above):
                continue
            shown.update(opened)
            parts.append(f"cart line {line} ({_num(price)}):\n" + "\n".join(
                ["OPENED CHEAPER " + self.review_record(i, 1200, 200) for i in opened]
                + ["ROW CHEAPER " + _dump(_clip(self.rows.get(i), 120))[:500] for i in rows]
                + ["OPENED NOT CHEAPER " + self.review_record(i, 1200, 200) for i in above]))
        if not parts:
            return "", set()
        return ("\n\nLISTINGS OF EACH CART LINE'S DEPARTMENT (in stock; OPENED = read in full, ROW = a search row only; CHEAPER = priced"
                " below that cart line, NOT CHEAPER = priced at or above it):\n" + "\n".join(parts)
                + '\nThe pick is the cheapest: judge each OPENED CHEAPER listing as if it were that cart line, by the reading rules above.'
                ' Add the key "cheaper_fits" to your JSON object: the ids of those that meet every requirement of that line\'s item,'
                ' cheapest first ([] when none); when one does, the cart is wrong. Name an OPENED NOT CHEAPER listing in "better" only'
                " when the cart line itself fails a requirement and that listing meets every one."), shown

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it, with the messages its reply carried; returns what
        the note changed."""
        observation, said = _unwrap(observation)
        self.absorb(observation)
        lines = [line for line in dict.fromkeys(self.take_unread(said) + self.loose_lines(observation)) if line not in self.heard]
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
        if self.left() <= 8 or time.time() - self.t0 > DEADLINE - 60:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1 and all(n >= 2 for n, _ in self.shelf_reads()):
            return True
        return self.step >= max(10, min(self.max_steps // 3, ASK_BY))

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
        """Each item's keys are ranked once (ask_orders: the store's order, the planner's skips and hints); each message
        asks one item about its likeliest open keys (next_ask), and each answer is read before the next message: the
        store's key names mark the answers that state a need (the heavy model reads those once, after the last message),
        the light model reads the others at once. Code then steers the rest: an item whose need came drops back; a reply
        that only did not follow marks nothing free, lowers its listed keys like a no-preference answer, and puts the
        item behind the others (twice puzzled, or once past PUZZLE_AFTER messages, it gets no more); a crowded reply to several keys switches to one key per message and
        re-asks that item's best key alone (reroute); a clear answer with no need lowers each listed key by how surely its
        place in the list is weighed (weigh), an answer with a need the keys before the one it answered. A message to an
        item whose need came closes every key it names. Past ASK_UNTIL the remaining questions go out in one burst and
        everything unread is read once. When more items wait for their first question than messages are left, one
        message names several of them, each with its leading keys (next_pack); its answer is read for whichever item it
        names, and a puzzled reply puts all of them behind the others."""
        self.asked = True
        if time.time() - self.t0 > DEADLINE - 60:
            # too late for questions whose answers can still be read and acted on
            return ""
        self.told_before = {n: len(item.told) for n, item in enumerate(self.items, 1)}
        dropped, start = sum(len(it.dropped) for it in self.items), len(self.changes)
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        thin = [ids[: 2 - n] for n, ids in self.shelf_reads() if n < 2 and ids]
        if thin:
            self.fill_facts([i for ids in thin for i in ids], count + 8)
        table = self.key_table()
        orders = self.ask_orders(table, count)
        stated: set[int] = set()
        closed: set[tuple[int, str]] = set()
        front: list[tuple[int, list[str]]] = []
        rounds: dict[int, int] = {}
        puzzled: dict[int, int] = {}                 # item -> answers that only said the shopper did not follow
        weight: dict[tuple[int, str], float] = {}    # (item, key) -> how far the key is still unweighed (1 = never asked)
        sent: list[tuple[int, str, str]] = []
        later: list[tuple[int, str, str]] = []
        told: list[str] = []
        took = 0.0
        slow = False                                 # a read between messages was slow: the rest wait for the last read
        named_any = False                            # an answer named a listed key: the last read is the heavy model's
        while not (self.finished or self.ask_failed) and self.msgs_free() > 0:
            burst = time.time() - self.t0 >= ASK_UNTIL
            pack = self.next_pack(orders, stated, closed, front, rounds, puzzled, weight, len(sent))
            if pack:
                # more items wait for a first question than messages are left: one message names several of them
                if burst and later and time.time() - self.t0 + took > LATE - 5:
                    self.take_heard(self.hear(later, (HEAVY, LIGHT)), told)
                    later = []
                for number, labels in pack:
                    rounds[number] = rounds.get(number, 0) + 1
                    if burst:
                        closed.add((number, _words(labels[0])))
                text = self.pack_question(pack)
                self.packed[text] = [number for number, _ in pack]
                begun = time.time()
                got = self.send([(pack[0][0], "; ".join(", ".join(labels) for _, labels in pack), text)])
                took = time.time() - begun
                if not got:
                    break
                entry = got[0]
                sent.append(entry)
                every = [label for _, labels in pack for label in labels]
                named = self.named_keys(entry[2], every, table)
                hit = {number: [labels.index(label) for label in named if label in labels] for number, labels in pack}
                if any(hit.values()) and not (len(every) >= 3 and len(named) >= 3):
                    # an answer naming a listed key of one of the items: that item's need is taken as come (the heavy
                    # model reads it below); its keys before the one answered were weighed, the others' first keys too
                    for number, labels in pack:
                        closed.update((number, _words(label)) for label in named if label in labels)
                        if hit[number]:
                            stated.add(number)
                            self.weigh(number, labels[:min(hit[number])], weight)
                        else:
                            self.weigh(number, labels[:1], weight)
                    later.append(entry)
                    named_any = True
                    continue
                if burst or self.msgs_free() <= 0 or (len(sent) > 1 and (
                        slow or self.light_lat() > READ_SLOW or time.time() - self.t0 + self.light_lat() > ASK_UNTIL - READ_SLACK)):
                    # no message follows, or a read now would come too late to steer one: the answer waits for the last read
                    for number, labels in pack:
                        self.weigh(number, labels[:1], weight)
                    later.append(entry)
                    continue
                reading = time.time()
                outer = self.call_until
                if len(sent) > 1:
                    self.call_until = min(outer or float("inf"), reading + READ_SLOW + 3)
                try:
                    heard = self.hear([entry], (LIGHT,), 800)
                    if not heard and not self.finished and time.time() - self.t0 < LATE - 10 and (len(sent) == 1 or time.time() - reading <= READ_SLOW):
                        heard = self.hear([entry], (LIGHT,), 800)
                finally:
                    self.call_until = outer
                slow = time.time() - reading > READ_SLOW
                if not heard:
                    for number, labels in pack:
                        self.weigh(number, labels[:1], weight)
                    later.append(entry)
                    continue
                # a 'does not mind' to several items does not show which properties the shopper weighed
                heard = {k: v for k, v in heard.items() if k != "free"}
                lost = 1 in {_count(x) for x in _items(heard.get("lost"))}
                if 1 in {_count(x) for x in _items(heard.get("crowded"))} - ({1} if lost else set()):
                    # the shopper wants one thing at a time: the rest go one item and one property a message
                    self.no_pack = True
                    self.one_key = True
                grew = self.grown(heard, told)
                stated |= grew
                for number, labels in pack:
                    if number in grew:
                        # its keys before the one the new need answered were weighed (as when the answer names a key)
                        at = self.answered_at(number, labels, closed)
                        self.weigh(number, labels[:at] if at is not None else labels[:1], weight)
                        continue
                    if lost:
                        # race 17 answered a message naming several items with a puzzled reply whenever it named none
                        # of their needs' properties (90 of 91): its keys count as weighed and the items wait behind
                        puzzled[number] = puzzled.get(number, 0) + 1
                    self.weigh(number, labels, weight)
                continue
            pick = self.next_ask(orders, stated, closed, front, rounds, puzzled, weight, len(sent))
            if pick is None:
                break
            number, labels = pick
            if burst and later and time.time() - self.t0 + took > LATE - 5:
                # the next answer would come after the last read could start: read the waiting ones first
                self.take_heard(self.hear(later, (HEAVY, LIGHT)), told)
                later = []
            if number in stated:
                # an item whose need came already: every key the message names is closed (the others go first)
                closed.update((number, _words(label)) for label in labels)
            elif burst:
                # no reading comes before the next message: the first key counts as weighed
                closed.add((number, _words(labels[0])))
            rounds[number] = rounds.get(number, 0) + 1
            begun = time.time()
            got = self.send([(number, ", ".join(labels), self.question(number, labels))])
            took = time.time() - begun
            if not got:
                break
            entry = got[0]
            sent.append(entry)
            named = self.named_keys(entry[2], labels, table)
            closed.update((number, _words(label)) for label in named if label in labels)
            if named and not (len(labels) >= 3 and len(named) >= 3):
                # an answer naming a listed key: its item's need is taken as come (the heavy model reads it below);
                # the keys listed before the one it answered were weighed and passed over
                stated.add(number)
                later.append(entry)
                named_any = True
                hits = [labels.index(label) for label in named if label in labels]
                self.weigh(number, labels[:min(hits)] if hits else labels[:1], weight)
                continue
            if burst:
                later.append(entry)
                continue
            if self.msgs_free() <= 0:
                # no message follows: the answer waits for the last read
                self.weigh(number, labels[:1], weight)
                later.append(entry)
                continue
            if len(sent) > 1:
                # past the first answer (always read, as v5.6's first wave was) a read must be quick, end well before
                # ASK_UNTIL and have a next message to steer
                expect = self.light_lat()
                if (slow or expect > READ_SLOW or time.time() - self.t0 + expect > ASK_UNTIL - READ_SLACK
                        or self.next_ask(orders, stated, set(closed), list(front), dict(rounds), puzzled, dict(weight), len(sent)) is None):
                    self.weigh(number, labels[:1], weight)
                    later.append(entry)
                    continue
            # every other answer is read before the next message, so the next message can follow it
            reading = time.time()
            outer = self.call_until
            if len(sent) > 1:
                # a read past the first answer ends by READ_SLOW + 3 s and gets no second try once slow: the answer then
                # waits for the last read, as in v5.6
                self.call_until = min(outer or float("inf"), reading + READ_SLOW + 3)
            try:
                heard = self.hear([entry], (LIGHT,), 800)
                if (not heard and not self.finished and time.time() - self.t0 < LATE - 10
                        and (len(sent) == 1 or time.time() - reading <= READ_SLOW)):
                    heard = self.hear([entry], (LIGHT,), 800)
            finally:
                self.call_until = outer
            slow = time.time() - reading > READ_SLOW
            if not heard:
                # unread: it waits for the last read, and its first key counts as weighed
                self.weigh(number, labels[:1], weight)
                later.append(entry)
                continue
            if len(labels) > 1:
                # a 'does not mind' to a list does not show which of its properties the shopper weighed
                heard = {k: v for k, v in heard.items() if k != "free"}
            lost = 1 in {_count(x) for x in _items(heard.get("lost"))}
            crowded = 1 in {_count(x) for x in _items(heard.get("crowded"))}
            grew = self.grown(heard, told)
            stated |= grew
            self.reroute(heard, [(number, labels, entry)], stated, closed, front, table)
            if number in grew:
                # a need only the read found (the answer named no listed key in code's reading): the keys before the one it
                # answered were weighed, as on the named path, so the next message does not lead with this one's lead again
                at = self.answered_at(number, labels, closed)
                self.weigh(number, labels[:at] if at is not None else labels[:1], weight)
                continue
            if lost:
                # race 15 said 'lost' whenever a message named none of the item's needs (the same text drew the same
                # reply 99% of the time; a re-ask that led with the need was answered 6 times in 6, one that left it out
                # 0 in 15): the listed keys count as weighed (nothing is marked free in the ledger), and the item waits
                # behind the others
                puzzled[number] = puzzled.get(number, 0) + 1
                self.weigh(number, labels, weight)
                continue
            puzzled.pop(number, None)
            if not crowded or len(labels) == 1:
                # a clear answer with no need: each listed key was weighed as surely as its place lets (a crowded reply to
                # one key closes it too, as v5.6 did: reroute moves on and the same question never goes out again)
                self.weigh(number, labels, weight)
        if later and not self.finished:
            # one read of the answers still unread: the heavy model when one of them named a key (or in a burst), else
            # the light one, as before; a read that returned nothing gets one more light try
            models, room = ((HEAVY, LIGHT), 1500) if named_any or burst else ((LIGHT,), 800)
            heard = self.hear(later, models, room)
            if not heard and not self.finished and time.time() - self.t0 < LATE - 10:
                heard = self.hear(later, (LIGHT,), room)
            self.take_heard(heard, told)
        self.trace.append({"kind": "asked", "picks": [f"{n}:{q[:60]}" for n, q, _ in sent], "stated": sorted(stated),
                           "free": [it.free for it in self.items], "puzzled": dict(puzzled)})
        if not sent:
            return ""
        if self.answer_notes:
            notes = [line for line in self.answer_notes if line not in self.heard]
            self.answer_notes = []
            meaning = self.take_note(notes) if notes else ""
            if meaning:
                self.queued.append("(came with the shopper's answers) " + meaning)
        if told:
            self.version += 1
            if self.loosening(sum(len(it.dropped) for it in self.items) > dropped, self.changes[start:]):
                self.relaxed = self.version
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        chained = self.resolve_refs() if self.chain_due() else ""
        if chained:
            summary += "\n" + chained
        looked = self.told_search()
        if looked:
            summary += "\n" + looked
        pointed = [need for _, need in self.refs if any(need in t for t in told) and need not in self.ref_values and need not in self.chain_tried]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def grown(self, heard: dict[str, Any], told: list[str]) -> set[int]:
        """take_heard, and the items whose told requirements it added to (their new ones kept in last_grown)."""
        sizes = {n: len(item.told) for n, item in enumerate(self.items, 1)}
        self.take_heard(heard, told)
        self.last_grown = {n: item.told[sizes[n]:] for n, item in enumerate(self.items, 1) if len(item.told) > sizes[n]}
        return set(self.last_grown)

    def answered_at(self, number: int, labels: list[str], closed: set[tuple[int, str]]) -> int | None:
        """The place of the listed key a need the last read added to the item answered: one the item's requirements now
        settle, else one whose words hold, or are held by, the new need's property words (a longer store spelling such as
        'graphics memory new' for 'graphics memory'), which is then closed. None when no listed key matches."""
        hits = [i for i, label in enumerate(labels) if self.settled_for(number, label)]
        if not hits:
            props = [_stem(m.split(":", 1)[0]) for m in self.last_grown.get(number, []) if ":" in m]
            hits = [i for i, label in enumerate(labels) if any(_names(p, _stem(label)) or _names(_stem(label), p) for p in props)]
            if hits:
                closed.add((number, _words(labels[min(hits)])))
        return min(hits) if hits else None

    def item_labels(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> set[str]:
        """The table's category labels that are the item's own shelf; when none is, the labels no other item owns."""
        def own(j: int) -> set[str]:
            item = self.items[j - 1]
            shelf, name = _words(item.shelf or ""), _words(item.name)
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            return {label for label, _, _, _ in table if _words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long}

        mine = own(number)
        if mine:
            return mine
        others = {label for j in range(1, len(self.items) + 1) if j != number for label in own(j)}
        return {label for label, _, _, _ in table if label not in others}

    def key_rank(self, labels: set[str], table: list[tuple[str, str, int, list[str]]]) -> list[dict[str, Any]]:
        """The table's keys of these categories, ranked by the opened listings' own records: the listings carrying the key,
        most first (more distinct values first among equals); a key on under 20% of the listings after those, a key with a
        single value on SINGLE_MIN or more listings last (race 14: keys on >= 80% of opened listings held a need 22% of the
        time, on < 20% 0.1%, single-valued keys 0.2%; race 18: keys on >= 80% of a shelf's listings 36%, on < 20% none, and
        by carried listings first the need's key ranked in the first three 36% of the time, by carried times values 27%).
        Each entry: key, carried, distinct, tier (0 good, 1 rare, 2 single-valued)."""
        names = list(dict.fromkeys(name for label, name, _, _ in table if label in labels))
        cat_key = self.category_key()
        total = 0
        stats: dict[str, tuple[int, set[str]]] = {name: (0, set()) for name in names}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            if label not in labels:
                continue
            total += 1
            done: set[str] = set()
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if name not in stats or name in done:
                    continue
                done.add(name)
                carried, values = stats[name]
                values.add(str(value).strip().lower())
                stats[name] = (carried + 1, values)
        out: list[dict[str, Any]] = []
        for place, name in enumerate(names):
            carried, values = stats[name]
            tier = 2 if len(values) <= 1 and carried >= SINGLE_MIN else (1 if total and carried < 0.2 * total else 0)
            out.append({"key": name, "carried": carried, "distinct": len(values), "tier": tier, "place": place})
        out.sort(key=lambda e: (e["tier"], -e["carried"], -e["distinct"], e["place"]))
        return out

    def ask_orders(self, table: list[tuple[str, str, int, list[str]]], count: int) -> dict[int, list[dict[str, Any]]]:
        """One PLAN call, shown each category's keys in the store's order (key_rank); its picks shape each item's ordered
        keys (order_keys). A failed plan leaves the store's order."""
        sparse = {n for n in range(1, len(self.items) + 1) if len(set(self.own_keys(n, table))) < SPARSE_KEYS}
        settled = [f"item {n}: {it.name} - settled by the request: " + ("; ".join(it.musts) or "nothing") + (" (few store keys)" if n in sparse else "")
                   for n, it in enumerate(self.items, 1)]
        listing: list[str] = []
        index = 0
        for label in dict.fromkeys(lab for lab, _, _, _ in table):
            listing.append(f"category {label!r}:")
            values = {name: vals for lab, name, _, vals in table if lab == label}
            for entry in self.key_rank({label}, table):
                index += 1
                listing.append(f"  {index}. {entry['key']} (on {entry['carried']}) [{entry['distinct']} values]: " + " | ".join(values.get(entry["key"]) or [])
                               + (" (one value on every listing)" if entry["carried"] >= SINGLE_MIN and entry["distinct"] == 1 else ""))
        each = PLAN_KEYS
        prompt = PLAN.replace("SPARSE_N", str(SPARSE_COUNT)).replace("COUNT", str(each)).replace("BAND", str(PLAN_KEYS)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        if sparse and self.raw_seen:
            prompt += "\n\nSTORE REPLIES (the last ones, as the store wrote them):\n" + "\n".join(self.raw_seen)
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        # the items' own name words say nothing about which property an entry asks about
        own = {w for it in self.items for w in _stem(f"{it.name} {it.shelf} {it.noun}")}
        firsts = [{w for w in _stem(x) if len(w) >= 3} - own for x in first]
        picks: list[tuple[int, str, bool, bool]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            if not isinstance(pick.get("label"), str):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per label (PLAN is told so): a label listing two by punctuation is cut to the first; a key name
            # in underscores reads as words; a label without letters, or naming a listing, is no property
            label = re.split(r"[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if "_" in label:
                label = _words(label)
            if not label or len(label.split()) > 5 or not re.search(r"[^\W\d_]", label) or label in self.rows:
                continue
            label = self.narrow(number, label, table)
            if self.settled_for(number, label) or any(n == number and _words(other) == _words(label) for n, other, _, _ in picks):
                continue
            mine = {w for w in _stem(label) if len(w) >= 3} - own
            asked_first = bool(mine) and any(mine <= words for words in firsts)
            picks.append((number, label, asked_first, bool(str(pick.get("hint") or "").strip())))
        orders = {n: self.order_keys(n, table, [p for p in picks if p[0] == n]) for n in range(1, len(self.items) + 1)}
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "order": {n: [e["label"] for e in v[:10]] for n, v in orders.items()}})
        return orders

    def order_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], mine: list[tuple[int, str, bool, bool]]) -> list[dict[str, Any]]:
        """The item's keys in asking order, each {label, first, good, tier}: what the request asks to check; the planner's
        picks that name a key, in the planner's order (single-valued ones keep their store place); the store order
        (key_rank) without keys the planner left out above its last plain (unhinted) pick; hinted properties no key
        names; the left-out keys; unhinted unlisted labels; the single-valued keys. The skips are read off the planner's
        walk only up to its last plain pick with at most PLAN_KEYS // 2 left-out keys above it. 'good' marks a key worth a
        place in a full message. With no listing of the item's shelf read, the planner's labels in its order (when it gave
        none, the store order of the shelves no other item owns, else of the keys two or more shelves share)."""
        ranked = [e for e in self.key_rank(self.item_labels(number, table), table) if not self.settled_for(number, e["key"])]
        out: list[dict[str, Any]] = []

        tiers = {e["key"]: e["tier"] for e in ranked}

        def add(label: str | None, first: bool = False, good: bool = False) -> None:
            if label and not any(_words(e["label"]) == _words(label) for e in out):
                out.append({"label": label, "first": first, "good": good, "tier": 0 if good else tiers.get(label, 1)})

        def inside(a: frozenset[str], b: frozenset[str]) -> bool:
            return bool(a) and all(any(_near_word(w, x) for x in b) for w in a)

        def key_of(label: str) -> str | None:
            want = _stem(label)
            exact = [e["key"] for e in ranked if _same_prop(_stem(e["key"]), want)]
            loose = [e["key"] for e in ranked if inside(want, _stem(e["key"])) or inside(_stem(e["key"]), want)]
            return (exact or loose or [None])[0]

        if len(set(self.own_keys(number, table))) < SPARSE_KEYS:
            # few keys of its own (records unreadable, nothing of its shelf opened): its own keys with several values
            # first, then the planner's labels in its order, each a full-message key, then the item's other few keys (an
            # item with none of its own gets no other shelf's keys)
            own = bool(self.own_keys(number, table))
            for e in (ranked if own and mine else []):
                if e["tier"] == 0 and e["distinct"] > 1:
                    add(e["key"], good=True)
            for _, label, first, _ in mine:
                add(label, first, True)
            if mine:
                for e in (ranked if own else []):
                    if any(_same_prop(_stem(e["key"]), _stem(x["label"])) for x in out):
                        continue
                    add(e["key"], good=e["tier"] == 0)
            else:
                if not ranked:
                    shared = {name for name, labels in self.key_shelves(table).items() if len(labels) >= 2}
                    ranked = [e for e in self.key_rank({label for label, _, _, _ in table}, table)
                              if e["key"] in shared and not self.settled_for(number, e["key"])]
                for e in ranked:
                    add(e["key"], good=e["tier"] == 0)
            return out
        keyed = [(label, first, hint, key_of(label)) for _, label, first, hint in mine]
        for label, first, _, key in keyed:
            if first:
                add(key or label, True, True)
        for _, _, _, key in keyed:
            if key and tiers.get(key, 0) < 2:
                add(key, good=True)
        kept = {key for _, _, _, key in keyed if key}
        # the planner's skips are read off its walk down the default order: keys above its last plain (unhinted) pick,
        # while few keys above that pick were left out (one deep pick is a jump, not a walk)
        plain = {key for _, first, hint, key in keyed if key and not (first or hint)}
        last = max((i for i, e in enumerate(ranked) if e["key"] in plain
                    and sum(1 for x in ranked[:i] if x["key"] not in kept) <= PLAN_KEYS // 2), default=-1)
        skipped = {e["key"] for i, e in enumerate(ranked) if i < last and e["key"] not in kept}
        for e in ranked:
            if e["tier"] < 2 and e["key"] not in skipped:
                add(e["key"], good=e["tier"] == 0)
        for label, _, hint, key in keyed:
            if hint and not key:
                add(label, good=True)
        for e in ranked:
            if e["key"] in skipped and e["tier"] < 2:
                add(e["key"])
        for label, _, _, key in keyed:
            if not key:
                add(label)
        for e in ranked:
            add(e["key"])
        return out

    @staticmethod
    def key_shelves(table: list[tuple[str, str, int, list[str]]]) -> dict[str, set[str]]:
        """Each key name -> the category labels whose listings carry it."""
        out: dict[str, set[str]] = {}
        for label, name, _, _ in table:
            out.setdefault(name, set()).add(label)
        return out

    def carried_together(self, a: str, b: str) -> bool:
        """Some opened listing's attributes carry both keys (two properties, not two spellings of one)."""
        want = {_words(a), _words(b)}
        return any(want <= {_words(key) for key in (_fact_block(r) or {})} for r in self.full.values() if isinstance(r, dict))

    def listed(self, number: int, label: str, table: list[tuple[str, str, int, list[str]]]) -> bool:
        """The label names a key of the item's own listings: the label's words inside a key's, or a key's words inside the
        label's, up to one letter a word; or no listing of its shelf was read, so nothing says otherwise."""
        own = [_stem(k) for k in self.own_keys(number, table)]
        want = _stem(label)

        def inside(a: frozenset[str], b: frozenset[str]) -> bool:
            return bool(a) and all(any(_near_word(w, x) for x in b) for w in a)

        return not own or any(inside(want, k) or inside(k, want) for k in own)

    def settled_for(self, number: int, label: str) -> bool:
        """The item's requirements in force already settle this property (spelling-tolerant)."""
        item = self.items[number - 1]
        want = _stem(label)
        return any(_same_prop(want, _stem(m.split(":")[0])) for m in item.musts + item.told if ":" in m and m not in item.dropped)

    def next_ask(self, orders: dict[int, list[dict[str, Any]]], stated: set[int], closed: set[tuple[int, str]],
                 front: list[tuple[int, list[str]]], rounds: dict[int, int], puzzled: dict[int, int] | None = None,
                 weight: dict[tuple[int, str], float] | None = None, count: int = 0) -> tuple[int, list[str]] | None:
        """The next message: a move a crowded reply asked for first; else the item asked least among those with no need
        yet (an open ask-first key, then the item the others have to fit, first), with its best open keys: the leading
        good ones up to ASK_WIDTH, topped up to ASK_SPARE with keys not single-valued (tier < 2) when fewer than 3 are good,
        ASK_SPARE of any when none is; one key only after a crowded reply. A front move for an item already asked waits
        while the messages left do not exceed the other items with no question and no need yet. Of two keys where one's
        words hold the other's (the extra words the item's own name or shelf words, or no opened listing carrying both:
        two spellings of one property), the message keeps the longer. An item that puzzled the shopper comes after the
        others, and gets none once puzzled twice or once `count` messages reach PUZZLE_AFTER. A message's keys go
        likeliest first, its lead included (the planner's place times the share earlier messages left unweighed, ask-first
        keys ahead); a key that led a message is closed (its unweighed share is 0)."""
        puzzled = puzzled if puzzled is not None else {}
        weight = weight if weight is not None else {}

        def is_open(number: int, label: str) -> bool:
            return ((number, _words(label)) not in closed and not self.settled_for(number, label)
                    and weight.get((number, _words(label)), 1.0) > OPEN_MIN)

        def unasked(number: int) -> int:
            return sum(1 for j, entries in orders.items() if j != number and not rounds.get(j) and j not in stated
                       and any(is_open(j, e["label"]) for e in entries))

        for place, (number, labels) in enumerate(list(front)):
            if rounds.get(number) and self.msgs_free() <= unasked(number):
                continue
            front.pop(place)
            labels = [label for label in labels if is_open(number, label)]
            if labels:
                return number, labels[:1] if self.one_key else labels
            return self.next_ask(orders, stated, closed, front, rounds, puzzled, weight, count)
        hosts = {it.fits for it in self.items if it.fits}
        best: tuple[tuple[Any, ...], int, list[dict[str, Any]]] | None = None
        for number, entries in orders.items():
            if puzzled.get(number, 0) >= 2 or (puzzled.get(number) and count >= PUZZLE_AFTER):
                # race 15: later messages on an item that puzzled the shopper came back clear 1 time in 185
                continue
            place = {id(e): k for k, e in enumerate(entries)}
            free = [e for e in entries if is_open(number, e["label"])]
            if not free:
                continue
            # every key goes likeliest first, the lead too: the planner place's prior times the share earlier messages left
            # unweighed (ask-first keys ahead while no message has weighed them). After a six-key 'no preference' list the
            # next message leads with that list's least-weighed likely key (its 4th, heard about 0.22-0.5 of the time),
            # never with its 2nd (v5.10 did: in race 18 those re-asked leads drew a need 8% of the time, other leads 31-37%)
            free = sorted(free, key=lambda e: (not (e["first"] and weight.get((number, _words(e["label"])), 1.0) >= 1.0),
                                               -(PRIOR_STEP ** place[id(e)]) * weight.get((number, _words(e["label"])), 1.0)))
            # an ask-first lead counts for the item's turn only while no message weighed it, as in the sort above
            lead_first = free[0]["first"] and weight.get((number, _words(free[0]["label"])), 1.0) >= 1.0
            rank = (bool(puzzled.get(number)) and not lead_first, number in stated, not lead_first,
                    rounds.get(number, 0), number not in hosts, number)
            if best is None or rank < best[0]:
                best = (rank, number, free)
        if best is None:
            return None
        _, number, free = best
        if self.one_key:
            return number, [next((e for e in free if e["good"]), free[0])["label"]]
        size = 0
        # the message: its good keys in that order (a key that is not good no longer cuts the list short, now that weighed
        # keys sort among fresh ones), then, when fewer than 3 are good, keys not single-valued after them up to ASK_SPARE;
        # ASK_SPARE of any when none is good
        chosen = [e for e in free if e["good"]][:ASK_WIDTH]
        if not chosen:
            chosen = free[:ASK_SPARE]
        elif len(chosen) < 3:
            chosen += [e for e in free if not e["good"] and e["tier"] < 2][:min(ASK_SPARE, len(free)) - len(chosen)]
        free, size = chosen, len(chosen)
        item = self.items[number - 1]
        own = _stem(f"{item.name} {item.shelf} {item.noun}")

        def twins(a: str, b: str) -> bool:
            x, y = _stem(a), _stem(b)
            small, big = (x, y) if len(x) < len(y) else (y, x)
            return bool(small) and small < big and (big - small <= own or not self.carried_together(a, b))

        labels: list[str] = []
        for e in free[:size]:
            twin = next((x for x in labels if twins(x, e["label"])), None)
            if twin is None:
                labels.append(e["label"])
                continue
            longer, shorter = (e["label"], twin) if len(_stem(e["label"])) > len(_stem(twin)) else (twin, e["label"])
            labels[labels.index(twin)] = longer
            closed.add((number, _words(shorter)))
        return number, labels

    def light_lat(self) -> float:
        """The light model's recent call time: the median of its last five calls (0 before any)."""
        if not self.light_times:
            return 0.0
        return sorted(self.light_times)[len(self.light_times) // 2]

    def weigh(self, number: int, labels: list[str], weight: dict[tuple[int, str], float]) -> None:
        """These listed keys drew no need: each one's unweighed share drops by how surely the shopper weighs a property
        at its place in a list (HEARD_AT)."""
        for place, label in enumerate(labels):
            seen = HEARD_AT[place] if place < len(HEARD_AT) else HEARD_AT[-1]
            key = (number, _words(label))
            weight[key] = weight.get(key, 1.0) * (1.0 - seen)

    def reroute(self, heard: dict[str, Any], loose: list[tuple[int, list[str], tuple[int, str, str]]], stated: set[int],
                closed: set[tuple[int, str]], front: list[tuple[int, list[str]]], table: list[tuple[str, str, int, list[str]]]) -> None:
        """What the light read of an answer says for the next message. A reply that only did not follow the question
        weighed no property (none is marked free; the interview puts its item behind the others). A crowded reply to several properties switches to one
        property per message for the rest of the episode; its keys open again and that item's best one goes first. A
        'one at a time' reply to one property on an item with no need yet sends the same word to the twin item whose own
        keys carry it (one with no need yet first, the property not settled there), else a plainer key of this item;
        never the same question with example values."""
        lost = {_count(x) for x in _items(heard.get("lost"))}
        crowded = {_count(x) for x in _items(heard.get("crowded"))} - lost
        for index, (number, labels, _) in enumerate(loose, 1):
            if index in lost:
                continue
            if index not in crowded:
                continue
            if len(labels) > 1:
                self.one_key = True
                closed.difference_update((number, _words(label)) for label in labels)
                if number not in stated:
                    front.insert(0, (number, labels[:1]))
                continue
            if number in stated:
                continue
            label = labels[0]
            want = _stem(label)
            twins = [j for j in range(1, len(self.items) + 1) if j != number and (j, _words(label)) not in closed
                     and not self.settled_for(j, label) and any(_same_prop(_stem(k), want) for k in self.own_keys(j, table))]
            twins.sort(key=lambda j: j in stated)
            if twins:
                front.append((twins[0], [label]))
                continue
            other = self.other_key(number, label, table, {w for n, w in closed if n == number})
            if other:
                front.append((number, [other]))

    def names_key(self, answer: str, label: str | list[str], table: list[tuple[str, str, int, list[str]]]) -> bool:
        """The answer names an asked property or a key the listings carry (named_keys)."""
        return bool(Kyo.named_keys(answer, label, table))

    @staticmethod
    def named_keys(answer: str, label: str | list[str], table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The asked labels and the listed keys the answer names, every word of each: a key of two words or more, or a
        one-word key whose listed values share a word with the answer (a store-wide one-word key such as a variants key
        made 'any option is fine' look like a stated need). Each asked label is tested on its own, so an answer about
        the third property a message listed names that one. A one-word asked label also needs one of its key's listed
        values in the answer ("as you see fit" does not name 'fit')."""
        said = _stem(re.sub(r"[^A-Za-z0-9]+", " ", answer))
        out: list[str] = []
        for asked in [label] if isinstance(label, str) else list(label):
            want = _stem(asked)
            if want and want <= said and len("".join(want)) >= 3 and asked not in out:
                if len(want) == 1 and not any(_stem(str(v)) & said for _, name, _, values in table if _stem(name) == want for v in values):
                    continue
                out.append(asked)
        for _, name, _, values in table:
            key = _stem(name)
            if not key or len("".join(key)) < 3 or not key <= said or name in out:
                continue
            if len(key) >= 2 or any(_stem(str(v)) & said for v in values):
                out.append(name)
        return out

    def told_values(self, number: int, new: list[str]) -> list[str]:
        """The words to search each new need by: the answer reader's 'find' (a figure it gives is spelt as the listings
        print it; "" for an exclusion or a pointer, never searched), else v4's rule: the value when it is a few words
        without a figure, or a short limit on one figure searched by that figure as the listings print it (printed_figure,
        such as "20000mAh")."""
        out: list[str] = []
        for need in new:
            if (number, need) in self.find_words:
                value = self.find_words[(number, need)]
                if re.search(r"\d", value) and not re.search(r"[A-Za-z]", value):
                    value = ""          # a bare figure, no unit: it matches any listing's numbers
                elif re.match(r"\d", value):
                    value = self.printed_figure(value) or value
            else:
                value = need.split(":", 1)[1].strip()
                value = value if len(value.split()) <= 5 and not re.search(r"\d", value) else ""
                if not value:
                    value = self.printed_figure(need.split(":", 1)[1].strip())
            if value and len(value.split()) <= 6 and value.lower() not in [v.lower() for v in out]:
                out.append(value)
        return out

    def told_search(self) -> str:
        """After the answers, for each item with new worded values: Kyo searches them with the item's other stated
        search words and its product word (the brief's noun, else its store label, else its name); then with the item's
        name (v4's query); then the first value alone; each only when the one before returned no rows. It opens the best
        unread rows inside the budget: its department first (when none is of its own, rows of others that show the value or
        the product's words), then rows whose own words show the value, then rows showing more of its other values, then the store's order (under a cheapest
        pick: values stated, then price). With fewer than 22 steps beyond the finish it is v4's size: one search and two openings. The walker
        is told what each opened listing's attributes state for these properties."""
        tool = self.roles.get("search")
        if self.finished or not tool or self.left() < 14 or time.time() - self.t0 > 180 or self.cost > self.hard:
            return ""
        refs = {need for _, need in self.refs}
        work = []
        for number, item in enumerate(self.items, 1):
            new = [m for m in item.told[self.told_before.get(number, 0):] if m not in refs and m not in item.dropped and ":" in m]
            values = self.told_values(number, new)
            if values:
                work.append((number, item, new, values))
        # the scan's room: the finish, two steps an item to cart and look, and eight to judge stay with the walker
        each = min(6, (self.left() - self.tail() - 8 - 2 * len(self.items)) // max(1, len(work)))
        if not work or each < 2:
            return ""
        small = self.left() - self.tail() < 22
        cat_key = self.category_key()
        lines: list[str] = []
        for number, item, new, values in work:
            if self.finished or time.time() - self.t0 > 190 or self.left() < 12:
                break
            begin = self.step
            live = [m for m in item.musts + item.told if ":" in m and m not in item.dropped and m not in refs and m not in new]
            stated = [v for v in (item.find.get(m) if m in item.find else self.find_words.get((number, m), "") for m in live)
                      if v and len(v.split()) <= 3]
            word = item.noun or item.shelf or item.name
            plain = " ".join(values[:2] + [item.name]).strip()
            narrow = " ".join(values[:2] + stated[:2] + [word]).strip()
            # the narrow query first while it stays short (keyword searches read few words), then v4's, then the first
            # value alone
            ladder = list(dict.fromkeys(q for q in (narrow if len(narrow.split()) <= 5 else "", plain, values[0]) if q))
            ran: list[str] = []
            pool: list[str] = []
            for query in ladder:
                # the next query only when the one before returned no rows; a small scan tries v4's query after an
                # empty first one while it has three steps
                if len(ran) >= ((2 if each >= 3 else 1) if small else each - 1):
                    break
                before, call = self.step, getattr(self, "last_search", None)
                found = self.chain_search(query)
                if self.step == before:
                    break
                ran.append(query)
                if self.last_search is not call:
                    # the scan's own call: a walker repeat of it costs no step
                    self.read_cache[tool + _dump(self.last_search)] = (self.version, self.step, self.show(self.last_seen))
                    self.sweep_calls.add(tool + json.dumps(self.last_search, sort_keys=True, default=str))
                if found:
                    pool = [i for i, _ in self.records_in(self.last_seen)]
                    break
            if not ran:
                continue
            told_toks = [_toks(v, True) for v in values]
            other_toks = {w for v in stated if not re.search(r"\d", v) for w in _toks(v, True) if len(w) >= 3}

            def row_toks(i: str) -> frozenset[str]:
                return _toks(" ".join(v for v in (self.rows.get(i) or {}).values() if isinstance(v, str)), True)

            def seen_by(i: str) -> int:
                # 0: the row's own words show a told value; 1: a one-page word search holding the value returned it
                # (the store's word search reads page facts too); 2: neither
                if any(t and t <= row_toks(i) for t in told_toks):
                    return 0
                return 1 if any(self.query_shows(i, t) for t in told_toks) else 2

            def shows(i: str) -> bool:
                return seen_by(i) < 2

            def dept(i: str) -> int:
                # 0: the item's department, 1: no label known, 2: another department
                return 1 if not self.label_of(i, cat_key) else (0 if self.dept_match(number, i, cat_key, fold=True) else 2)

            def over(i: str, price: float | None) -> bool:
                return (self.budget is not None and len(self.items) == 1 and self.brief.get("budget_scope") != "each"
                        and price is not None and price * item.qty > self.budget)

            def in_budget(i: str) -> bool:
                return not over(i, self.price_of(i))

            unread = [i for i in pool if i not in self.full and self.buyable(i) and in_budget(i)]
            if any(dept(i) == 0 for i in unread):
                unread = [i for i in unread if dept(i) < 2]
            else:
                # no row of its own department: the rows that show the value or the product's words, else all of them
                named = {w for w in _toks(f"{word} {item.name}", True) if len(w) >= 3}
                unread = [i for i in unread if shows(i) or named & row_toks(i)] or unread
            order = {i: k for k, i in enumerate(pool)}
            if self.brief.get("pick") == "cheapest":
                # stage 4's order under a cheapest pick: the department, the values a row states, then the lower price
                unread.sort(key=lambda i: (dept(i), -self.screen(i, values), seen_by(i), self.price_of(i) if self.price_of(i) is not None else float("inf")))
            else:
                # a row over the budget by its one number field (no shape reading yet) goes after the others
                unread.sort(key=lambda i: (dept(i), over(i, self.near_price(i)), seen_by(i), -len(other_toks & row_toks(i)), order[i]))
            # a search with few hits (all on one page) gets more of its rows opened while steps allow
            wide = 6 if self.last_hits is not None and self.last_hits <= 12 and self.left() - self.tail() > 14 else 4
            room = min(2 if small or time.time() - self.t0 > 150 else wide, each - (self.step - begin))
            ids: list[str] = []
            if unread and room > 0:
                if not small and room >= 2 and self.compare_target() and self.roles.get("compare") and not self.compare_bare():
                    self.open_rows(unread[: min(self.compare_room(), room)], self.tail() + 8)
                    ids = [i for i in unread[:room] if i in self.full]
                if not ids:
                    ids = [t.split(":", 1)[0] for t in self.fill_facts(unread[:room], self.tail() + 8)]
            self.trace.append({"kind": "told_search", "words": ran, "pool": len(pool), "cand": unread[:6], "opened": ids})
            asked = " or ".join(repr(q) for q in ran)
            if not pool:
                lines.append(f"item {number} ({'; '.join(new)}): Kyo's keyword search for {asked} showed no rows; search shorter words yourself")
                continue
            props = [m.split(":", 1)[0].strip() for m in new]
            props += [p for p in (m.split(":", 1)[0].strip() for m in live) if p not in props]
            seen_lines = []
            for i in ids:
                block = _fact_block(self.full.get(i) or {}) or {}
                hits = [f"{k}={str(v)[:40]}" for p in props for k, v in block.items() if _names(_stem(p), _stem(k))]
                price = self.price_of(i)
                said = ", ".join(dict.fromkeys(hits[:4])) or ("no attribute names " + " or ".join(props[:2]) + "; check its title and description")
                seen_lines.append(i + (f" ({_num(price)})" if price is not None else "") + ": " + said)
            # only rows whose own words show the value are named to the walker as showing it
            rest = [i for i in unread if i not in ids and seen_by(i) == 0][:4]
            text = f"item {number} ({'; '.join(new)}): searched {asked}"
            if seen_lines:
                text += "; opened " + "; ".join(seen_lines)
            if rest:
                text += "; unread rows showing the value: " + ", ".join(rest)
            if not seen_lines and not rest:
                text += "; no unread row of it to open"
            lines.append(text)
        if not lines:
            return ""
        judged = any("; opened " in line or "; unread rows" in line for line in lines)
        said = ("[Kyo searched the shopper's new values itself] " + " | ".join(lines)
                + (". Judge these listings first: cart one whose records state every requirement in force (the attribute for that"
                   " property, else its title or description)." if judged else "."))
        self.trace.append({"kind": "told_text", "text": said[:600]})
        return said

    def printed_figure(self, value: str) -> str:
        """A short told limit on one figure as the store's listings print it: the figure and the word right after it,
        joined or spaced and spelt as most seen records write that word after a figure (spaced, as told, when none does);
        the figure alone when no word follows it. "" for a longer value or one with several figures."""
        figures = re.findall(r"\d+(?:[.,]\d+)*", value)
        if len(figures) != 1 or len(value.split()) > 5:
            return ""
        figure = figures[0].replace(",", "")
        after = re.search(re.escape(figures[0]) + r"\s*([A-Za-z]+)", value)
        if not after:
            return figure
        unit = after.group(1)
        text = " ".join(_dump(r) for r in list(self.rows.values())[-80:])
        seen = re.findall(r"\d( ?)(" + re.escape(unit) + r")(?![A-Za-z])", text, re.I)
        if not seen:
            return figure + " " + unit
        space, spelt = max(sorted(set(seen)), key=seen.count)
        return figure + space + spelt

    def menu_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[tuple[str, int]]:
        """The menu keys of the item's shelf (or of every shelf when none matches), most carried first."""
        item = self.items[number - 1]
        shelf, name = _words(item.shelf or ""), _words(item.name)
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        mine = [(key, n) for label, key, n, _ in table if _words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long]
        return sorted(mine or [(key, n) for _, key, n, _ in table], key=lambda kn: -kn[1])

    def own_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The keys of the item's own shelf only (no fallback to other shelves)."""
        item = self.items[number - 1]
        shelf, name = _words(item.shelf or ""), _words(item.name)
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        return [key for label, key, _, _ in table if _words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long]

    def narrow(self, number: int, label: str, table: list[tuple[str, str, int, list[str]]]) -> str:
        """'shape' beside 'case shape' and 'dial shape' asks about two things: the most carried key holding the label's words."""
        keys = self.menu_keys(number, table)
        want = _stem(label)
        if any(_stem(k) == want for k, _ in keys):
            return label
        wider = [(k, n) for k, n in keys if want and want < _stem(k)]
        return wider[0][0] if wider else label

    def other_key(self, number: int, label: str, table: list[tuple[str, str, int, list[str]]], used: set[str]) -> str | None:
        """After 'one at a time': a menu key that shares a word with the label but is not it (the plainer one first)."""
        want = _stem(label)
        item = self.items[number - 1]
        settled = {_stem(m.split(":")[0]) for m in item.musts + item.told if ":" in m}
        keys = [(k, n) for k, n in self.menu_keys(number, table) if _stem(k) != want and _words(k) not in used and _stem(k) & want and _stem(k) not in settled]
        keys.sort(key=lambda kn: (not _stem(kn[0]) < want, -kn[1]))
        return keys[0][0] if keys else None

    def question(self, number: int, labels: str | list[str]) -> str:
        """One property: "What A would you like for the X?"; several: one message naming them all, best first."""
        keys = [labels] if isinstance(labels, str) else list(labels)
        name = self.items[number - 1].name
        if len(keys) == 1:
            return f"What {keys[0]} would you like for the {name}?"
        return f"For the {name}, what {', '.join(keys[:-1])} or {keys[-1]} would you like?"

    def pack_question(self, pack: list[tuple[int, list[str]]]) -> str:
        """Several items in one message: "What A or B would you like for the X, and what C or D for the Y?"."""
        def joined(keys: list[str]) -> str:
            return keys[0] if len(keys) == 1 else f"{', '.join(keys[:-1])} or {keys[-1]}"

        parts = [(self.items[number - 1].name, joined(labels)) for number, labels in pack]
        head = f"What {parts[0][1]} would you like for the {parts[0][0]}"
        rest = [f"what {keys} for the {name}" for name, keys in parts[1:]]
        if len(rest) == 1:
            return f"{head}, and {rest[0]}?"
        return f"{head}, " + ", ".join(rest[:-1]) + f", and {rest[-1]}?"

    def next_pack(self, orders: dict[int, list[dict[str, Any]]], stated: set[int], closed: set[tuple[int, str]],
                  front: list[tuple[int, list[str]]], rounds: dict[int, int], puzzled: dict[int, int],
                  weight: dict[tuple[int, str], float], count: int) -> list[tuple[int, list[str]]] | None:
        """Several items in the next message: when more items wait for their first question (no need yet, never asked,
        never puzzled, an open key) than messages are left, the message names ceil(waiting / messages left) of them, at
        most PACK_MAX, in next_ask's order, each with its leading keys (PACK_WIDTH shared out, at most PACK_KEYS an item;
        one each after a crowded reply). None when one item a message reaches them all, a front move waits, or a
        crowded reply came to such a message."""
        left = self.msgs_free()
        if front or left <= 0 or self.no_pack:
            return None

        def is_open(number: int, label: str) -> bool:
            return ((number, _words(label)) not in closed and not self.settled_for(number, label)
                    and weight.get((number, _words(label)), 1.0) > OPEN_MIN)

        waiting = [n for n, entries in orders.items() if n not in stated and not rounds.get(n) and not puzzled.get(n)
                   and any(is_open(n, e["label"]) for e in entries)]
        if len(waiting) <= left:
            return None
        size = min(PACK_MAX, -(-len(waiting) // left))
        each = 1 if self.one_key else max(1, min(PACK_KEYS, PACK_WIDTH // size))
        probe_closed, probe_puzzled = set(closed), dict(puzzled)
        pack: list[tuple[int, list[str]]] = []
        while len(pack) < size:
            pick = self.next_ask(orders, stated, probe_closed, [], rounds, probe_puzzled, weight, count)
            if pick is None:
                break
            number, labels = pick
            if number not in waiting or any(number == n for n, _ in pack):
                break
            pack.append((number, labels[:each]))
            # next_ask passes over an item twice puzzled: the next pick is another item
            probe_puzzled[number] = 2
        if len(pack) < 2:
            return None
        # the twin merges next_ask made while picking stand
        closed.update(probe_closed - {(n, _words(label)) for n, labels in pack for label in labels})
        return pack

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6, process: bool = False) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            # no message past LATE, nor one after which a cart call and the checkout could not land by HARD_END-5 s
            elapsed = time.time() - self.t0
            # (a process step's message at the gate needs only itself and the checkout after it)
            late = elapsed + 2 * self.store_lat > HARD_END - 5 if process else (elapsed > LATE or elapsed + 3 * self.store_lat > HARD_END - 5)
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit or late:
                break
            args = self.message_args(text)
            if not args:
                self.ask_failed = True
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get("observation")
            # a change that came with the answer goes to the note reader; the answer alone goes to the hearing
            said = self.take_unread(reply.get("said"))
            self.answer_notes += [t for t in said if t not in self.answer_notes]
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r}{_said(said)[:400]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
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
            find = entry.get("find")
            if isinstance(find, list) and find and all(isinstance(x, str) for x in find):
                find = " ".join(x.strip() for x in find if x.strip())
            if isinstance(find, str):
                # "" says the need is no value to search by (a limit, an exclusion, a pointer); no string: v4's rule
                self.find_words[(number, need)] = find.strip()
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
        moved = self.set_budget(heard.get("budget"), "answer")
        if moved:
            told.append(moved)

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT), max_tokens: int = 1500) -> dict[str, Any]:
        """Read the answers; under 900 tokens the light model answers with its thinking off."""
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        def about(n: int, q: str) -> str:
            several = self.packed.get(q) or []
            if len(several) < 2:
                return f"item {n}"
            return "items " + ", ".join(str(x) for x in several[:-1]) + f" and {several[-1]}"

        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about {about(n, q)}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), max_tokens) or {}

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
                batch = self.send([(1, "", text)], 1, process=True)
                if not batch:
                    break
                self.gate_msgs += 1
                notes, self.answer_notes = self.answer_notes, []
                fresh = [line for line in notes if line not in self.heard]
                said.append(f"Q: {text}\nA: {batch[0][2]}" + _said(notes, self.take_note(fresh) if fresh else ""))
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            notes = self.take_unread(reply.get("said"))
            extra = _said(notes, self.take_note(notes) if notes else "")
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]}{extra[:400]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}" + extra)
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
        start = len(self.changes)
        changed = []
        moved = self.set_budget(got.get("budget"), "note")
        if moved:
            changed.append(moved)
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
        self.changes += [{"step": self.step, "kind": "drop", "what": str(e["need"]).strip()} for e in drops]
        self.changes += [{"step": self.step, "kind": "add", "what": str(e["need"]).strip()} for e in adds]
        if self.note_sold:
            self.changes.append({"step": self.step, "kind": "gone", "what": ", ".join(gone)})
        if got.get("price_moved") is True:
            self.changes.append({"step": self.step, "kind": "price", "what": ""})
        loose = self.loosening(bool(drops), self.changes[start:])
        if len(self.changes) > start:
            self.change_due = True
            if loose or self.note_sold:
                self.rearm()
            elif got.get("price_moved") is True:
                # a price-only note: the wider search is re-armed only when a re-read line's price rose or the total breaks
                # the budget (judged before the next checkout)
                self.price_wait = self.step
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            if loose:
                self.relaxed = self.version
            # Kyo's own reading first: the note's summary follows its wording, which does not show which way a number went
            self.notes.append("; ".join(changed) or summary)
            if not changed:
                self.free_notes.append(summary)
        if moved or not summary:
            return "; ".join(changed)
        return ("; ".join(changed) + f" (the note says: {summary})") if changed else summary

    def take_unread(self, said: Any = None) -> list[str]:
        """The messages act() took off replies since they were last read, with `said`, minus those read already."""
        lines = [t for t in dict.fromkeys(self.unread + [t for t in _items(said) if isinstance(t, str)]) if t not in self.heard]
        self.unread = []
        return lines

    def set_budget(self, value: Any, source: str) -> str:
        """A new budget from a note or an answer. Its direction comes from the numbers; the message's wording says nothing
        about it. The change is recorded, and when a budget already in force changes with no price key known yet, the
        rows' one number key becomes it, so the cart's total can be checked. Returns Kyo's words for it, or ""."""
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0 or value == self.budget:
            return ""
        value = int(value) if float(value).is_integer() else float(value)
        old = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        self.budget = value
        way = "set" if old is None else ("raised" if value > old else "cut")
        self.changes.append({"step": self.step, "kind": "budget", "old": old, "new": value, "way": way, "source": source})
        self.change_due = True
        self.window = (float(old), float(value)) if old is not None and value > old else None
        if old is not None and not self.price_key:
            self.price_key = self.row_key(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
            self.trace.append({"kind": "price_key", "key": self.price_key})
        return f"budget {way}" + (f" from {_num(old)}" if old is not None else "") + f" to {_num(value)}"

    def loosening(self, drops: bool, changes: list[dict[str, Any]]) -> bool:
        """Whether a change can make another listing the right pick: a dropped requirement, or a budget among `changes`
        (the ones just recorded) that a cart line now breaks, or a changed budget while the cart is empty. A higher
        budget over a carted line that fits changes no cheapest pick, and a first budget only rules listings out."""
        if drops:
            return True
        ways = [c["way"] for c in changes if c["kind"] == "budget"]
        if not ways:
            return False
        ids = list(dict.fromkeys(self.cart_ids()))
        if not ids:
            return ways[-1] != "set"
        if self.brief.get("budget_scope") == "each":
            return any((self.price_of(i) or 0.0) > self.budget for i in ids)
        return self.cart_broken()

    def rearm(self) -> None:
        """A change that can make a cheaper listing the right one under a cheapest pick: the wider search before the
        checkout is held once more, naming the change."""
        if self.brief.get("pick") == "cheapest":
            self.wider_held = False
            self.rewiden = True

    def change_lines(self) -> list[str]:
        """What each change of this episode means for the picks, in plain words (newest last)."""
        cheapest = self.brief.get("pick") == "cheapest"
        out = []
        for c in self.changes[-6:]:
            if c["kind"] == "budget" and c["way"] == "raised":
                out.append(f"budget raised from {_num(c['old'])} to {_num(c['new'])}: a listing priced up to {_num(c['new'])} is inside the budget now, so one turned down only for its price may fit")
            elif c["kind"] == "budget" and c["way"] == "cut":
                out.append(f"budget cut from {_num(c['old'])} to {_num(c['new'])}: the cart's total must be at most {_num(c['new'])}")
            elif c["kind"] == "budget":
                out.append(f"budget now {_num(c['new'])}")
            elif c["kind"] == "drop":
                out.append(f"dropped {c['what']}: a listing turned down only for that may fit now" + ("; a cheaper one may now be the cheapest fit" if cheapest else ""))
            elif c["kind"] == "add":
                out.append(f"added {c['what']}: every cart line must meet it too")
            elif c["kind"] == "gone":
                out.append("a chosen listing sold out" + (f" ({c['what']})" if c["what"] else "") + ": its line needs a replacement" + (", the cheapest other fit" if cheapest else ""))
            elif c["kind"] == "price":
                out.append("a price moved: the cart's total must stay inside the budget" + (" and its line must still be the cheapest fit" if cheapest else ""))
        return out

    def carried(self, reply: dict[str, Any] | None, ending: str = "") -> str:
        """The messages a store reply carried beside its result, for any text built from it; `ending` names the ending the
        store turned down (the reply did not finish the episode)."""
        notes = [t for t in _items((reply or {}).get("said")) if isinstance(t, str)]
        if not notes:
            return ""
        text = _said(notes)
        if ending and not (reply or {}).get("finished"):
            text += f"\n[Kyo] The {ending} did not go through: the store sent these messages instead. The episode goes on."
        return text

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

    def review_record(self, ident: str, limit: int = 1200, short: int = 200, record: dict[str, Any] | None = None) -> str:
        """A listing's record for a review prompt in at most `limit` characters: its id first, then its attribute block
        (values clipped at 120, shorter when the block alone is too long), then its other fields up to `short` characters,
        then its longer texts sharing the room left; a field that would cross the limit is left out. A fixed cut of the
        dumped record could drop the id or part of the attributes. `record` defaults to the fullest record seen."""
        if record is None:
            record = self.full.get(ident) or self.rows.get(ident)
        record = record if isinstance(record, dict) else {}
        out: dict[str, Any] = {self.id_key or "id": ident}
        block = _fact_key(record)
        if block is not None and block not in out:
            facts = _merge_spellings(record[block]) if isinstance(record[block], dict) else record[block]
            live = self.live_values()

            def cut_to(value: Any, cut: int) -> Any:
                # a clipped text keeps the live requirement values whose words its full text holds
                if not isinstance(value, str) or len(value) <= cut:
                    return _clip(value, cut)
                short_text = value[:cut]
                words = _toks(value, True)
                kept = [v for v in live if _toks(v, True) <= words and v.lower() not in short_text.lower()]
                return short_text + "..." + (" (" + "; ".join(kept[:4]) + ")" if kept else "")

            for cut in (120, 60, 30, 12, 0):
                part = {k: (cut_to(v, cut) if cut else "") for k, v in facts.items()}
                if len(_dump({**out, block: part})) <= limit:
                    break
            else:
                part = {}
                for k in facts:
                    if len(_dump({**out, block: {**part, k: ""}})) > limit:
                        break
                    part[k] = ""
            out[block] = part
        rest = [k for k in record if k not in out]
        prose = [k for k in rest if isinstance(record[k], str) and len(record[k]) > short]
        for key in rest:
            if key not in prose:
                value = _clip(record[key], short)
                if len(_dump({**out, key: value})) <= limit:
                    out[key] = value
        # a short prose field (a pointer or note sentence) whole first, while it fits beside a 40-character share for each
        # longer one; the long texts then share the room left
        for key in sorted(prose, key=lambda k: len(record[k])):
            rest_long = [k for k in prose if k != key and k not in out]
            if len(record[key]) <= 3 * short and len(_dump({**out, key: record[key]})) + 60 * len(rest_long) <= limit:
                out[key] = record[key]
        prose = [k for k in prose if k not in out]
        for number, key in enumerate(prose):
            text = record[key]
            size = (limit - len(_dump(out))) // (len(prose) - number) - len(_dump(key)) - 6
            while size >= 40:
                value = text if len(text) <= size else text[:size] + "..."
                over = len(_dump({**out, key: value})) - limit
                if over <= 0:
                    out[key] = value
                    break
                size -= over
        return _dump(out)

    def live_values(self) -> list[str]:
        """The worded values of every requirement in force (as told, and the words to search them by), a few words each."""
        refs = {need for _, need in self.refs}
        out: list[str] = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if ":" not in need or need in item.dropped or need in refs:
                    continue
                found = item.find.get(need) if need in item.find else self.find_words.get((number, need), "")
                for value in (need.split(":", 1)[1].strip(), found or ""):
                    if value and re.search(r"[A-Za-z]", value) and len(value.split()) <= 6 and value not in out:
                        out.append(value)
        return out

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

    def departments(self) -> str:
        """The store's department labels seen on listings (the category key's values)."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        labels = sorted({str(r.get(cat_key)).strip() for r in self.rows.values() if isinstance(r.get(cat_key), str) and r.get(cat_key).strip()})
        return ("\n\nDEPARTMENT LABELS SEEN IN THIS STORE:\n" + "; ".join(labels[:60])) if labels else ""

    def check_key(self, ids: list[str]) -> str:
        return "|".join(f"{i}x{self.cart_lines.get(i, 0)}" for i in sorted(ids)) + f"#{self.version}.{self.sweep_mark}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None, models: list[str] | None = None,
                   extra: list[str] | None = None) -> str:
        """One check of the cart (or of planned lines `ids`). `extra`: listings to show as OTHER LISTINGS, so the check
        can name them in "better". Returns the problems for the walker, or "" when the cart passed or no check ran."""
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        limit = self.soft if limit is None else limit
        self.cheaper_fits = []
        if key in self.checked or self.cost > max(limit, self.hard) or not ids:
            return ""
        chosen = models
        models = list(chosen) if chosen else ([HEAVY, LIGHT] if self.cost <= limit else [LIGHT])
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if self.changes:
            prompt += ("\n\nCHANGES DURING THE EPISODE (the requirements above already apply them; where they differ from the request,"
                       " they win):\n- " + "\n- ".join(self.change_lines()))
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        block, cheaper_open = self.cheaper_block(ids) if self.brief.get("pick") == "cheapest" and not lead else ("", set())
        if block:
            prompt += block
        elif self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        elif (self.changes and len(self.changes) not in self.others_shown
              and (any(c.get("kind") == "drop" or c.get("way") == "raised" for c in self.changes[max(self.others_shown, default=0):])
                   or not self.inside_budget({i: self.cart_lines.get(i, 1) for i in ids}))):
            # the first check after a loosening (a raised budget, a dropped requirement) or with the cart over a new
            # budget weighs the other listings too: one may fit now that the cart's choice did not beat before (the heavy
            # model only inside the soft spend)
            self.others_shown.add(len(self.changes))
            prompt += ('\n\nOTHER LISTINGS SEEN (newest first; after the changes above, name in "better" any that meets every requirement'
                       " in force when a cart line does not):\n" + self.others(ids, 6))
            if not chosen:
                models = [HEAVY, LIGHT] if self.cost <= min(limit, self.soft) else [LIGHT]
        try:
            prompt += self.fit_block(ids)
        except Exception as error:      # the code lines are context: a fault in them never stops the check
            self.trace.append({"kind": "fit_error", "error": repr(error)[:200]})
        shown = [i for i in dict.fromkeys(extra or []) if i not in ids and i in self.rows and i not in prompt]
        if shown:
            prompt += ('\n\nOTHER LISTINGS SEEN (of the items the cart gets wrong; name in "better" any that meets every requirement'
                       " in force when a cart line does not):\n" + "\n".join("OPENED " + self.review_record(i, 600, 100) for i in shown[:10]))
        prompt += self.departments()
        if not lead:
            units = "; ".join(f"{i}: {self.cart_lines.get(i, 1)} unit(s)" for i in ids)
            prompt += "\n\nUNITS PER CART LINE (the store's own cart reply when it listed one, else Kyo's record):\n" + units
            prompt += "\nREQUESTED QUANTITIES: " + "; ".join(f"item {n} ({it.name}): {it.qty}" for n, it in enumerate(self.items, 1))
        got = self.ask_json(prompt, models, 1500)
        heavy = self.answered_by == HEAVY
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        if got and cheaper_open:
            named = [str(b) for b in _items(got.get("cheaper_fits")) if isinstance(b, (str, int)) and not isinstance(b, bool)]
            fits = [b for b in dict.fromkeys(named) if b in cheaper_open and b not in ids and self.buyable(b)]
            # code acts on them only when the heavy model itself judged the cart wrong; otherwise they are only traced
            if fits and heavy and got.get("ok") is False:
                self.cheaper_fits = fits[:4]
            if named:
                self.trace.append({"kind": "cheaper_fits", "named": named[:6], "kept": self.cheaper_fits, "heavy": heavy})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if got:
            self.keeps[key] = _items(got.get("keep"))
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        self.betters[key] = better
        self.pointed += [b for b in better if not self.was_opened(b) and b not in self.pointed]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def fit_record(self, ident: str) -> dict[str, Any]:
        record = self.rows.get(ident) or {}
        return {**record, **(self.full.get(ident) or {})}

    def fit_lines(self, ident: str) -> list[str]:
        """The listing's own sentences that carry a code (its texts' first 600 characters, attribute values), at most 6;
        a fragment under 12 characters is joined to the next one."""
        record = self.fit_record(ident)
        texts = [str(v)[:600] for v in record.values() if isinstance(v, str) and v not in self.rows]
        texts += [f"{k} {v}" for b in record.values() if isinstance(b, dict) for k, v in b.items() if isinstance(v, str)]
        found: list[str] = []
        for text in texts:
            merged: list[str] = []
            for part in re.split(r"(?<=[.;!?])\s+|\n", text):
                if merged and len(merged[-1]) < 12:
                    merged[-1] += " " + part
                else:
                    merged.append(part)
            for sentence in (x.strip() for x in merged[:12]):
                if sentence and len(sentence) <= 220 and _code_set(sentence) and sentence not in found:
                    found.append(sentence)
        return found[:6]

    def own_codes(self, ident: str) -> set[str]:
        """The listing's own model or part codes: a bracketed code ending one of its texts, and the code of each of the
        first two sentences of its longest text that names one code only (a list names other listings' codes)."""
        texts = [v for v in self.fit_record(ident).values() if isinstance(v, str) and v not in self.rows]
        tails = [m.group(1) for t in texts for m in [re.search(r"[\[(]([^\[\]()]{3,24})[\])]\s*$", t)] if m]
        first = re.split(r"(?<=[.;!?])\s+", max(texts, key=len, default=""))[:2]
        own: set[str] = set()
        for sentence in tails + first:
            if len(_code_set(sentence)) == 1:
                own |= _code_set(sentence, strict=True)
        return own

    def cross_lines(self, ids: list[str], cap: int = 8) -> list[str]:
        """Sentences on other seen listings that name one of these listings' own codes together with another code, a
        family list excepted: the rest of a list a page leaves to another part's page."""
        codes = set().union(*[self.own_codes(i) for i in ids]) if ids else set()
        found: list[str] = []
        if not codes:
            return found
        for other in reversed(self.seen_order):
            if other in ids or len(found) >= cap:
                continue
            for text in [str(v)[:600] for v in self.fit_record(other).values() if isinstance(v, str) and v not in self.rows]:
                for sentence in re.split(r"(?<=[.;!?])\s+", text)[:6]:
                    named = _code_set(sentence)
                    if not FAMILY_CODE.search(sentence) and codes & named and len(named) >= 2:
                        found.append(f"{other}: {sentence.strip()[:200]}")
        return found[:cap]

    @staticmethod
    def code_linked(lines: list[str]) -> bool:
        """Some sentence names two codes or a family: the lines state a compatibility, not just a model number."""
        return any(len(_code_set(s)) >= 2 or FAMILY_CODE.search(s) for s in lines)

    def fit_read(self, device: str, number: int, line: str) -> dict[str, Any]:
        """The light model's reading of whether item `number`'s cart line and its department's other seen buyable listings
        fit the device line, from their code sentences. Asked only when the device names a model or part code and a shown
        sentence links two codes or names a family; at most 3 calls per task, none past 200 s or above the hard spend, and
        a prompt already asked is answered from its cached verdict. A 'not' whose reason names no code of the shown lines
        counts as unknown. Returns {fits, not, unknown, why, read} with candidate ids only, or {}."""
        if not 1 <= number <= len(self.items) or not self.items[number - 1].fits or line == device:
            return {}
        item = self.items[number - 1]
        own = self.fit_lines(device)
        if not any(STRICT_CODE.search(s) for s in own):
            return {}
        cat_key = self.category_key()
        pool = [i for i in reversed(self.seen_order) if i not in (line, device) and self.buyable(i) and self.dept_match(number, i, cat_key)
                and not self.dept_match(item.fits, i, cat_key, True) and self.fit_lines(i)]
        cands = ([line] if self.fit_lines(line) else []) + pool[:11]
        if not cands:
            return {}
        cross = self.cross_lines([device] + cands)
        shown = own + [s for i in cands for s in self.fit_lines(i)] + [c.split(": ", 1)[-1] for c in cross]
        if not self.code_linked(shown):
            return {}
        prompt = (FITREAD + f"\n\nITEM THAT MUST FIT THE DEVICE: {item.name}\nDEVICE ITEM: {self.items[item.fits - 1].name}"
                  + f"\n\nDEVICE {device}: " + " | ".join(own)
                  + "".join(f"\nCANDIDATE {i}: " + " | ".join(self.fit_lines(i)) for i in cands)
                  + ("\n\nLINES ON OTHER LISTINGS' PAGES:\n" + "\n".join(cross) if cross else ""))
        if prompt in self.fit_seen:
            return self.fit_seen[prompt]
        if self.fit_calls >= 3 or self.cost > self.hard or time.time() - self.t0 > 200:
            return {}
        self.fit_calls += 1
        got = self.ask_json(prompt, [LIGHT], 700) or {}

        def named(key: str) -> list[str]:
            values = [str(v).strip() for v in _items(got.get(key)) if isinstance(v, (str, int)) and not isinstance(v, bool)]
            return [v for v in dict.fromkeys(values) if v in cands]

        codes = set().union(*[_code_set(s) | {c.upper() for c in SERIES_CODE.findall(s)} for s in shown])
        why = got.get("why") if isinstance(got.get("why"), dict) else {}
        why = {str(k).strip(): str(v)[:120] for k, v in why.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        fits = named("fits")
        ruled = [i for i in named("not") if i not in fits]
        bad = [i for i in ruled if _code_set(why.get(i, "")) & codes]
        unknown = [i for i in dict.fromkeys(named("unknown") + ruled) if i not in fits and i not in bad]
        read = [str(c).strip() for c in _items(got.get("read")) if isinstance(c, str) and c.strip()][:2]
        verdict = {"fits": fits, "not": bad, "unknown": unknown, "why": {i: why[i] for i in bad + fits if i in why}, "read": read}
        self.trace.append({"kind": "fit", "device": device, "item": number, "line": line, "got": _clip(verdict, 120)})
        self.fit_seen[prompt] = verdict
        self.fit_pairs[f"{device}|{line}"] = verdict
        return verdict

    def fit_note(self) -> str:
        """After a cart change, for each cart pair the request says must fit: Kyo's reading of the codes, told once per pair
        when it rules the line out. Advisory: the walker gets the code lines and decides. An unknown pair first opens the
        page of the part the reading names, when steps allow. A line in the device's own department is never paired."""
        if not any(item.fits for item in self.items):
            return ""
        per = self.item_lines()
        cat_key = self.category_key()
        notes: list[str] = []
        for number, item in enumerate(self.items, 1):
            device, line = per.get(item.fits), per.get(number)
            if not item.fits or not device or not line or device == line or self.dept_match(item.fits, line, cat_key, True):
                continue
            mark = f"{device}|{line}"
            if mark in self.fit_told:
                continue
            got = self.fit_read(device, number, line)
            if line in got.get("unknown", []) and got.get("read"):
                codes = {re.sub(r"[\s-]", "", c).upper() for c in got["read"]}
                rows = [i for i in self.seen_order if i not in self.opened and codes & self.own_codes(i)][:2]
                if rows and self.left() > self.tail() + len(rows) + 3:
                    self.page_first(rows, self.tail() + 3)
                    got = self.fit_read(device, number, line)
            if line not in got.get("not", []):
                continue
            self.fit_told.add(mark)
            good = [i for i in got.get("fits", []) if i != line][:4]
            why = got.get("why", {}).get(line)
            codes = " / ".join(f"{i}: " + " | ".join(self.fit_lines(i)[:3]) for i in (device, line))
            notes.append(f"{line} (item {number}) does not fit {device} (item {item.fits}) by the codes" + (f" ({why})" if why else "")
                         + f"; code lines: {codes}; "
                         + (f"item {number} listings already seen that the reading counts as fitting: {', '.join(good)}" if good
                            else f"no item {number} listing seen so far fits by the codes: browse that department and compare the codes its rows show, or switch item {item.fits} to a listing they fit"))
        return ("[Kyo's reading of the codes, advisory: check the code lines yourself] " + " | ".join(notes) + ".") if notes else ""

    def fit_block(self, ids: list[str]) -> str:
        """The cart check's view of the codes when the request says one item must fit another: the cart lines' code
        sentences, other pages' sentences about their codes, and Kyo's reading of the pairs in this cart."""
        if not any(item.fits for item in self.items):
            return ""
        lines = {i: self.fit_lines(i) for i in ids}
        cross = self.cross_lines(ids)
        if not any(STRICT_CODE.search(x) for v in lines.values() for x in v) or not self.code_linked([x for v in lines.values() for x in v] + cross):
            return ""
        said = [f"{pair}: " + _dump(_clip(got, 80))[:300] for pair, got in self.fit_pairs.items() if got and all(p in ids for p in pair.split("|"))]
        return ("\n\nCODE LINES OF THE CART'S LISTINGS (the request says one item must fit another; codes match whatever their spaces,"
                " dashes and case, a family written with x's covers every code that begins the same way less the codes it excludes, a"
                " listing's other names are its codes too, and what another listing's page says about one of their codes counts as said"
                " on their own pages):\n" + "\n".join(f"{i}: " + " | ".join(v) for i, v in lines.items() if v)
                + ("\nLINES ON OTHER LISTINGS' PAGES ABOUT THEIR CODES:\n" + "\n".join(cross) if cross else "")
                + ("\nKYO'S READING OF THE CODES (advisory): " + " | ".join(said[-2:]) if said else ""))

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
        while pending and self.trails < 4 and self.cost < self.hard:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + self.review_record(ident, 1400, 300, record))
                else:
                    records.append("ROW " + self.review_record(ident, 360, 80, self.rows.get(ident) or {}))
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
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def chain_due(self) -> bool:
        live = {need for _, need in self.needs_now()}
        return any(need in live and need not in self.ref_values and need not in self.chain_tried for _, need in self.refs)

    def resolve_refs(self) -> str:
        """Code follows each requirement whose value sits on another listing, once, while steps and time allow."""
        live = {need for _, need in self.needs_now()}
        done: list[str] = []
        lost: list[str] = []
        for number, need in list(self.refs):
            if need not in live or need in self.ref_values or need in self.chain_tried:
                continue
            self.chain_tried.add(need)
            if self.finished or self.left() < 8 or time.time() - self.t0 > 200 or self.cost > self.hard - 0.01:
                break
            got = self.follow_chain(number, need)
            (done if got else lost).append(got or need)
        self.trace.append({"kind": "chains", "done": done, "lost": lost[:3]})
        text = ""
        if done:
            text += ("[Kyo followed the listings these requirements point at and opened every page of each chain] " + " | ".join(done)
                     + ". Shop with these values now; the chain is done and needs no further opening.")
        if lost:
            text += ("\n[chains Kyo could not finish] " + " | ".join(lost[:3]) + ". Find the listing each points at by its shop and kind, open its page,"
                     " follow every page that refers to a further listing; the last page gives the value.")
        return text.strip()

    def note_mode(self, tool: str, args: dict[str, Any]) -> None:
        """A walker search that carried free words and showed rows: its mode arguments (those with allowed values) are the
        ones Kyo's own word searches reuse."""
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)

        def listed(k: str) -> bool:
            return isinstance((props.get(k) or {}).get("enum"), list) and args[k] in props[k]["enum"]

        words = args.get(target) if target else None
        # a word search: its words, values from allowed lists and a paging number, no second free text
        extra = [k for k, v in args.items() if k != target and not listed(k) and not (isinstance(v, int) and not isinstance(v, bool))]
        if isinstance(words, str) and re.search(r"[A-Za-z]", words) and not extra:
            self.word_mode = {k: args[k] for k in args if k in required and listed(k)}
        elif not (isinstance(words, str) and words.strip()):
            # a listing without free words (a department browse): its mode is not the one a word search takes
            for k in args:
                if k in required and listed(k):
                    self.plain_modes.add((k, str(args[k])))

    def mode_for(self, tool: str, key: str) -> Any:
        """The value for a required mode argument of a Kyo search: the walker's last word search's, else the first allowed
        value the walker never used for a listing without free words, else the first allowed."""
        options = ((self.schemas.get(tool) or {}).get("properties") or {}).get(key, {}).get("enum") or [None]
        if tool != self.roles.get("search"):
            return options[0]
        value = self.word_mode.get(key)
        if value in options:
            return value
        return next((o for o in options if (key, str(o)) not in self.plain_modes), options[0])

    def chain_search(self, words: str) -> bool:
        """One search of Kyo's own for the listing a chain names; True when the result showed listing rows."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not tool or not target or not words.strip() or self.left() < 5 or self.finished:
            return False
        args: dict[str, Any] = {target: words.strip()}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = self.mode_for(tool, key)
                else:
                    return False
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return False
        reply = self.act(tool, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_view(tool, seen)
        rows = self.records_in(seen)
        self.last_search, self.last_seen = args, seen
        self.last_hits = self.note_hits(tool, args, seen) if rows else None
        self.trace.append({"kind": "chain_search", "words": words, "rows": len(rows)})
        return bool(rows)

    def open_both(self, ident: str, facts: bool = True) -> None:
        """Open a chain listing's page and (with `facts`) its attributes, each view once: every page of a chain must be opened."""
        page = self.roles.get("offer") or self.roles.get("open")
        for tool in (page, self.facts_tool() if facts else None):
            if not tool or self.left() < 4 or self.finished:
                continue
            if (tool == page and ident in self.opened) or (tool != page and ident in self.full):
                continue
            target = self.single_target(tool)
            if not target:
                continue
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                return
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if tool != page and isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))

    def read_link(self, text: str) -> dict[str, Any] | None:
        """A chain link's text read by LINK, once per text: its shop, kind, picking properties and values, the property
        and the relation; None when it points at no further listing."""
        if text in self.link_reads:
            return dict(self.link_reads[text]) if self.link_reads[text] else None
        got = self.ask_json(LINK + "\n\nTEXT:\n" + text[:1500], [LIGHT, HEAVY], 500) or {}
        seller = re.sub(r"'s$", "", str(got.get("seller") or "").strip()).strip()
        self.link_reads[text] = None
        if not seller:
            return None
        link = {"seller": seller, "kind": str(got.get("kind") or "").strip(), "with": [str(w).strip() for w in _items(got.get("with")) if str(w).strip()][:6],
                "values": [str(v).strip() for v in _items(got.get("values")) if isinstance(v, (str, int, float)) and str(v).strip()][:6],
                "property": str(got.get("property") or "").strip(), "relation": str(got.get("relation") or "same").strip().lower(),
                "margin": str(got.get("margin") or "").strip()}
        self.link_reads[text] = link
        return dict(link)

    def shop_rows(self, seller: str, skip: list[str]) -> list[str]:
        """Listings whose record names this shop (any field equal to the name), newest first."""
        name = seller.lower()
        return [i for i in reversed(self.seen_order) if i not in skip
                and any(isinstance(v, str) and v.strip().lower() == name for v in (self.rows.get(i) or {}).values())]

    def find_link(self, link: dict[str, Any], prop: str, skip: list[str]) -> tuple[str, str, str] | None:
        """The listing a link describes, opened in full: (id, the property's value or "", the words sending it on or "");
        a pairing rule its page states for the property is kept in self.chain_rules."""
        kind_words = {w for w in _toks(link["kind"], True) if len(w) >= 3}

        def of_kind(i: str) -> bool:
            return not kind_words or bool(kind_words & _toks(_dump(self.rows.get(i) or {})[:1500], True))

        # the picking values alone (a figure glued to a unit of 2+ letters split off, so it matches the figure as pages
        # print it; a one-letter tail such as 4k or 5g is part of the name)
        values = [re.sub(r"\b(\d+(?:[.,]\d+)?)[A-Za-z]{2,}\b", r"\1", v) for v in link.get("values") or []]
        first = " ".join(x for x in [link["seller"]] + (values or [link["kind"]] + list(link["with"])) if x)
        # the shop with the values first (they pick the listing out), then with the first value, then shop and kind, then
        # the shop alone; a later query only once the candidates of the ones before are used up
        queue = list(dict.fromkeys(q for q in (first, " ".join(x for x in (link["seller"], values[0] if values else "") if x),
                                                " ".join(x for x in (link["seller"], link["kind"]) if x), link["seller"]) if q.strip()))
        fresh: list[str] = []
        value_words = _stem(" ".join(values or link["with"]))

        def run_next() -> bool:
            # the next query not run for a link yet; True when it showed rows of that shop
            while queue and not self.finished and self.left() >= 6 and self.step < self.chain_stop:
                query = queue.pop(0)
                if query.lower() in self.link_searched:
                    continue
                self.link_searched.add(query.lower())
                if not self.chain_search(query):
                    continue
                shop = set(self.shop_rows(link["seller"], skip))
                got = [i for i, _ in self.records_in(self.last_seen) if i in shop]
                if got:
                    fresh[:] = list(dict.fromkeys(got))
                    return True
            return False

        def rank(i: str) -> tuple[int, int, int, bool]:
            # the rows of the query just run in store order, then the words shared with the values, then the kind
            return (i not in fresh, fresh.index(i) if i in fresh else 0, -len(value_words & _stem(_dump(self.rows.get(i) or {})[:1500])), not of_kind(i))

        if link["with"] or values or not [i for i in self.shop_rows(link["seller"], skip) if of_kind(i)]:
            run_next()
        asked_again = False
        spent = 0               # pages opened for this link (LINK_OPENS at most before it is given up)
        for _ in range(8):
            ids = sorted(self.shop_rows(link["seller"], skip), key=rank)[:12]
            if not ids and run_next():
                ids = sorted(self.shop_rows(link["seller"], skip), key=rank)[:12]
            if not ids or self.finished:
                return None
            # the best unread candidates are opened before PICK reads them, keeping the finish
            ahead = [i for i in ids if i not in self.opened and (i in fresh or of_kind(i))][:1]
            if (self.left() > self.tail() + 6 and time.time() - self.t0 < 215 and self.step < self.chain_stop + 2
                    and not any(i in self.opened for i in ids)):
                for ident in ahead:
                    self.open_both(ident, facts=False)
                    spent += 1
            lines = [("OPENED " if (i in self.opened or i in self.full) else "ROW ") + i + " " + _dump(_clip(self.rows.get(i), 700))[:2400] for i in ids]
            prompt = PICK + "\n\nLINK: " + _dump(link) + "\nPROPERTY: " + prop + "\n\nRECORDS:\n" + "\n".join(lines)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
            ident = str(got.get("id") or "").strip()
            if ident in ids and (ident in self.opened or ident in self.full):
                value, onward = str(got.get("value") or "").strip(), str(got.get("next") or "").strip()
                pairs = str(got.get("pairs") or "").strip()
                if pairs and pairs not in self.chain_rules:
                    self.chain_rules.append(pairs)
                if ident not in self.opened and self.left() > 4:
                    # its page is a link of the chain and may send the property on: open it and read again
                    self.open_both(ident, facts=False)
                    spent += 1
                    continue
                if not value and not onward and ident not in self.full and self.left() > 4:
                    self.open_both(ident)
                    spent += 1
                    continue
                if value and not onward:
                    onward = self.onward_guard(ident, prop, link["seller"])
                return ident, value, onward
            unread = [i for i in ids if i not in self.opened]
            opens = [str(x) for x in _items(got.get("open")) if str(x) in unread]
            if not opens:
                # unread candidates of that shop and kind are opened before the link is given up, inside its budget
                opens = sorted([i for i in unread if of_kind(i) or i in fresh], key=rank)
            if not opens and time.time() - self.t0 < 215 and run_next():
                continue
            if not opens and not asked_again and "id" not in got and time.time() - self.t0 < 215:
                # a reply that is no reading at all (a failed call, junk) is asked once more before the link is given up
                asked_again = True
                continue
            if (not opens or self.left() < 6 or time.time() - self.t0 > 215 or self.step >= self.chain_stop + 2
                    or spent >= LINK_OPENS):
                # (a search already run still gets its best rows opened up to 2 steps past the stop)
                return None
            for ident in opens[: min(2 if self.left() > 12 else 1, LINK_OPENS - spent)]:
                self.open_both(ident, facts=False)
                spent += 1
        return None

    def onward_guard(self, ident: str, prop: str, seller: str) -> str:
        """A page whose value PICK took with no onward words: its prose sentences holding every word of the property and a
        capitalised name other than its own shop or brand are read as a link; when they name a different shop, they are
        the onward words. "" otherwise (one light call at most)."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        own = {str(v).strip().lower() for v in record.values() if isinstance(v, str) and 0 < len(v.strip()) <= 40} | {seller.lower()}
        words = _stem(prop)
        if not words or self.finished or self.left() < 8 or time.time() - self.t0 > 205 or self.cost > self.hard:
            return ""
        texts = [v for v in record.values() if isinstance(v, str) and len(v) > 40]
        block = _fact_block(record) or {}
        texts += [v for v in block.values() if isinstance(v, str) and len(v) > 40]
        found: list[str] = []
        for text in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if not words <= _stem(sentence):
                    continue
                names = re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]+(?: [A-Z][a-z]+)*", sentence)
                if any(n.lower() not in own and not any(n.lower() in o for o in own) for n in names) and sentence not in found:
                    found.append(sentence.strip())
        if not found:
            return ""
        text = " ".join(found)[:800]
        link = self.read_link(text)
        if not link or link["seller"].lower() in own or link["seller"].lower() not in text.lower():
            return ""
        self.trace.append({"kind": "onward_guard", "id": ident, "seller": link["seller"]})
        return text

    def pair_value(self, prop: str, value: str) -> str:
        """The value a store pairing rule matches with `value` (rules from the chain's pages, else any opened page naming
        the property next to the word 'with')."""
        rules = list(self.chain_rules)
        if not rules:
            words = _stem(prop)
            for record in self.rows.values():
                for text in (record.get(k) for k in record if isinstance(record.get(k), str) and len(record.get(k)) > 40):
                    if " with " in text and words and words <= _stem(text):
                        rules.append(text[:400])
            rules = rules[:6]
        if not rules:
            return ""
        got = self.ask_json(PAIRING + f"\n\nPROPERTY: {prop}\nVALUE: {value}\nRULES:\n" + "\n".join(rules), [LIGHT, HEAVY], 300) or {}
        return str(got.get("value") or "").strip()

    def follow_chain(self, number: int, need: str) -> str:
        """Follow one requirement's chain to the page that states the value; returns a line for the walker, or "". The
        chain stops at chain_stop: at most half the steps beyond the finish, 6 to 14."""
        link = self.read_link(need)
        if not link:
            return ""
        self.chain_stop = self.step + max(6, min(14, (self.left() - self.tail()) // 2))
        prop = link["property"] or need.split(":")[0].strip()
        relation = link["relation"]
        margin = link.get("margin") or ""
        path: list[str] = []
        spare = ""
        for hop in range(7):
            if self.finished or self.left() < 6 or time.time() - self.t0 > 215 or self.cost > self.hard or self.step >= self.chain_stop:
                break
            if hop >= 5 and self.left() <= self.tail() + 10:
                break
            found = self.find_link(link, prop, path)
            value = ""
            if found:
                ident, value, onward = found
                path.append(ident)
                if onward:
                    nxt = self.read_link(onward)
                    if nxt:
                        # the next link (a listing already on the path is never picked again)
                        link = dict(nxt, property=prop)
                        spare = value or spare
                        continue
            value = value or spare
            if value and relation == "compatible":
                paired = self.pair_value(prop, value)
                if paired:
                    value = paired
            if value:
                shown = {"different": f"anything but {value}", "at least": f"at least {value}", "at most": f"at most {value}",
                         "within": f"within {margin} either way of {value}" if margin else value}.get(relation, value)
                self.ref_values[need] = shown
                self.chain_paths[need] = path
                self.version += 1
                self.trace.append({"kind": "chain", "need": need[:200], "path": path, "value": shown})
                return f"item {number}: {prop}: {shown} (pages opened: {' -> '.join(path)})"
            break
        self.chain_paths[need] = path
        self.trace.append({"kind": "chain_lost", "need": need[:200], "path": path})
        return ""

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
        if stale and self.left() < len(stale) + self.tail() + 2:
            # few steps: only the lines whose page was never opened (an order holding one fails), keeping the finish and
            # one step for a turned-down checkout's retry
            stale = [i for i in stale if i not in self.opened]
            if self.left() < len(stale) + self.tail() + 1:
                stale = []
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

    def adds_of(self, name: str, args: dict[str, Any]) -> list[str]:
        """The listing ids a cart call puts in: none for a take-off mode or a count of zero or less; basket lists of lines
        and plain ids are both read."""
        spec = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            options = spec[key].get("enum") if isinstance(spec.get(key), dict) else None
            if isinstance(options, list) and len(options) >= 2 and value == self.mode_values(name, key, options)[1]:
                return []
        loose = next((_count(v) for v in args.values() if not isinstance(v, (str, list, dict)) and _count(v) is not None), None)
        found: list[str] = []
        for value in args.values():
            for line in (value if isinstance(value, list) else [value]):
                if isinstance(line, dict):
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    count = next((_count(v) for v in line.values() if not isinstance(v, str) and _count(v) is not None), 1)
                    if ident and count > 0:
                        found.append(ident)
                elif isinstance(line, str) and line in self.rows and (loose is None or loose > 0):
                    found.append(line)
        return list(dict.fromkeys(found))

    def page_first(self, ids: list[str], keep: int) -> list[str]:
        """Open the page view (the offer view, else the single-listing view) of listings about to be carted or ordered that
        no call has shown with it, keeping `keep` steps: in stores with a separate page view an order holding such a line
        nearly never passed. No page opens past OPEN_END. Returns the changes those pages carried."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        todo = [i for i in dict.fromkeys(ids) if i in self.rows and i not in self.opened and i not in self.gone]
        if not tool or not target or not todo:
            return []
        said: list[str] = []
        done: list[str] = []
        for ident in todo:
            # the open and the checkout after it must both land before HARD_END-5 s at the slowest latency seen
            if (self.left() <= keep or self.finished or time.time() - self.t0 > OPEN_END
                    or time.time() - self.t0 + 2 * self.store_lat > HARD_END - 5):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            done.append(ident)
            seen = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.note_view(tool, seen)
            change = self.observe(tool, seen)
            if change:
                said.append(change)
        self.trace.append({"kind": "page_first", "ids": done, "said": said[:3]})
        return said

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
        if self.cost > self.soft:
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(self.review_record(ident, 1200, 200))
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

    def cart_total(self) -> float | None:
        """The cart's total by the listings' newest prices (the price key); None when a line's price is unknown."""
        total = 0.0
        for ident in dict.fromkeys(self.cart_ids()):
            price = self.price_of(ident)
            if price is None:
                return None
            total += price * self.cart_lines.get(ident, 1)
        return total

    def after_change(self) -> str:
        """Once per change, in code, before the next checkout: a cart whose total breaks the budget in force is named with
        the line to replace; after a raise, when an item has no cart line, the cart holds a closest-listing pick, or the
        pick is the most or least of a measure, the unread rows of the items' departments priced inside the new room are
        read. Returns what it found, or ""."""
        if not self.change_due or self.finished:
            return ""
        self.change_due = False
        recent = [c for c in self.changes if c["step"] >= self.change_from]
        self.change_from = self.step + 1
        said: list[str] = []
        ids = list(dict.fromkeys(self.cart_ids()))
        total = self.cart_total()
        if isinstance(self.budget, (int, float)) and total is not None and ids and self.brief.get("budget_scope") != "each" and total > self.budget:
            dear = max(ids, key=lambda i: (self.price_of(i) or 0.0) * self.cart_lines.get(i, 1))
            said.append(f"The cart's total {_num(total)} is over the budget {_num(self.budget)} now: replace {dear} (or another line) by a fitting listing that brings the total inside it.")
        if (self.window and self.price_key and (self.cart_short() or self.trusted or self.brief.get("pick") in ("most", "least"))
                and self.left() > 6 + len(self.items) and time.time() - self.t0 < 200):
            low, high = self.window
            self.window = None
            cat_key = self.category_key()
            rows = [i for i in reversed(self.seen_order) if i not in self.full and self.buyable(i) and low < (self.price_of(i) or 0.0) <= high
                    and (not cat_key or any(self.dept_match(n, i, cat_key) for n in range(1, len(self.items) + 1)))]
            if self.brief.get("pick") == "cheapest":
                rows.sort(key=lambda i: self.price_of(i) or 0.0)
            read = self.fill_facts(rows[: 2 * len(self.items)], self.tail() + 4 + len(self.items))
            if read:
                said.append("Kyo read the unread listings the higher budget lets in:\n" + "\n".join(read))
        self.trace.append({"kind": "after_change", "changes": recent[-4:], "said": [t[:120] for t in said]})
        return "\n".join(said)

    def after_refusal(self) -> str:
        """The store turned down an ending without a purchase with a change: the change is applied in code, the records are
        audited under the requirements in force now, and when the audit names a listing for every item and a fresh check
        of those picks passes, Kyo carts them and checks out. Under a cheapest pick with steps to spare the walker gets
        the listings and the pick rule instead. Returns what was done, for the walker, or ""."""
        if (self.finished or self.left() < len(self.items) + 3 or self.cost > self.hard or time.time() - self.t0 > 215
                or not self.roles.get("cart") or not self.roles.get("buy")):
            return ""
        found = self.after_change()
        got = self.audit_call(light=self.cost > self.soft)
        self.audited = True
        named = self.audit_fits(got)
        if len(named) < len(self.items):
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.full and not self.sold_out(str(i))]
            if rows and self.left() > len(self.items) + 5 and self.open_rows(rows):
                named = self.audit_fits(self.audit_call(light=True))
        self.trace.append({"kind": "after_refusal", "named": named})
        if len(named) < len(self.items):
            return found
        self.lead_items, self.leads = named, [i for ids in named.values() for i in ids]
        listed = "; ".join(f"item {k}: {', '.join(v)}" for k, v in sorted(named.items()))
        if self.brief.get("pick") == "cheapest" and self.left() >= 10 + 2 * len(self.items):
            self.rearm()
            return (found + "\n[Kyo] Under the requirements in force now a review found listings that fit: " + listed
                    + ". The pick is the cheapest: cart the cheapest listing of each item that meets every requirement in force, then check out.").strip()
        picks = [ids[0] for _, ids in sorted(named.items())]
        self.check_cart(picks, self.soft)
        if not self.verdicts.get(self.check_key(picks)):
            return found
        self.confirmed = picks[0]
        self.completed = False
        return (found + "\n" + self.complete_cart()).strip()

    def run_cart(self, calls: list[dict[str, Any]], until: float | None = None) -> list[str]:
        """Send cart calls Kyo planned, keeping the ledger in step with the store's replies. A call that cannot be shaped
        or fails stops the rest, and so does the clock passing `until` seconds. Returns the calls sent."""
        cart = self.roles.get("cart") or ""
        done: list[str] = []
        for args in calls:
            if until is not None and time.time() - self.t0 >= until:
                break
            # a call planned for the store's removal tool names that tool
            tool = str(args.get(TOOL_KEY) or cart)
            args = {k: v for k, v in args.items() if k != TOOL_KEY}
            faults: list[str] = []
            args = _shape(args, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.cart_lines)
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            if tool == cart:
                self.note_cart(cart, args)
                self.learn_cart(cart, args, reply.get("observation"), before)
                other = self.mode_retry(cart, args, reply.get("observation"), before)
                if (other is None and self.empty_unsure and self.roles.get("cart_view")
                        and self.mode_other(cart, args, before) is not None):
                    # the reply's empty container may be the cart or a part of an acknowledgement: the cart view says
                    # whether the line went in
                    seen = self.look_cart()
                    if seen is not None:
                        other = self.mode_retry(cart, args, seen, before, view=True)
                if other is not None:
                    # the store's own cart lacks the line this call put in: the mode value did not put; one resend with the
                    # other value (cart_moved learns from its reply)
                    before = dict(self.cart_lines)
                    reply = self.act(tool, other)
                    if reply is None or self.finished:
                        break
                    self.note_cart(cart, other)
                    self.learn_cart(cart, other, reply.get("observation"), before)
                    args = other
            else:
                self.took_off(tool, args, reply.get("observation"))
            self.observe(tool, reply.get("observation"))
            done.append(_dump(args) if tool == cart else tool + " " + _dump(args))
        return done

    def swap_better(self, ids: list[str] | None = None, need_change: bool = True, why: str = "after the change", mode: str = "after",
                    hold: set[str] | frozenset[str] = frozenset(), until: float | None = None) -> tuple[bool, str]:
        """Code puts a better listing in place of one cart line. By default the candidates are the listings the last
        failed check of this cart named, and only after a change of the task; a caller may pass its own
        (need_change=False). A candidate must be opened, buyable, inside the budget and of the same department label as
        exactly one cart line, and cheaper than that line under a cheapest pick. At most two swaps per episode, with the
        checkout's steps kept. mode "after": the cart is checked again after the swap. mode "before" (the cheaper swap):
        the candidate must state at least as many of the item's worded values as the line, no swap past DEADLINE-30 s,
        and a check of the planned cart must pass before any cart call. mode "final" (the final review): no check, a
        dearer candidate may replace a wrong line, lines in `hold` stay. In both, no swap spends an order approval that
        can no longer be asked again. mode "repair" (a repeated checkout of a cart the check found wrong): every line the
        check did not keep may take one candidate of its own item's department (dept_match) stating at least as many of
        the item's worded values, a dearer one too, at the item's quantity; the whole planned cart must keep the budget
        and pass a check of the planned cart (heavy model only inside the soft spend, none past LATE-8 s) before any cart
        call. Cart calls stop at `until` seconds. Returns (swapped and, in mode "after", the new check passed; what was
        done)."""
        now = list(dict.fromkeys(self.cart_ids()))
        if ids is None:
            key = self.check_key(now)
            ids = self.betters.get(key, []) if self.verdicts.get(key) is False else []
        better = [b for b in dict.fromkeys(ids) if self.was_opened(b) and b in self.rows and self.buyable(b) and b not in now]
        cat_key = self.category_key()
        if (not self.roles.get("cart") or not better or not cat_key or (need_change and not self.changes) or self.swaps >= 2
                or self.finished):
            return False, ""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        if mode != "after" and talk and self.msg_used >= self.msg_limit and not self.gate_due("order"):
            return False, ""
        if mode == "before" and time.time() - self.t0 > DEADLINE - 30:
            return False, ""
        lines_of = {ident: number for number, ident in self.item_lines().items()}

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()

        if mode == "repair":
            if time.time() - self.t0 > LATE - 8:
                return False, ""
            kept = {str(e.get("id")) for e in self.keeps.get(self.check_key(now)) or [] if isinstance(e, dict)}
            pairs: dict[str, str] = {}
            for new in better:
                same = [i for i in now if kind(i) and kind(i) == kind(new) and i not in kept and i not in pairs]
                if len(same) != 1 or same[0] in hold or new in pairs.values():
                    continue
                old = same[0]
                number = lines_of.get(old) or (1 if len(self.items) == 1 else 0)
                if not number or not self.dept_match(number, new, cat_key):
                    continue
                values = self.worded_values(number)
                if self.screen(new, values) < self.screen(old, values):
                    continue
                pairs[old] = new
            if not pairs:
                return False, ""
            lines = {i: self.cart_lines.get(i, 1) for i in now if i not in pairs}
            lines.update({new: self.items[(lines_of.get(old) or 1) - 1].qty for old, new in pairs.items()})
            if not self.inside_budget(lines):
                self.trace.append({"kind": "swap_better", "pairs": pairs, "refused": "budget", "mode": mode})
                return False, ""
            calls = self.cart_calls(lines, list(pairs))
            if not calls or len(calls) + self.tail(True) + 1 > self.left():
                return False, ""
            self.check_cart(list(lines), self.soft)
            if self.verdicts.get(self.check_key(list(lines))) is not True:
                self.trace.append({"kind": "swap_better", "pairs": pairs, "refused": "lead check", "mode": mode})
                return False, ""
            self.swaps += 1
            done = self.run_cart(calls, until)
            if len(done) == len(calls):
                self.page_first(list(pairs.values()), self.tail(True))
            self.trace.append({"kind": "swap_better", "pairs": pairs, "calls": done, "mode": mode})
            said = "Kyo put " + ", ".join(f"{n} in place of {o}" for o, n in pairs.items()) + f" {why} (" + "; ".join(done) + ")."
            return len(done) == len(calls), said

        for new in better:
            same = [i for i in now if kind(i) and kind(i) == kind(new)]
            if len(same) != 1 or same[0] in hold:
                continue
            old = same[0]
            p_new, p_old = self.near_price(new), self.near_price(old)
            if self.brief.get("pick") == "cheapest" and mode != "final" and (p_new is None or p_old is None or p_new >= p_old):
                continue
            lines = {**{i: self.cart_lines.get(i, 1) for i in now if i != old}, new: self.cart_lines.get(old, 1)}
            if not self.inside_budget(lines):
                continue
            if mode == "before":
                number = lines_of.get(old) or (1 if len(self.items) == 1 else 0)
                if not number:
                    continue
                values = self.worded_values(number)
                if self.screen(new, values) < self.screen(old, values):
                    continue
            calls = self.cart_calls(lines, [old])
            if not calls or len(calls) + (self.tail(True) if mode == "final" else 1 + self.tail()) > self.left():
                return False, ""
            if mode == "before":
                # the planned cart is judged before any cart call, by the heavy model from its own pool; a check that
                # cannot run, or a light verdict, means no swap
                self.pool_as, self.answered_by = "cheaper", None
                try:
                    self.check_cart(list(lines), self.hard)
                finally:
                    self.pool_as = ""
                if self.verdicts.get(self.check_key(list(lines))) is not True or self.answered_by != HEAVY:
                    self.trace.append({"kind": "swap_better", "old": old, "new": new, "refused": "lead check"})
                    return False, ""
            self.swaps += 1
            done = self.run_cart(calls, until)
            if len(done) == len(calls) and mode != "final":
                # the swapped-in listing's page, before the checkout that follows (a sweep may have read its attributes only)
                self.page_first([new], self.tail(True))
            self.trace.append({"kind": "swap_better", "old": old, "new": new, "calls": done, "mode": mode})
            said = f"Kyo put {new} in place of {old} {why} (" + "; ".join(done) + ")."
            if mode != "after":
                return len(done) == len(calls), said
            again = self.check_cart(None, self.soft)
            return (not again and bool(self.verdicts.get(self.check_key(self.cart_ids())))), (said + (" " + again if again else ""))
        return False, ""

    def repair_cart(self) -> tuple[str, str]:
        """The walker sends a checkout again for a cart the check found wrong. Once per cart key, before it goes through,
        code tries in order: the check's keep fix; a swap to listings that check named better (up to two unopened ones
        opened first); for a missing item, the closest fill from opened listings of its department (one search of its name
        and told values first when it has none; the fill checks out); for each faulted item, one search of the told values
        its line's record does not state with the item's name, its two best unread rows of its department opened, one
        more check showing the item's opened listings, and a swap to one it names. Every swap passes swap_better's
        "repair" gates. Each fix keeps the order's steps plus one; no search past DEADLINE-20 s, no re-check past DEADLINE-10 s
        and no check past LATE-8 s;
        the heavy model only inside the soft spend. Returns (state, what was done): state "fixed" (the cart was set and
        its planned cart or re-check passed), "held" (the cart changed and is still judged wrong), "ended" (Kyo checked
        out itself), or "" (nothing done: the checkout goes through as it is)."""
        now = list(dict.fromkeys(self.cart_ids()))
        key = self.check_key(now)
        if (key in self.repaired or self.verdicts.get(key) is not False or self.finished or not now or not self.roles.get("cart")
                or time.time() - self.t0 > LATE - 8 or self.cost > self.hard):
            return "", ""
        self.repaired.add(key)
        log: dict[str, Any] = {"kind": "repair", "key": key, "left": self.left(), "t": round(time.time() - self.t0)}
        why = "(the check named it better)"
        cat_key = self.category_key()
        whole = self.whole_cart()
        # 1. the check's keep fix
        fixed, said = self.code_fix(until=HARD_END - 25, limit=self.soft, room=self.tail(True) + 1)
        if said:
            self.trace.append({**log, "step": "keep", "fixed": fixed})
            return ("fixed" if fixed else "held"), said
        # 2. the listings it named better, the unopened ones opened first
        better = list(self.betters.get(key, []))
        unread = [b for b in better if not self.was_opened(b) and self.buyable(b)][:2]
        if unread and time.time() - self.t0 < OPEN_END - 10:
            self.fill_facts(unread, (1 if whole else 2) + self.tail(True) + 1)
        if better:
            fixed, said = self.swap_better(better, need_change=False, why=why, mode="repair", until=HARD_END - 25)
            if said:
                self.trace.append({**log, "step": "better", "fixed": fixed})
                return ("fixed" if fixed else "held"), said
        refs = {need for _, need in self.refs}
        held = self.item_lines()

        def find(number: int, words: list[str]) -> list[str]:
            """One search of `words` and the item's name; its two best unread rows of the item's department opened."""
            item = self.items[number - 1]
            keep = (1 if whole else 2) + self.tail(True) + 1
            if (time.time() - self.t0 > DEADLINE - 20 or self.left() < 3 + keep or self.cost > self.hard
                    or not self.chain_search(" ".join(words[:2] + [item.name or item.shelf]))):
                return []
            rows = [i for i, _ in self.records_in(self.last_seen) if self.buyable(i) and not self.was_opened(i) and self.dept_match(number, i, cat_key)]
            opened = [t.split(":", 1)[0] for t in self.fill_facts(rows[:2], keep)]
            self.trace.append({"kind": "repair_search", "item": number, "words": words[:2], "opened": opened})
            return opened

        def told_words(number: int, line: str | None) -> list[str]:
            """The item's told values as the told scan searches them (told_values), leaving out those its line states."""
            item = self.items[number - 1]
            record = _toks(_dump(self.full.get(line) or self.rows.get(line) or {})) if line else frozenset()
            told = [m for m in item.told if m not in refs and m not in item.dropped and ":" in m]
            return [w for w in self.told_values(number, told) if not (_toks(w) and _toks(w) <= record)]

        # 3. a missing item: the closest fill from opened listings of its department (searched first when it has none)
        if self.cart_short() and not self.cart_broken() and held and len(held) == len([i for i in now if not self.sold_out(i)]):
            for number in range(1, len(self.items) + 1):
                if number not in held and not self.near_code(number, opened_only=True):
                    find(number, told_words(number, None))
            done = self.near_buy(fill=True)
            if done:
                self.trace.append({**log, "step": "fill"})
                return "ended", done
        # 4. a faulted item: its told values searched, then one more check that sees the item's opened listings
        kept = {_count(e.get("item")) for e in self.keeps.get(key) or [] if isinstance(e, dict)}
        faulted = [n for n in range(1, len(self.items) + 1) if n not in kept and (n in held or len(self.items) == 1)]
        shown: list[str] = []

        def label(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""

        for number in faulted:
            line = held.get(number) or (now[0] if len(now) == 1 else None)
            words = told_words(number, line)
            fresh = find(number, words) if words else []
            if not fresh and line and time.time() - self.t0 < OPEN_END - 10:
                # no search to make or it opened nothing: two never-opened rows of the line's department that keep the
                # budget (cheapest first under a cheapest pick, else in the order first seen)
                rest = {j: self.cart_lines.get(j, 1) for j in now if j != line}
                unread = [i for i in self.seen_order if not self.was_opened(i) and self.buyable(i) and i not in now and label(i) == label(line)
                          and self.inside_budget({**rest, i: self.items[number - 1].qty})]
                if self.brief.get("pick") == "cheapest":
                    unread.sort(key=lambda i: self.near_price(i) if self.near_price(i) is not None else float("inf"))
                keep = (1 if whole else 2) + self.tail(True) + 1
                fresh = [t.split(":", 1)[0] for t in self.fill_facts(unread[:2], keep)]
            # the item's opened listings of the faulted line's own department (a swap takes only those): those whose
            # record carries most of the item's requirement values first (a figure also as the listings print it), then
            # the newest opened
            item = self.items[number - 1]
            wants = []
            for need in item.musts + item.told:
                if need not in item.dropped and ":" in need:
                    value = need.split(":", 1)[1]
                    wants += [w for w in (_toks(value), _toks(self.printed_figure(value))) if w]
            pool = [i for i in reversed(self.seen_order) if self.was_opened(i) and self.buyable(i) and i not in now and i not in fresh
                    and i not in shown and self.dept_match(number, i, cat_key) and (not line or label(i) == label(line))]
            pool.sort(key=lambda i: -sum(1 for w in wants if w <= _toks(_dump(self.full.get(i) or self.rows.get(i) or {}))))
            # room shared among the faulted items: at least three listings each
            shown += (fresh + pool)[: max(3, 9 // len(faulted))]
        # the re-check only while the planned cart's check can still follow it before LATE-8 s
        if not shown or self.finished or time.time() - self.t0 > DEADLINE - 10 or self.cost > self.hard:
            self.trace.append({**log, "step": "none", "faulted": faulted})
            return "", ""
        self.checked.discard(key)
        self.check_cart(None, self.soft, extra=shown)
        named = [b for b in self.betters.get(key, []) if b in shown] if self.verdicts.get(key) is False else []
        if named:
            fixed, said = self.swap_better(named, need_change=False, why=why, mode="repair", until=HARD_END - 25)
            if said:
                self.trace.append({**log, "step": "search", "fixed": fixed, "shown": shown})
                return ("fixed" if fixed else "held"), said
        self.trace.append({**log, "step": "recheck", "shown": shown, "named": named, "ok": self.verdicts.get(key)})
        if self.verdicts.get(key) is False and self.roles.get("refuse") and not self.finished and self.relaxed < 0:
            # the check found the cart wrong twice and named nothing better: when NEAR also finds an item's closest opened
            # listing breaking a requirement, nothing of it fits, and the ending without a purchase is the one that can pass
            # (not after a loosening: a dropped requirement or a raised budget says the looser task admits a listing)
            needs, late = self.near_rank({})
            broken = {n: b for n, b in self.near_breaks.items() if b} if needs == self.near_needs else {}
            if broken and self.decline_look(broken, late):
                self.near_needs = None
                needs, late = self.near_rank({})
                broken = {n: b for n, b in self.near_breaks.items() if b} if needs == self.near_needs else {}
            # never an ending without a purchase with steps to spare and no look made first
            room = self.left() > self.tail() + len(self.items) + 2 and not late and time.time() - self.t0 < OPEN_END - 20
            self.trace.append({"kind": "repair_near", "broken": {str(n): b[:3] for n, b in broken.items()}, "late": late, "room": room})
            if broken and (self.decline_looked or not room):
                self.no_fit = True
                said = self.decline_now()
                if said:
                    return "ended", said
        return "", ""

    def decline_now(self) -> str:
        """Kyo ends the episode without a purchase itself (the request's ending arguments, after the ending's gate);
        returns the store's reply, or "" when the ending cannot be sent."""
        refuse = self.roles.get("refuse")
        faults: list[str] = []
        args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults) if refuse else {}
        if not refuse or faults or self.finished or self.left() <= 0:
            return ""
        if self.left() >= 2:
            self.gate("end")
        if self.finished or self.left() <= 0:
            return ""
        reply = self.act(refuse, args)
        self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200), "no_fit": self.no_fit, "by": "repair"})
        if reply is None:
            return ""
        if not self.finished:
            self.observe(refuse, reply.get("observation"))
        return ("Kyo ended without a purchase: " if self.finished else "Kyo tried to end without a purchase and the store replied: ") + self.show(reply.get("observation"))

    def inside_budget(self, lines: dict[str, int]) -> bool:
        """Whether planned cart lines keep a numeric budget by the listings' known prices (a line over a per-item budget
        breaks it; an unknown price is not held against the plan)."""
        budget = self.budget
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return True
        prices = {i: self.near_price(i) for i in lines}
        if self.brief.get("budget_scope") == "each":
            return all(p is None or p <= budget for p in prices.values())
        return sum((p or 0.0) * q for i, q in lines.items() for p in [prices[i]]) <= budget

    def final_review(self, placed: set[str]) -> str:
        """Once, in finish() before the forced checkout, and only when a fix still fits: one check of the held cart (the
        light model late or near the soft spend, its timeout cut through call_until), then code sets the lines it names
        right (code_fix without a re-check), else swaps at most one line for an opened, buyable listing it named better.
        Lines the closest-listing buy placed in this finish (`placed`) stay. Returns what was done, or ""."""
        if self.final_done or self.finished or not self.roles.get("cart"):
            return ""
        self.final_done = True
        now = list(dict.fromkeys(self.cart_ids()))
        elapsed = time.time() - self.t0
        if not now or self.left() < self.tail(True) + (1 if self.whole_cart() else 2) or elapsed >= LATE - 8:
            self.trace.append({"kind": "final_review", "skip": True, "left": self.left(), "t": round(elapsed)})
            return ""
        key = self.check_key(now)
        self.trace.append({"kind": "final_review", "start": round(elapsed), "left": self.left(), "checked": key in self.checked})
        if key not in self.checked:
            light = elapsed > DEADLINE - 10 or self.cost > self.soft - 0.012
            self.call_until = time.time() + max(0.0, min(20.0, LATE + 5 - elapsed))
            try:
                self.check_cart(None, self.hard + 0.012, [LIGHT] if light else [HEAVY, LIGHT])
            finally:
                self.call_until = 0.0
            self.cheaper_fits = []
        if self.finished or self.verdicts.get(key) is not False:
            return ""
        fixed, said = self.code_fix(recheck=False, until=HARD_END - 25, hold=placed)
        if not said:
            fixed, said = self.swap_better(None, need_change=False, why="in the last review", mode="final", hold=placed, until=HARD_END - 25)
        self.trace.append({"kind": "final_review", "fixed": fixed, "said": said[:160], "t": round(time.time() - self.t0)})
        return said

    def audit(self) -> str:
        if self.audited or self.cost > self.hard:
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

    def late_rescue(self) -> str:
        """The walker ends without a purchase in the last steps with nothing carted: one audit of the records (opening a
        promising row when the steps allow), a fresh check of its picks, and code carts and checks out what both confirm."""
        if self.cost > self.hard + 0.01 or self.left() < 2 or time.time() - self.t0 > LATE - 30:
            return ""
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.left() >= 4 and self.open_rows(rows[:1] if self.compare_bare() or not self.roles.get("compare") else rows, floor=3):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named or len(named) < len(self.items):
            return ""
        picks = [ids[0] for _, ids in sorted(named.items())]
        self.check_cart(picks, self.hard + 0.01)
        if not self.verdicts.get(self.check_key(picks)):
            return ""
        self.lead_items, self.leads, self.confirmed = named, [i for ids in named.values() for i in ids], picks[0]
        self.trace.append({"kind": "late_rescue", "picks": picks})
        return self.complete_cart()

    def decline_widen(self) -> str:
        """Before the final decline: one plain search per item by its own name, the unread rows of that search opened,
        then the audit again; code buys what the audit and a fresh check confirm."""
        if self.widened or self.finished or len(self.items) > 2 or self.left() < 8 + len(self.items) or self.cost > self.hard or time.time() - self.t0 > 210:
            return ""
        self.widened = True
        fresh: list[str] = []
        for item in self.items:
            before = set(self.rows)
            self.search_pages(item.name or item.shelf, 2 if self.left() > 20 else 1)
            new = [i for i in reversed(self.seen_order) if i not in before and i not in self.full and self.buyable(i)]
            cat_key = self.category_key()
            want = _stem(item.shelf or "") | _stem(item.name)
            # the item's own department first: the store pads search pages with other departments
            new.sort(key=lambda i: not (cat_key and {w for w in _stem(str((self.rows.get(i) or {}).get(cat_key) or "")) if len(w) >= 4} & want))
            fresh += new[:3]
        if fresh:
            self.fill_facts(fresh[: 3 * len(self.items)], 8)
        got = self.audit_call(light=True)
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.full and not self.sold_out(str(i))]
            if rows and self.left() > 7:
                self.fill_facts(rows[:3], 4)
                got = self.audit_call(light=True)
                named = self.audit_fits(got)
        self.trace.append({"kind": "decline_widen", "fresh": fresh, "named": named})
        if not named or len(named) < len(self.items):
            return ""
        picks = [ids[0] for _, ids in sorted(named.items())]
        self.check_cart(picks, self.hard + 0.01)
        if not self.verdicts.get(self.check_key(picks)):
            return ""
        self.lead_items, self.leads, self.confirmed = named, [i for ids in named.values() for i in ids], picks[0]
        self.completed = False
        return self.complete_cart()

    def search_pages(self, words: str, extra: int) -> None:
        """A plain search of Kyo's own, then up to `extra` further pages of it (the store's own next-page value)."""
        tool = self.roles.get("search")
        if not tool or not self.chain_search(words):
            return
        call, seen = self.last_search, self.last_seen
        for _ in range(extra):
            plan = self.paging(tool, seen)
            if not plan or plan[3] is None or self.left() < 12 or self.finished:
                return
            faults: list[str] = []
            args = _shape({**call, plan[2]: plan[3]}, self.schemas.get(tool), "args", faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            call = args

    def dept_match(self, number: int, ident: str, cat_key: str | None = None, exact: bool = False, fold: bool = False) -> bool:
        """Whether the listing's department label names requested item `number`: the label is the item's store label or
        name, or shares a word of four letters or more with them (the rule the shelf reads use); with `exact`, only the
        label being the store label or name counts; with `fold`, words of three letters or more count, plurals folded
        (rug and Area Rugs, mattress and Mattresses). False without a label. `cat_key` saves reading the category key again
        inside a loop."""
        cat_key = cat_key or self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        if not label or not 1 <= number <= len(self.items):
            return False
        same, shared = self.dept_score(number, label, fold)
        if exact:
            return same
        # the item's exact label wins whenever a seen row carries it; else only the labels sharing the most of its words
        best = self.dept_top(number, cat_key, fold)
        if best[0]:
            return same
        return same or (shared > 0 and shared >= best[1])

    def dept_score(self, number: int, label: str, fold: bool = False) -> tuple[bool, int]:
        """(the label is item `number`'s store label or name, plurals folded; how many of the item's words it shares:
        words of four letters or more, with `fold` also the noun's and the name's last word of three letters)."""
        item = self.items[number - 1]

        def folded(text: str) -> list[str]:
            return [_fold(w) for w in re.findall(r"[a-z]+", _words(text))]

        own = folded(label)
        same = bool(own) and own in (folded(item.shelf or ""), folded(item.name))
        if fold:
            # a 3-letter word inside the item's free-worded name ('and', 'for') names no department: of the name, only
            # its last word (the product's noun) or words of four letters or more count
            named = re.findall(r"[a-z]+", _words(item.name))
            mine = {_fold(w) for w in re.findall(r"[a-z]+", _words(f"{item.shelf} {item.noun}")) + named[-1:] if len(w) >= 3}
            mine |= {_fold(w) for w in named if len(w) >= 4}
            return same, len({w for w in own if len(w) >= 3} & mine)
        long = {w for w in folded(item.shelf or "") + folded(item.name) if len(w) >= 4}
        return same, len({w for w in own if len(w) >= 4} & long)

    def dept_top(self, number: int, cat_key: str | None, fold: bool = False) -> tuple[bool, int]:
        """Over the department labels of the rows seen: whether one is item `number`'s exact label, and the most of its
        words any label shares (dept_score), kept until more rows are seen."""
        item = self.items[number - 1]
        key = (number, cat_key, fold, len(self.rows), item.shelf, item.name, item.noun)
        if key not in self.dept_best:
            labels = {str(r.get(cat_key) or "") for r in self.rows.values() if isinstance(r, dict) and r.get(cat_key)} if cat_key else set()
            scores = [self.dept_score(number, lab, fold) for lab in labels]
            self.dept_best[key] = (any(s[0] for s in scores), max([s[1] for s in scores] or [0]))
        return self.dept_best[key]

    def item_lines(self) -> dict[int, str]:
        """The cart's buyable lines matched to the requested items by department label (dept_match), in cart order, one
        line per item: item number -> listing id. Exact labels are matched first, so a line whose label shares a word
        with two items goes to the item it names. A line whose label names no item still unmatched is left out."""
        cat_key = self.category_key()
        ids = [i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
        out: dict[int, str] = {}
        for exact in (True, False):
            for ident in ids:
                if ident in out.values():
                    continue
                number = next((n for n in range(1, len(self.items) + 1) if n not in out and self.dept_match(n, ident, cat_key, exact)), None)
                if number:
                    out[number] = ident
        return dict(sorted(out.items(), key=lambda pair: ids.index(pair[1])))

    def missing_items(self, pacing: bool = False) -> list[int]:
        """Requested items with no buyable cart line while the cart holds fewer lines than items. Counted only for an item
        whose department label the store showed on some listing, so a request read as more items than the store sells is
        no gap. With `pacing`, every item while the cart is empty and without the department condition."""
        ids = [i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
        if len(self.items) < 2 or len(ids) >= len(self.items):
            return []
        if not ids:
            return list(range(1, len(self.items) + 1)) if pacing else []
        cat_key = self.category_key()
        if not cat_key:
            return []
        per = self.item_lines()
        return [n for n in range(1, len(self.items) + 1) if n not in per and (pacing or any(self.dept_match(n, i, cat_key) for i in self.rows))]

    def cart_short(self) -> bool:
        """The cart holds fewer buyable lines than there are requested items."""
        return len([i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]) < len(self.items)

    def cart_broken(self) -> bool:
        """A held line is sold out, or the cart's total (by the prices the shape reading keys, every one known) is over a
        total budget."""
        ids = list(dict.fromkeys(self.cart_ids()))
        if not ids:
            return False
        if any(self.sold_out(i) for i in ids):
            return True
        total = self.cart_total()
        if self.budget is None or self.brief.get("budget_scope") == "each" or total is None:
            return False
        return total > self.budget

    def buy_steps(self) -> int:
        """Store steps a closest-listing buy still takes: the cart calls for the missing lines (one call when the cart
        tool takes the whole list), one spare for a turned-down checkout, and the finish (tail: the checkout, and the
        message an order step needs)."""
        missing = max(0, len(self.items) - len([i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]))
        return (min(1, missing) if self.whole_cart() else missing) + 1 + self.tail()

    def whole_cart(self) -> bool:
        """The cart tool takes the whole list of lines in one call."""
        props = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        return any((isinstance(spec, dict) and "array" in _kinds(spec) and isinstance(spec.get("items"), dict)
                    and isinstance(spec["items"].get("properties"), dict)) or _map_spec(spec) for spec in props.values())

    def running_out(self) -> bool:
        """The walk is about to stop on time, model spend or turns (the step limit has its own warnings)."""
        turns, spend = self.walk_caps
        return time.time() - self.t0 >= DEADLINE - 25 or self.cost >= spend - 0.008 or self.loop_calls >= turns - 4

    def commit_due(self) -> bool:
        """Only the steps a buy takes are left and the cart misses an item or is broken: code buys the closest listings now."""
        return (bool(self.roles.get("cart") and self.roles.get("buy")) and bool(self.items) and self.left() <= self.buy_steps()
                and (self.cart_short() or self.cart_broken()))

    def near_state(self) -> tuple[Any, ...]:
        """What a turned-down checkout can change: the task version, the budget, the sold-out listings, the requirements."""
        return (self.version, self.budget, len(self.gone), len(self.unavailable), _dump(self.needs_now()))

    def row_key(self, test: Any) -> str | None:
        """The one key that most of the newest result rows carry with a value passing `test`, else None."""
        rows = [r for r in list(self.rows.values())[-60:] if isinstance(r, dict)]
        keys = [k for k in {k for r in rows for k, v in r.items() if test(v)} if sum(1 for r in rows if test(r.get(k))) >= 0.8 * len(rows)]
        return keys[0] if len(rows) >= 2 and len(keys) == 1 else None

    def near_price(self, ident: str) -> float | None:
        """The listing's price for ranking: under the key the shape reading named, else under the rows' one number key."""
        price = self.price_of(ident)
        if price is None and not self.price_key:
            key = self.row_key(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
            value = (self.rows.get(ident) or {}).get(key) if key else None
            price = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        return price

    def near_code(self, number: int, opened_only: bool = False) -> list[str]:
        """Requested item `number`'s buyable listings ranked by code alone: the agent's own choices first (cart lines,
        candidates, built lines, the confirmed one), then opened before rows, then how many of the item's live requirement
        values the record's words carry, then a stock flag read off the rows, then (for a cheapest pick) price. The pool
        is the item's department; with `opened_only` its opened listings only. Otherwise, when the store shows no
        department label every listing is in it, and when the item's department shows none, opened listings of a
        department no other item claims."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        wants = [w for w in (_toks(m.split(":", 1)[1]) for m in item.musts + item.told if m not in item.dropped and ":" in m) if w]
        pool = [i for i in self.seen_order if self.buyable(i) and self.dept_match(number, i, cat_key) and (i in self.full or not opened_only)]
        if not pool and not opened_only:
            if not cat_key:
                pool = [i for i in self.seen_order if self.buyable(i)]
            else:
                others = [n for n in range(1, len(self.items) + 1) if n != number]
                pool = [i for i in self.seen_order if i in self.full and self.buyable(i) and not any(self.dept_match(n, i, cat_key) for n in others)]
        mine = set(self.cart_ids()) | set(self.leads) | set(self.built) | ({self.confirmed} if self.confirmed else set())
        flag = None if self.stock_key else self.row_key(lambda v: isinstance(v, bool))
        cheap = self.brief.get("pick") == "cheapest"

        def rank(ident: str) -> tuple[Any, ...]:
            words = _toks(_dump(self.full.get(ident) or self.rows.get(ident) or {}))
            price = self.near_price(ident)
            doubt = bool(flag) and (self.rows.get(ident) or {}).get(flag) is False
            # a label that is the item's own store label or name beats one that only shares a word with it
            return (not self.dept_match(number, ident, cat_key, True), ident not in mine, ident not in self.full, -sum(1 for w in wants if w <= words), doubt,
                    (price if price is not None else float("inf")) if cheap else 0.0)

        return sorted(pool, key=rank)

    def exact_seen(self, number: int, cat_key: str | None) -> bool:
        """Some seen listing's department label is item `number`'s own store label or name."""
        return any(self.dept_match(number, i, cat_key, True) for i in self.seen_order)

    def cover_items(self) -> None:
        """Before a full closest-listing pick: an item with no opened buyable listing of its department (or of a
        department no other item claims) gets its best rows opened, after one plain search of its name when no row of it
        was seen, keeping the steps the buy takes. When a seen row carries the item's exact label, only an opened listing
        of that label counts. Nothing of this past OPEN_END."""
        cat_key = self.category_key()
        if not cat_key:
            return
        for number, item in enumerate(self.items, 1):
            if self.finished or time.time() - self.t0 > OPEN_END:
                return
            others = [n for n in range(1, len(self.items) + 1) if n != number]
            exact = self.exact_seen(number, cat_key)
            if any(i in self.full and self.buyable(i) and (self.dept_match(number, i, cat_key, exact) or (not exact and not any(self.dept_match(n, i, cat_key) for n in others)))
                   for i in self.seen_order):
                continue
            keep = self.buy_steps() + 1
            rows = [i for i in self.near_code(number) if i not in self.full]
            if not rows and self.left() > keep + 2:
                self.chain_search(item.name or item.shelf)
                cat_key = self.category_key() or cat_key
                rows = [i for i in self.near_code(number) if i not in self.full]
            if rows:
                self.fill_facts(rows[:2], keep)

    def fit_budget(self, named: dict[str, list[str]]) -> tuple[list[str], bool]:
        """One listing per item, no two of one department: each item's first choice, then, while the total (or a line,
        for a budget per item) is over the budget, the best-ranked cheaper alternate whose swap alone brings it inside,
        else the swap saving most. Returns the picks in item order ([] when an item has no choice left) and whether they
        fit the budget by the prices known."""
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or ident).strip().lower() if cat_key else ident

        each = self.brief.get("budget_scope") == "each"

        def cost(number: int, ident: str) -> float:
            # a per-item budget limits the unit price; a total one the price times the quantity
            return (self.near_price(ident) or 0.0) * (1 if each else self.items[number - 1].qty)

        picks: dict[int, str] = {}
        for number in range(1, len(self.items) + 1):
            taken = {kind(p) for p in picks.values()}
            first = next((i for i in named.get(str(number)) or [] if i not in picks.values() and kind(i) not in taken), None)
            if not first:
                return [], False
            picks[number] = first
        if self.budget is None:
            return [picks[n] for n in sorted(picks)], True
        for _ in range(3 * len(self.items) + 3):
            total = sum(cost(n, i) for n, i in picks.items())
            over = [n for n in picks if cost(n, picks[n]) > self.budget] if each else ([] if total <= self.budget else list(picks))
            if not over:
                return [picks[n] for n in sorted(picks)], True
            best = None
            for n in over:
                taken = {kind(p) for m, p in picks.items() if m != n}
                for place, j in enumerate(named.get(str(n)) or []):
                    if j in picks.values() or kind(j) in taken or cost(n, j) >= cost(n, picks[n]):
                        continue
                    fits = cost(n, j) <= self.budget if each else total - cost(n, picks[n]) + cost(n, j) <= self.budget
                    key = (not fits, place if fits else cost(n, j) - cost(n, picks[n]))
                    if best is None or key < best[0]:
                        best = (key, n, j)
            if best is None:
                break
            picks[best[1]] = best[2]
        return [picks[n] for n in sorted(picks)], False

    def code_breaks(self, number: int, ident: str) -> list[str]:
        """Item `number`'s requirements in force that the listing's record contradicts by code's own reading: a worded
        value to search by (no figure, no exclusion or pointer) whose words are in none of the record's attributes named
        like its property, nor anywhere else in the record. [] when code cannot tell."""
        item = self.items[number - 1]
        record = self.full.get(ident) or self.rows.get(ident) or {}
        refs = {need for _, need in self.refs}
        dumped = _dump(record)[:6000]
        # a short letters-and-figure code the record prints only with another figure (wanted cl30, record CL32)
        out = self.code_clashes(number, ident)
        block = _fact_block(record) or {}
        if not block:
            return out
        text = _toks(dumped, True)
        for need in item.musts + item.told:
            if need in item.dropped or need in refs or ":" not in need or need in out:
                continue
            value = item.find.get(need) if need in item.find else self.find_words.get((number, need))
            if not value or re.search(r"\d", value) or len(value.split()) > 4:
                continue
            words = _toks(value, True)
            prop = _stem(need.split(":", 1)[0])
            keys = [k for k in block if _names(prop, _stem(k))]
            if words and keys and not any(words <= _toks(str(block[k]), True) for k in keys) and not words <= text:
                out.append(need)
        return out

    def code_clashes(self, number: int, ident: str) -> list[str]:
        """Item `number`'s requirements in force that the listing's records contradict by a short code printed with another
        figure (_code_clash: wanted cl30, records CL32)."""
        item = self.items[number - 1]
        refs = {need for _, need in self.refs}
        dumped = _dump(self.full.get(ident) or self.rows.get(ident) or {})[:6000]
        out = []
        for need in item.musts + item.told:
            if need in item.dropped or need in refs or ":" not in need:
                continue
            own = need.split(":", 1)[1].strip().lower()
            whole = need.lower()
            if _NOT_PLAIN.search(whole) or _LIMIT_OR.search(whole):
                # a limit or a choice of codes in the property or the value (CL30 or lower, max cas latency: cl30, DDR4 or
                # newer), whatever bare code the find words hold
                continue
            value = (item.find.get(need) if need in item.find else self.find_words.get((number, need))) or own
            if _code_clash(value, dumped):
                out.append(need)
        return out

    def line_clashes(self) -> list[str]:
        """Cart lines that contradict a requirement of the item they fill by code_clashes: one "listing does not meet
        'requirement'" each. With one requested item, every buyable line is read against it even when no department label
        matched it to the item."""
        pairs = list(self.item_lines().items())
        if not pairs and len(self.items) == 1:
            pairs = [(1, i) for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
        return [f"{ident} does not meet '{need}' (its records print another code)"
                for number, ident in pairs for need in self.code_clashes(number, ident)]

    def told_rows(self, numbers: list[int], purpose: str = "buy", room: int = 6) -> None:
        """Before a closest-listing buy: unread rows of each item's department whose own row text shows a told value
        (every word of it; a bare figure shows nothing) are opened, rows showing more values first (then the lower price
        under a cheapest pick), up to `room` per episode and purpose, keeping the buy's steps; nothing past OPEN_END.
        NEAR ranks opened pages only, and an accepted listing is often a row no one opened."""
        cat_key = self.category_key()
        refs = {need for _, need in self.refs}
        room -= self.told_opened.get(purpose, 0)
        picks: list[str] = []
        cheap = self.brief.get("pick") == "cheapest"
        for number in numbers:
            if room <= 0:
                break
            item = self.items[number - 1]
            values = self.told_values(number, [m for m in item.musts + item.told if ":" in m and m not in item.dropped and m not in refs])
            told = [t for t in (_toks(v, True) for v in values if re.search(r"[A-Za-z]", v)) if t]
            if not told:
                continue

            def shown(i: str, by_query: bool = False) -> int:
                # told values the row's own text shows; with `by_query`, also those a one-page word search holding the
                # value showed it by (the store's word search reads page facts too)
                row = _toks(" ".join(str(v) for v in (self.rows.get(i) or {}).values() if isinstance(v, (str, int, float))), True)
                return sum(1 for t in told if t <= row or (by_query and self.query_shows(i, t, lead=True)))

            rows = [i for i in self.seen_order if i not in picks and not self.was_opened(i) and self.buyable(i)
                    and (not cat_key or self.dept_match(number, i, cat_key)) and shown(i, True)]
            rows.sort(key=lambda i: (not shown(i), -shown(i, True), (self.price_of(i) if self.price_of(i) is not None else float("inf")) if cheap else 0.0))
            picks += rows[: room - len(picks)]
        if picks and time.time() - self.t0 < OPEN_END:
            got = self.fill_facts(picks, self.buy_steps() + 1)
            self.told_opened[purpose] = self.told_opened.get(purpose, 0) + len(got)
            self.trace.append({"kind": "told_rows", "rows": picks, "opened": len(got), "purpose": purpose})

    def near_buy(self, fill: bool = False) -> str:
        """No ending without a purchase while each item has a buyable listing: code carts each item's closest listing and
        checks out, once per state (cart and near_state). With `fill` (a held, unbroken cart) the held lines stay and a
        missing item is filled only from opened buyable listings of its department; when one has none, or a held line
        names no requested item, nothing is done and the held cart is bought as it is by the caller. Without `fill`,
        items with no opened listing of their department first get their best rows opened. The light model ranks the
        opened listings (keeping the agent's own choices unless contradicted, with alternates) when the clock, the call
        cap and the spend allow and no ranking was made under the requirements in force; code's ranking follows as
        alternates or stands alone. Cheaper alternates bring the total inside the budget. Sets no_fit when an item has
        nothing buyable, or the cheapest picks break the budget by the prices the shape reading keys or after NEAR found
        nothing for an item. A checkout the store turns down with a change gets one re-pick from the cached ranking."""
        mark = _dump([self.cart_mark(), *self.near_state()])
        if (self.finished or mark in self.near_marks or not self.items or not self.roles.get("cart") or not self.roles.get("buy")
                or self.left() < self.buy_steps() - 1):
            return ""
        self.near_marks.add(mark)
        self.no_fit = False
        cat_key = self.category_key()
        lines = [i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
        held = self.item_lines() if fill and lines else {}
        if held and any(self.code_clashes(n, i) for n, i in held.items()):
            # a held line whose records print another code than one of its item's requirements cannot pass: every item
            # is picked again
            self.trace.append({"kind": "near_pick", "held_clash": held})
            held, fill = {}, False
        if (held and len(held) != len(lines)) or (fill and lines and not held):
            # a held line whose department names no requested item: which item is missing is unknown
            self.trace.append({"kind": "near_pick", "picks": [], "held": held, "why": "unmatched line"})
            return ""
        fill = bool(held)
        self.told_rows([n for n in range(1, len(self.items) + 1) if n not in held])
        if self.finished:
            return ""
        bare = [n for n in range(1, len(self.items) + 1) if n not in held
                and not [i for i in self.near_code(n, opened_only=True) if not self.exact_seen(n, cat_key) or self.dept_match(n, i, cat_key, True)]] if fill else []
        if fill and (len(held) >= len(self.items) or bare):
            # nothing to fill, or a missing item with no opened buyable listing of its department: the held cart as it is
            self.trace.append({"kind": "near_pick", "picks": [], "held": held, "bare": bare})
            return ""
        if not fill:
            # up to two rows a review named (AUDIT's rows to open, a check's better listings) are opened first, so the
            # closest-listing pick can judge them
            pointed = [i for i in dict.fromkeys(self.pointed) if i in self.rows and not self.was_opened(i) and self.buyable(i)]
            if pointed and time.time() - self.t0 < OPEN_END - 10:
                got = self.fill_facts(pointed[:2], self.buy_steps() + 1)
                self.trace.append({"kind": "near_open", "rows": pointed[:2], "opened": len(got)})
            self.cover_items()
            if self.finished:
                return ""
        needs, late = self.near_rank(held)
        named: dict[str, list[str]] = {str(n): [i] for n, i in held.items()}
        judged = needs == self.near_needs
        broken: dict[int, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            if number in held:
                continue
            model = [i for i in self.near_named.get(str(number)) or [] if i in self.full and self.buyable(i)
                     and (not fill or self.dept_match(number, i, cat_key)) and not self.code_clashes(number, i)]
            if judged and self.near_breaks.get(number):
                # NEAR's closest listing of the item clearly breaks a requirement: nothing of it fits
                broken[number] = self.near_breaks[number]
                continue
            code = [i for i in self.near_code(number, opened_only=fill) if i not in model]
            kept = [i for i in code if not self.code_breaks(number, i)]
            if code and not kept and not model:
                # every listing code ranked for it contradicts a requirement by its records
                broken[number] = self.code_breaks(number, code[0])
            named[str(number)] = list(dict.fromkeys(model + kept))
        if broken and self.decline_look(broken, late):
            # NEAR ranks again only when the look opened something
            self.near_marks.discard(mark)
            self.near_needs = None
            return self.near_buy(fill)
        if broken:
            self.no_fit = True
            self.trace.append({"kind": "near_decline", "broken": {str(n): b[:3] for n, b in broken.items()}, "held": held, "judged": judged})
            return ""
        empty = [n for n in range(1, len(self.items) + 1) if not named.get(str(n))]
        picks, fits = self.fit_budget(named) if not empty else ([], False)
        # a missing item with nothing to fill it leaves the held cart to be bought as it is; with nothing held it is no fit
        self.no_fit = (bool(empty) and not fill) or (bool(picks) and not fits and bool(self.price_key or self.near_empty))
        self.trace.append({"kind": "near_pick", "picks": picks, "fits": fits, "late": late, "held": held, "empty": empty, "no_fit": self.no_fit})
        if not picks or self.no_fit:
            return ""
        self.lead_items, self.leads, self.confirmed = {str(n): [i] for n, i in enumerate(picks, 1)}, list(picks), picks[0]
        self.completed = False
        self.trusted = True
        self.near_before = ()
        try:
            done = self.complete_cart()
        finally:
            self.trusted = False
        if done and not self.finished and self.near_rounds < 1 and self.near_before and self.near_state() != self.near_before:
            # the store turned the checkout down with a change (a budget, a requirement, a sold-out line): pick once more,
            # keeping the held lines unless the cart broke or the requirements changed
            self.near_rounds += 1
            same_needs = self.near_state()[-1] == self.near_before[-1]
            return self.near_buy(fill=same_needs and not self.cart_broken()) or done
        return done

    def decline_look(self, broken: dict[int, list[str]], late: bool) -> bool:
        """Once per episode before a decline, while steps and clock allow: the broken items' worded values searched with
        their product word, and the rows showing them opened. True when that opened a listing."""
        if self.decline_looked or late or self.left() <= self.buy_steps() + 3 or time.time() - self.t0 >= OPEN_END - 20:
            return False
        self.decline_looked = True
        before = len(self.full)
        refs = {need for _, need in self.refs}
        for number in broken:
            item = self.items[number - 1]
            values = self.told_values(number, [m for m in item.musts + item.told if ":" in m and m not in item.dropped and m not in refs])
            words = " ".join(values[:2] + [item.noun or item.name]).strip()
            walked = any(words.lower() == v.strip().lower() for _, a, _ in self.searches for v in a.values() if isinstance(v, str))
            if values and not walked and self.left() > self.buy_steps() + 3:
                self.chain_search(words)
        self.told_rows(list(broken), "look", 4)
        self.trace.append({"kind": "decline_look", "items": list(broken), "opened": len(self.full) - before})
        return len(self.full) > before and not self.finished

    def near_rank(self, held: dict[int, str]) -> tuple[str, bool]:
        """NEAR's ranking of the opened listings (the held lines first) when the clock, the call cap and the spend allow and
        no ranking was made under the requirements in force; fills near_named, near_empty and near_breaks. Returns (the
        requirements in force, too late to rank)."""
        needs = _dump(self.needs_now())
        live = [i for i in reversed(self.seen_order) if i in self.full and not self.sold_out(i)][:24]
        if held:
            # the held lines first, with every view read of them: an accessory's fit is judged against the device held
            live = list(dict.fromkeys([i for i in held.values() if i in self.rows] + live))[:24]
        elapsed = time.time() - self.t0
        # past LATE-8 s the ranking still runs, cut to the time a cart call, the checkout and one more ending after it
        # leave before HARD_END-5 s at the slowest store latency seen, while that is 8 s or more
        room = HARD_END - 5 - 3 * self.store_lat - elapsed
        late = ((elapsed > LATE - 8 and (elapsed > LATE or room < 8))
                or self.calls >= self.all_calls - 1 or self.cost > self.hard + 0.04)
        if live and needs != self.near_needs and not late:
            chosen = list(dict.fromkeys(self.leads + self.built + ([self.confirmed] if self.confirmed else [])))
            prompt = NEAR + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + self.departments()
            prompt += "\n\nCART NOW: " + (", ".join(f"{i} ({self.cart_lines.get(i, 1)} units)" for i in dict.fromkeys(self.cart_ids())) or "empty")
            prompt += "\nAGENT'S PICKS (listings it named or carted): " + (", ".join(chosen[:12]) or "none")
            prompt += "\n\nOPENED LISTINGS (newest first):\n" + "\n".join("OPENED " + self.review_record(i, 1200, 200, self.full.get(i) or self.rows.get(i)) for i in live)
            outer = self.call_until
            if elapsed > LATE - 8:
                self.call_until = min(outer or float("inf"), time.time() + room)
            try:
                got = self.ask_json(prompt, [LIGHT], 1200) or {}
            finally:
                self.call_until = outer
            self.near_needs = needs
            self.near_named, self.near_empty, self.near_breaks = {}, set(), {}
            found = got.get("items") if isinstance(got.get("items"), dict) else {}
            for key, entry in found.items():
                number = _count(key)
                if not isinstance(entry, dict) or not number or not 1 <= number <= len(self.items):
                    continue
                first = str(entry.get("id") or "").strip()
                if not first:
                    self.near_empty.add(number)
                elif first in self.full:
                    self.near_breaks[number] = self.real_breaks(number, entry.get("breaks"))
                ids = [first] + [str(x).strip() for x in _items(entry.get("others"))]
                self.near_named[str(number)] = [i for i in dict.fromkeys(ids) if i in self.full]
            self.trace.append({"kind": "near", "got": _clip(got, 150)})
        return needs, late

    def real_breaks(self, number: int, breaks: Any) -> list[str]:
        """NEAR's breaks of an item that name one of its requirements in force (a shared word of 3+ letters); filler such
        as None, "" or a no-break phrase names none."""
        item = self.items[number - 1]
        need = [m for m in item.musts + item.told if m not in item.dropped]
        if item.fits:
            need.append(f"fit {self.items[item.fits - 1].name}" if 0 < item.fits <= len(self.items) else "fit")
        words = {w for m in need for w in _stem(m) if len(w) >= 3}
        out = []
        for b in _items(breaks):
            text = "" if b is None or isinstance(b, (dict, list)) else str(b).strip()
            if text and {w for w in _stem(text) if len(w) >= 3} & words:
                out.append(text)
        return out

    def audit_call(self, light: bool = False) -> dict[str, Any] | None:
        records = []
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        for ident in [i for i in live if i in self.full][:32]:
            records.append("OPENED " + self.review_record(ident, 1200, 200, self.full[ident]))
        for ident in [i for i in live if i not in self.full][: max(12, 44 - len(records))]:
            records.append("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500])
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += self.departments()
        prompt += "\n\nLISTING RECORDS SEEN (opened pages newest first, then search rows):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT] if self.cost <= self.soft and not light else [LIGHT], 1500)
        self.trace.append({"kind": "audit", "got": got})
        rows = [str(i) for i in _items((got or {}).get("open")) if isinstance(i, (str, int)) and not isinstance(i, bool)]
        self.pointed += [i for i in rows if i in self.rows and not self.was_opened(i) and i not in self.pointed]
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for k, v in fits.items():
            digits = re.search(r"\d+", str(k))
            number = int(digits.group()) if digits else (1 if len(self.items) == 1 else 0)
            ok = 1 <= number <= len(self.items)
            # a listing whose records print another code than one of the item's requirements is no fit, whatever the audit says
            named[str(k)] = [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))
                             and not (ok and self.code_clashes(number, str(i)))][:4]
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str], floor: int = 5) -> bool:
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
        if self.left() < len(plan) + floor:
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

    def remover(self) -> tuple[str, str] | None:
        """The store tool that takes a whole cart line out when the cart tool only adds: a tool no job names whose one
        argument is the cart tool's listing argument (a free string). (tool, argument), or None when not exactly one."""
        cart = self.roles.get("cart")
        props = (self.schemas.get(cart or "") or {}).get("properties") or {}
        keys = {k for k, sp in props.items() if isinstance(sp, dict) and "string" in _kinds(sp) and "enum" not in sp}
        named = {t for t in self.roles.values() if t}
        found = []
        for tool, schema in self.schemas.items():
            own = (schema or {}).get("properties") or {}
            if tool in named or len(own) != 1:
                continue
            key, spec = next(iter(own.items()))
            if key in keys and isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec:
                found.append((tool, key))
        return found[0] if len(found) == 1 else None

    def took_off(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A line taken out with the removal tool: the cart record drops it (or takes the reply's cart when it lists one)."""
        self.cart_touched = True
        self.cart_at = self.step
        listed = self.store_lines(observation)
        if listed is not None:
            self.cart_lines = dict(listed)
            return
        for value in args.values():
            if isinstance(value, str) and value in self.cart_lines:
                self.cart_lines.pop(value, None)
                self.dropped.add(value)
        self.trace.append({"kind": "took_off", "tool": name, "args": args})

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            if _map_spec(spec):
                # a {listing: units} map sets the whole cart in one call
                if not drop and set(self.cart_ids()) == set(lines) and all(self.cart_lines.get(i) == n for i, n in lines.items()):
                    return []
                return [{key: {i: n for i, n in lines.items() if n > 0}}]
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
                if not id_key:
                    return None
                if not drop and set(self.cart_ids()) == set(lines) and all(self.cart_lines.get(i) == n for i, n in lines.items()):
                    return []
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        put, cut = self.mode_values(name, mode, props[mode]["enum"]) if mode else (None, None)
        sets = bool(count) and self.cart_sets(name)
        negative = bool(count) and isinstance(props[count].get("minimum"), int)
        remover = self.remover()
        calls: list[dict[str, Any]] = []

        def take(ident: str, units: int) -> bool:
            """Calls that take `units` units of a line off; False when this tool cannot."""
            if mode and count:
                calls.append({target: ident, mode: cut, count: units})
            elif mode:
                calls.extend({target: ident, mode: cut} for _ in range(units))
            elif negative and props[count]["minimum"] <= -units:
                calls.append({target: ident, count: -units})
            elif remover and units >= self.cart_lines.get(ident, units):
                calls.append({TOOL_KEY: remover[0], remover[1]: ident})
            else:
                return False
            return True

        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if sets:
                calls.append({target: ident, **({mode: put} if mode else {}), count: 0})
            elif not take(ident, have):
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if sets:
                calls.append({target: ident, **({mode: put} if mode else {}), count: want})
            elif want > have:
                if count:
                    calls.append({target: ident, **({mode: put} if mode else {}), count: want - have})
                else:
                    # a tool without a count adds one unit per call
                    calls.extend({target: ident, **({mode: put} if mode else {})} for _ in range(want - have))
            elif not take(ident, have - want):
                return None
        return calls

    def mode_values(self, tool: str, mode: str, options: list[Any]) -> tuple[Any, Any]:
        """Which value of the cart tool's mode argument puts units in and which takes them out: what the cart's own
        replies showed (a line's units going up or down after a call with that value), else the values BRIEF read off
        the tool's description, else the schema's order (the first value puts in, the next one takes out)."""
        moved = self.mode_moves.get((tool, mode)) or {}
        put = next((o for o in options if moved.get(_dump(o)) == 1), None)
        cut = next((o for o in options if moved.get(_dump(o)) == -1 and o != put), None)
        named = self.brief.get("cart_modes") if isinstance(self.brief.get("cart_modes"), dict) else {}
        said_put, said_cut = named.get("put"), named.get("take")
        if said_put in options and said_cut in options and said_put != said_cut:
            put = said_put if put is None and said_put != cut else put
            cut = said_cut if cut is None and said_cut != put else cut
        if put is None:
            put = next(o for o in options if o != cut)
        if cut is None:
            cut = next(o for o in options if o != put)
        return put, cut

    def cart_moved(self, name: str, args: dict[str, Any], had: int, now: int) -> None:
        """Learn from a cart reply which mode value this call used: the line's units went up (puts in) or down (takes out).
        A line that held units and now holds the call's own count shows no direction (a set, or a stale count); the first
        direction learned for a value stays."""
        if now == had:
            return
        given = next((_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None), None)
        if had > 0 and given is not None and now == given:
            return
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            options = (props.get(key) or {}).get("enum") if isinstance(props.get(key), dict) else None
            if isinstance(options, list) and len(options) >= 2 and value in options:
                self.mode_moves.setdefault((name, key), {}).setdefault(_dump(value), 1 if now > had else -1)
                self.trace.append({"kind": "cart_mode", "tool": name, "value": value, "moved": now - had})

    def mode_retry(self, name: str, args: dict[str, Any], observation: Any, before: dict[str, int],
                   view: bool = False) -> dict[str, Any] | None:
        """The same call with the mode argument's other value, when this call used a value of a two-value mode not learned
        yet, the line was not held before and the store's own reply (or, with `view`, the cart view after it) lists the
        cart without it; else None."""
        listed = self.store_lines(observation, view)
        acted = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        if listed is None or not acted or listed.get(acted[0], 0):
            return None
        return self.mode_other(name, args, before)

    def look_cart(self) -> Any:
        """One cart view, read like any other: its observation, or None when it could not be sent."""
        faults: list[str] = []
        view_args = _shape({}, self.schemas.get(self.roles.get("cart_view") or ""), "args", faults)
        view = self.act(self.roles["cart_view"], view_args) if not faults else None
        if view is None or self.finished:
            return None
        self.cart_view = view.get("observation")
        self.cart_seen_at = self.step
        self.saw_cart(self.cart_view)
        self.observe(self.roles["cart_view"], self.cart_view)
        return self.cart_view

    def mode_other(self, name: str, args: dict[str, Any], before: dict[str, int]) -> dict[str, Any] | None:
        """The call with the other value of its two-value mode argument, when that value is not learned yet and the call
        put a line the cart did not hold; else None."""
        acted = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        if not acted or before.get(acted[0], 0):
            return None
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            options = (props.get(key) or {}).get("enum") if isinstance(props.get(key), dict) else None
            if (isinstance(options, list) and len(options) == 2 and value in options
                    and (self.mode_moves.get((name, key)) or {}).get(_dump(value)) is None):
                return {**args, key: next(o for o in options if o != value)}
        return None

    def sums_to(self, observation: Any, lines: dict[str, int]) -> bool:
        """The reply carries a number equal to units x price summed over these lines: they are the whole cart."""
        if not isinstance(observation, dict):
            return False
        prices = [self.near_price(i) for i in lines]
        if None in prices:
            return False
        total = sum(n * float(p or 0.0) for n, p in zip(lines.values(), prices))
        return any(isinstance(v, (int, float)) and not isinstance(v, bool) and abs(float(v) - total) < 0.011 for v in observation.values())

    def store_lines(self, observation: Any, view: bool = False) -> dict[str, int] | None:
        """The cart as the store's own reply lists it ({listing id: units}); None when the reply lists no counted lines.
        `view`: the reply is the cart view's, so an empty container in it is the empty cart. In other cart replies an
        empty container is the empty cart only under a key where this store listed the cart's lines before."""
        found: list[dict[str, int]] = []
        self.empty_unsure = False
        spec = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        count_args = {_words(k) for k, v in spec.items() if isinstance(v, dict) and "integer" in _kinds(v)}

        def number(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def counted(row: dict[str, Any]) -> tuple[str, int] | None:
            # the units: the row's one whole-number field besides its price (the price key, or the field equal to the
            # listing's known price) and besides a line total (another such field times the price); with several, the one
            # named as the cart tool's count argument
            ident = next((v for v in row.values() if isinstance(v, str) and v in self.rows), None)
            if not ident:
                return None
            known = self.near_price(ident)
            prices = [k for k, v in row.items() if number(v) and (k == self.price_key or (known is not None and abs(float(v) - known) < 0.005))]
            ints = [k for k, v in row.items() if k not in prices and not isinstance(v, (str, bool)) and _count(v) is not None]
            if len(ints) > 1 and prices:
                cost = float(row[prices[0]])
                ints = [k for k in ints if not any(abs(float(row[k]) - float(row[j]) * cost) < 0.011 for j in ints if j != k)]
            if len(ints) > 1:
                ints = [k for k in ints if _words(k) in count_args]
            return (ident, _count(row[ints[0]]) or 0) if len(ints) == 1 else None

        def rows_in(node: Any, depth: int = 0) -> bool:
            # a part of the reply holding records or a {listing: units} map
            if depth > 5:
                return False
            if isinstance(node, list):
                return any(isinstance(x, dict) and x for x in node) or any(rows_in(x, depth + 1) for x in node)
            if isinstance(node, dict):
                return any(k in self.rows for k in node) or any(rows_in(v, depth + 1) for v in node.values())
            return False

        bare = not rows_in(observation)

        def by_total(rows: list[dict[str, Any]], parent: Any) -> dict[str, int] | None:
            # rows naming their units with a key not read above: the price is the field equal to the listing's known
            # price (or the price key), the units the row's one other integer field; taken only when units x price summed
            # over the rows equals another number of the reply
            others = [v for v in (parent.values() if isinstance(parent, dict) else []) if number(v)]
            if not others:
                return None
            lines: dict[str, int] = {}
            total = 0.0
            for row in rows:
                ident = next((v for v in row.values() if isinstance(v, str) and v in self.rows), None)
                if not ident:
                    return None
                known = self.near_price(ident)
                price = self.price_key if self.price_key in row and number(row.get(self.price_key)) else None
                if price is None and known is not None:
                    hits = [k for k, v in row.items() if number(v) and abs(float(v) - known) < 0.005]
                    price = hits[0] if len(hits) == 1 else None
                ints = [k for k, v in row.items() if k != price and isinstance(v, int) and not isinstance(v, bool)]
                if not price or len(ints) != 1:
                    return None
                lines[ident] = lines.get(ident, 0) + row[ints[0]]
                total += row[ints[0]] * float(row[price])
            if not any(abs(total - float(v)) < 0.011 for v in others):
                return None
            return {i: n for i, n in lines.items() if n > 0}

        def listed_at(key: str, lines: dict[str, int]) -> None:
            found.append(lines)
            if key:
                self.line_keys.add(key)

        def walk(node: Any, depth: int, parent: Any = None, key: str = "") -> None:
            if depth > 5 or found:
                return
            if (isinstance(node, dict) and node and all(isinstance(k, str) and k in self.rows for k in node)
                    and all(not isinstance(v, str) and _count(v) is not None for v in node.values())):
                # a {listing: units} map
                listed_at(key, {k: _count(v) for k, v in node.items() if (_count(v) or 0) > 0})
                return
            if isinstance(node, dict):
                empty = [str(k) for k, value in node.items() if isinstance(value, (list, dict)) and not value]
                if bare and empty:
                    if view or any(k in self.line_keys for k in empty):
                        # an empty container where this store lists the cart, and no record anywhere: the cart is empty
                        found.append({})
                        return
                    # an empty container this store has not listed the cart in: an empty cart or a part of an acknowledgement
                    self.empty_unsure = True
                for k, value in node.items():
                    walk(value, depth + 1, node, str(k))
            elif isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
                pairs = [counted(x) for x in node]
                if all(pairs):
                    lines: dict[str, int] = {}
                    for ident, units in pairs:
                        lines[ident] = lines.get(ident, 0) + units
                    listed_at(key, lines)
                    return
                read = by_total(node, parent)
                if read is not None:
                    listed_at(key, read)
                    return
                for value in node:
                    walk(value, depth + 1, None, key)

        walk(observation, 0)
        return found[0] if found else None

    def learn_cart(self, name: str, args: dict[str, Any], observation: Any, before: dict[str, int]) -> None:
        """Take the cart from the store's own reply when it lists the whole cart, and learn from it whether this tool
        adds its count to a line or sets the line to it."""
        acted = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        given = next((_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None), None)
        if acted:
            self.last_cart = (name, acted[0], before.get(acted[0], 0), given)
        listed = self.store_lines(observation)
        if listed is None:
            if acted:
                # a reply that lists nothing: the next cart view tells which way this call moved its line
                self.mode_probe = (name, dict(args), before.get(acted[0], 0), acted[0])
            return
        self.mode_probe = None
        for ident in acted[:1]:
            had, now = before.get(ident, 0), listed.get(ident, 0)
            self.cart_moved(name, args, had, now)
            if given and given > 0 and had > 0 and now != had:
                if now == had + given and now != given:
                    self.cart_rule[name] = "add"
                elif now == given:
                    self.cart_rule[name] = "set"
        others = set(before) - set(acted)
        whole = any(isinstance(v, dict) for v in args.values()) or others <= set(listed) or self.sums_to(observation, listed)
        if whole:
            self.cart_lines = dict(listed)
            self.cart_view = observation
            self.cart_seen_at = self.step
            self.trace.append({"kind": "store_cart", "lines": dict(listed)})
        else:
            for ident in acted:
                if ident in listed:
                    self.cart_lines[ident] = listed[ident]

    def saw_cart(self, observation: Any) -> None:
        """A cart view: its lines are the cart; it also tells whether the last count added to a line or set it."""
        listed = self.store_lines(observation, view=True)
        if listed is None:
            return
        probe, self.mode_probe = self.mode_probe, None
        if probe:
            self.cart_moved(probe[0], probe[1], probe[2], listed.get(probe[3], 0))
        if self.last_cart:
            name, ident, had, given = self.last_cart
            now = listed.get(ident, 0)
            if given and given > 0 and had > 0 and now != had:
                if now == had + given and now != given:
                    self.cart_rule[name] = "add"
                elif now == given:
                    self.cart_rule[name] = "set"
        self.cart_lines = dict(listed)
        self.trace.append({"kind": "cart_view", "lines": dict(listed), "rule": dict(self.cart_rule)})

    def code_fix(self, recheck: bool = True, until: float | None = None, hold: set[str] | frozenset[str] = frozenset(),
                 limit: float | None = None, room: int | None = None) -> tuple[bool, str]:
        """The check found the cart wrong but named the right listing of every item: code takes the other lines off and
        sets each kept line to its item's quantity, then checks again (with `recheck`; without it the order's steps are
        kept and the fix counts when every cart call went through). A line in `hold` is never taken off. Cart calls stop at
        `until` seconds. `limit`: the spend limit of the re-check (default the hard cap); `room`: the steps to keep besides
        the cart calls (default 2 with the re-check, else the order's). Returns (fixed and passed, what was done)."""
        cart = self.roles.get("cart")
        now = self.cart_ids()
        keep = self.keeps.get(self.check_key(now)) or []
        if not cart or self.code_fixes >= 2 or not now:
            return False, ""
        plan: dict[str, int] = {}
        for entry in keep:
            if not isinstance(entry, dict):
                continue
            number, ident = _count(entry.get("item")), str(entry.get("id") or "")
            if not number or not 1 <= number <= len(self.items) or ident not in now or self.sold_out(ident) or number in plan.values():
                continue
            plan[ident] = number
        if len(set(plan.values())) < len(self.items):
            return False, ""
        lines = {i: self.items[n - 1].qty for i, n in plan.items()}
        drop = [i for i in now if i not in lines]
        if not drop and all(self.cart_lines.get(i, 1) == q for i, q in lines.items()):
            return False, ""
        if any(i in hold for i in drop):
            return False, ""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        if not recheck and talk and self.msg_used >= self.msg_limit and not self.gate_due("order"):
            # the cart was approved and no message is left to approve a changed one
            return False, ""
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + (room if room is not None else 2 if recheck else self.tail(True)) > self.left():
            return False, ""
        self.code_fixes += 1
        done = self.run_cart(calls, until)
        self.trace.append({"kind": "code_fix", "lines": lines, "drop": drop, "calls": done})
        said = "Kyo set the cart itself after the review (" + "; ".join(done) + ")."
        if not recheck:
            return len(done) == len(calls), said
        again = self.check_cart(None, self.hard if limit is None else limit)
        if again:
            return False, said + " " + again
        return bool(self.verdicts.get(self.check_key(self.cart_ids()))), said

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
            # a lead whose records print another code than one of its item's requirements is passed over (a clashing
            # `confirmed` then leaves the plan and nothing is bought here)
            pick = next((i for i in ids if not self.sold_out(i) and not (index and 1 <= index <= len(self.items) and self.code_clashes(index, i))), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {} if self.trusted else {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        # a kept line of another item whose records print another code than one of that item's requirements is not kept
        # (the plan then falls short of the items and nothing is bought here)
        pairs = list(self.item_lines().items()) or ([(1, x) for x in now] if len(self.items) == 1 else [])
        clashing = {i for n, i in pairs if self.code_clashes(n, i)}
        keep = {i: q for i, q in keep.items() if i not in clashing}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + self.tail(True) > self.left():
            return ""
        if time.time() - self.t0 + (len(calls) + 1) * self.store_lat > HARD_END - 5:
            # the cart calls and the checkout cannot all land in time: finish() takes the one-call ending
            self.trace.append({"kind": "buy_no_time", "calls": len(calls), "lat": round(self.store_lat, 1)})
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        if not getattr(self, "trusted", False):
            self.check_cart(list(lines), self.hard)
            if not self.verdicts.get(self.check_key(list(lines))):
                return ""
        done = self.run_cart(calls)
        if len(done) < len(calls):
            return ""
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        if self.finished or self.left() < 1:
            return ""
        replies = []
        seen = len(self.changes)
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines, applying the
        # change and a fresh check; a closest-listing pick (bought unchecked) only when the refusal brought a change, and
        # neither when the cart breaks the budget in force
        for attempt in range(2):
            if attempt:
                if self.trusted and len(self.changes) == seen:
                    break
                if self.relaxed == self.version and self.brief.get("pick") == "cheapest":
                    swept = self.price_sweep("change")
                    if swept:
                        replies.append(swept.strip() + " Put the cheapest listing that meets every requirement in force in the cart and check out.")
                        break
                self.reread_offers()
                change = self.after_change()
                if change:
                    replies.append("[Kyo] " + change)
                self.check_cart(list(lines), self.soft if self.trusted else self.hard)
                if not self.verdicts.get(self.check_key(list(lines))):
                    swapped, said = self.swap_better()
                    if said:
                        replies.append("[Kyo] " + said)
                    if swapped:
                        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
                # a per-item budget is broken by a unit price over it, whichever way 'each' is meant
                each = self.brief.get("budget_scope") == "each" and isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.cart_broken() or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines) or (each and any((self.price_of(i) or 0.0) > self.budget for i in lines))):
                    break
            # lines code carted (closest-listing buy, late rescue, decline review) or swapped get their pages opened
            # first; the first checkout keeps one step for its retry
            self.page_first(list(lines), self.tail() + (1 if attempt == 0 else 0))
            if self.finished or self.left() < 1:
                break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            self.near_before = self.near_state()
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")) + self.carried(reply, "checkout"))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): " + ("a closest-listing review found " if getattr(self, "trusted", False) else "the independent check confirmed ") + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
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
            if isinstance(value, str) and _id_like(value, True):
                fixed, problem = self.near_id(value.strip())
                # refused once: the same value sent again goes to the store, whose reply is the truth
                if problem and f"{name}|{value.strip()}" in self.id_refused:
                    return value, ""
                if problem:
                    self.id_refused.add(f"{name}|{value.strip()}")
                return fixed, problem
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

    def id_shape(self, value: str) -> bool:
        """Whether a value has the seen listing ids' shape: their common start and one of their lengths (any value while
        fewer than three ids are known)."""
        ids = list(self.rows)[:200]
        return len(ids) < 3 or (value.startswith(_common_prefix(ids)) and len(value) in {len(i) for i in ids})

    def near_id(self, value: str) -> tuple[str, str]:
        """An id-like value that is no seen listing id: (the value to send, why it is refused or ""). A case slip of one
        id is repaired; an id-shaped value the request, a record or a tool result names is sent; a cut, extension or
        one-character slip of exactly one seen id (8 characters or more) is repaired; anything else is refused."""
        low = value.lower()
        same = [i for i in self.rows if i.lower() == low]
        if len(same) == 1:
            return same[0], ""
        shaped = self.id_shape(value)
        # the value as a whole token, not a piece of a longer id
        word = re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.I)
        if shaped and (word.search(self.request) or any(word.search(_dump(r)) for r in self.rows.values())
                       or any(m.get("role") == "tool" and word.search(str(m.get("content") or "")) for m in self.chat)):
            return value, ""
        if len(value) >= 8:
            slips = [i for i in self.rows if _one_edit(low, i.lower())
                     or (len(i) >= 8 and (i.lower().startswith(low) or low.startswith(i.lower())))]
            if len(slips) == 1:
                self.trace.append({"kind": "id_fixed", "from": value, "to": slips[0]})
                return slips[0], ""
        if shaped:
            return value, f"{value!r} is not the id of any listing shown so far; copy an id exactly from a result row"
        named =[i for i in reversed(self.seen_order) if word.search(_dump(self.rows.get(i)))][:6]
        if named:
            return value, f"{value!r} is not a listing id but a word on the records of {', '.join(named)}; pass one of those ids"
        return value, f"{value!r} is not a listing id" + (f" (ids look like {self.seen_order[-1]})" if self.seen_order else "")

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        if name in self.cart_rule:
            return self.cart_rule[name] == "set"
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
            options = (spec.get(key) or {}).get("enum") if isinstance(spec.get(key), dict) else None
            if isinstance(options, list) and len(options) >= 2 and value == self.mode_values(name, key, options)[1]:
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
            elif isinstance(value, dict) and (_map_spec(spec.get(key)) or (value and all(k in self.rows for k in value))):
                # a {listing: units} map is the whole cart
                whole = True
                lines += [(k, _count(v)) for k, v in value.items() if k in self.rows and not isinstance(v, str)]
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                one_unit = not any(isinstance(sp, dict) and "integer" in _kinds(sp) for sp in spec.values())
                if count is None:
                    # an optional count left out takes off its schema default, else one unit
                    unit = next((_count(sp.get("default")) for sp in spec.values() if isinstance(sp, dict) and "integer" in _kinds(sp)
                                 and (_count(sp.get("default")) or 0) > 0), None)
                    have = have - (unit or 1)
                else:
                    have = have - count if count else (have - 1 if one_unit else 0)
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
            return min(spec["maxItems"], max(2, self.compare_bad - 1)) if self.compare_bad else spec["maxItems"]
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
            if name == self.roles.get("search"):
                self.read_page_param(name, args)
            jump = self.skipped_start(name, args)
            if jump:
                trimmed = f"\n[Kyo] Left out {jump}={args[jump]}: a new search starts at its first page (the store had handed out no such page for it, so the rows before it would go unseen)."
                args = {k: v for k, v in args.items() if k != jump}
            # the paging parameter still unknown: a later page asked on a search's first call is caught after the reply
            blind = name == self.roles.get("search") and not self.page_param(name)
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed += f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                if self.is_buy(name, args) and fresh and not self.cart_ids() and (self.empty_holds or self.left() <= self.buy_steps() + 2):
                    done = self.near_buy()
                    if done:
                        out.append((cid, done + (" [episode finished]" if self.finished else "")))
                        continue
                if self.is_buy(name, args) and fresh and not self.cart_ids() and self.empty_holds < 2:
                    self.empty_holds += 1
                    out.append((cid, "Not run (no step used): the store's cart is empty. Put each chosen listing in the cart at its requested quantity first"
                                + (f" (fitting listings named so far: {', '.join(self.leads[:4])})" if self.leads else "") + ", then check out."))
                    continue
                if self.is_buy(name, args):
                    # a price change recorded since the last such re-read: the cart lines are re-read first and the cart
                    # is viewed once, so the change below is judged on the new prices
                    early, view_due = "", False
                    if any(c["kind"] == "price" and c["step"] > self.price_done for c in self.changes):
                        self.price_done = self.step
                        early, view_due = self.reread_offers(), True
                        if self.price_wait >= 0:
                            if any(self.rose_at.get(i, -1) >= self.price_wait for i in self.cart_ids()) or self.cart_broken():
                                self.rearm()
                            self.price_wait = -1
                    # a change not handled yet comes first (a total over the budget, the rows a raise lets in), then the wider search
                    change = self.after_change()
                    if change:
                        out.append((cid, "Not run yet (no step used). The task changed: " + change
                                    + ("\nKyo re-read the cart's listings just now:\n" + early if early else "")
                                    + "\nCart now: " + self.cart_text() + "\nFix the cart if the change asks for it, then check out."))
                        continue
                    # the code sweep further down replaces the walker's wider search, except in a store without a capped
                    # listing while many steps are left (a next page of one search seldom reaches the cheapest fit)
                    ready = self.sweep_ready()
                    if ready:
                        self.price_cap_tool()
                    # with a capped listing tool the code sweep does that search: no hold after it ran
                    wide = not self.cap_tool and (not ready or self.left() >= 14 + 2 * len(self.items))
                    wider = self.wider_hold() if wide else ""
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = "\n".join(t for t in (early, self.reread_offers()) if t)
                    if (view_due or self.cart_touched and self.cart_seen_at < self.cart_at) and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            self.saw_cart(self.cart_view)
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
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    swept = self.price_sweep("order")
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + ([] if swept else self.cheaper_rows())
                    verdict = self.check_cart()
                    if verdict and self.cheaper_fits:
                        fixed, said = self.swap_better(self.cheaper_fits, need_change=False, why="(the review found it fits and is cheaper)", mode="before")
                        self.cheaper_fits = []
                        if fixed:
                            trimmed = "\n[Kyo] " + said + trimmed
                            verdict = ""
                    if verdict:
                        fixed, said = self.code_fix()
                        if not fixed and not said:
                            fixed, said = self.swap_better()
                        if fixed:
                            trimmed = "\n[Kyo] " + said + trimmed
                            verdict = ""
                        elif said:
                            verdict = said
                    if not verdict and self.verdicts.get(self.check_key(self.cart_ids())) is False:
                        # the same checkout again for a cart the check found wrong: code's fixes first, once per cart
                        state, said = self.repair_cart()
                        if state == "ended":
                            out.append((cid, said + (" [episode finished]" if self.finished else "")))
                            continue
                        if state == "held":
                            out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                            continue
                        if said:
                            trimmed = "\n[Kyo] " + said + trimmed
                    if not verdict and self.clash_blocks < 2:
                        # free of model calls, so it also stands when the spend caps skipped the check: a line whose
                        # records print another code than a requirement's (CL32 for cl30) cannot pass
                        clash = self.line_clashes()
                        if clash:
                            self.clash_blocks += 1
                            self.trace.append({"kind": "clash_block", "lines": clash})
                            verdict = ("Kyo's own reading of the cart's records: " + "; ".join(clash) + ". Replace that line with a listing"
                                       " whose records meet the requirement; when no listing of that item meets it, end without a purchase.")
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        seen += (swept.strip() + "\n") if swept else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.soft:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
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
                            self.check_cart([first], self.hard)
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
            if self.is_refuse(name, args) and self.left() <= self.buy_steps() + 2 and self.cart_short():
                done = self.near_buy(fill=bool(self.cart_ids()) and not self.cart_broken())
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.rescued and not self.cart_ids() and self.roles.get("cart") and self.roles.get("buy"):
                self.rescued = True
                done = self.late_rescue()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart() or self.decline_widen() or self.near_buy()
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
            if name == self.roles.get("cart") and not ending:
                adds = self.adds_of(name, args)
                unread = [i for i in adds if i not in self.opened and i not in self.gone]
                held = [i for i in dict.fromkeys(self.cart_ids()) if not self.sold_out(i)]
                bare = max(0, len(self.items) - len(set(held) | set(adds)))
                # a page costs a step: opened now only while the items not carted yet keep about four steps each
                if unread and self.left() - self.tail() > len(unread) + 4 * bare + 2:
                    said = self.page_first(unread, self.tail() + 1)
                    opened = [i for i in unread if i in self.opened]
                    lost = [i for i in opened if not self.buyable(i)]
                    if lost:
                        out.append((cid, f"Not run: Kyo opened the page of {', '.join(opened)} before carting it ({len(opened)} step(s) used), and "
                                    + ", ".join(lost) + " can no longer be bought." + (" " + " | ".join(said) if said else "")
                                    + " Cart another fitting listing for that item."))
                        continue
                    if said:
                        trimmed += "\n[Kyo opened the page before carting] " + " | ".join(said)
                    elif opened:
                        trimmed += f"\n[Kyo opened the page of {', '.join(opened)} before carting it ({len(opened)} step(s) used)]"
            upto = len(self.calls_made)
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            # the messages that came with the reply: all of them are shown, the unread ones go to the note reader
            unread = self.take_unread(reply.get("said"))
            brought = list(dict.fromkeys([t for t in _items(reply.get("said")) if isinstance(t, str)] + unread))
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
            if blind and not ending:
                self.read_shape(name, observation)
                trimmed += self.read_start(name, args, upto)
            if name not in plain and not ending and self.records_in(observation):
                self.searches = (self.searches + [(name, dict(args), observation)])[-12:]
                if name == self.roles.get("search"):
                    self.note_mode(name, args)
                    self.note_hits(name, args, observation)
                self.sweep_calls.add(name + json.dumps(args, sort_keys=True, default=str))
                prices = [self.price_of(i) for i, _ in self.records_in(observation)]
                if len(prices) >= 3 and None not in prices:
                    tally = self.order_seen.setdefault(name, [0, 0])
                    tally[0] += self.price_sorted(observation)
                    tally[1] += 1
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
                before = dict(self.cart_lines)
                self.note_cart(name, args)
                self.learn_cart(name, args, observation, before)
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
                try:
                    fit = self.fit_note()
                except Exception as error:      # an advisory reading never ends the episode
                    self.trace.append({"kind": "fit_error", "error": repr(error)[:200]})
                    fit = ""
                if fit:
                    trimmed += "\n" + fit
            if self.roles.get("cart") and name == (self.remover() or ("",))[0]:
                self.took_off(name, args, observation)
                self.recent = []
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
                self.saw_cart(observation)
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            if self.is_buy(name, args) and not reply.get("finished") and not self.finished and brought:
                trimmed += "\n[Kyo] The checkout did not go through: the store sent the messages above instead. The episode goes on."
            if self.is_refuse(name, args) and not reply.get("finished") and not self.finished:
                # the store turned the ending down (the shopper changed the task): the next ending is reviewed afresh
                self.audited = self.widened = self.pushed = False
                self.refuse_blocks = 0
                rescued = ""
                if unread:
                    # the change is read now, so the review right after works on the requirements in force
                    meaning = self.take_note(unread)
                    if meaning:
                        trimmed += "\n[note read] " + meaning + "."
                    rescued = self.after_refusal()
                if not self.finished:
                    trimmed += ("\n[Kyo] The store did not take ending without a purchase: the shopper changed the task. Apply the change (a new"
                                " budget or a dropped requirement can make a listing you already saw fit), look again at the listings you opened,"
                                " and buy the one that fits.")
                if rescued:
                    trimmed += "\n" + rescued + (" [episode finished]" if self.finished else "")
            text = self.show(observation) + _said(brought) + paged + trimmed
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
            lines = [line for line in dict.fromkeys(unread + self.loose_lines(observation)) if line not in self.heard]
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
                    if self.relaxed == self.version:
                        text += self.price_sweep("change")
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in " ".join(brought) + " " + self.spoken(observation) for line in self.sold_lines)):
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
        """End the episode, with a purchase whenever one is possible. A cart that misses an item, holds a sold-out line or
        breaks a total budget first gets the closest listings (a held, unbroken cart: only its missing items). A held
        cart is then checked out; a checkout the store turns down is returned for the walker with `again`, else it is
        picked again once when the refusal changed something, and checked out again. The episode ends without a purchase
        only when no_fit holds with no buyable line in the cart (or a broken one), or every checkout was turned down."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse, cart = self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("cart")
        start = set(self.cart_ids())
        # a held line that contradicts a requirement by a code it prints (CL32 for cl30) cannot pass: every item is picked
        # again (the closest-listing pick passes over such listings), and the episode ends without a purchase when no
        # item has a listing left
        clashing = bool(self.line_clashes())
        if buy and cart and (self.cart_short() or self.cart_broken() or clashing):
            holding = bool(self.cart_ids()) and not self.cart_broken()
            # the audit and its fresh check only while their calls leave the closest-listing buy its time (an empty or
            # broken cart only, as before: a full cart that only clashes goes straight to the closest-listing pick)
            if not holding and not self.rescued and self.left() >= 2 and time.time() - self.t0 < LATE - 30:
                self.rescued = True
                self.late_rescue()
                clashing = bool(self.line_clashes())
            if not self.finished and self.left() > 0 and (self.cart_short() or self.cart_broken() or clashing):
                done = self.near_buy(fill=bool(self.cart_ids()) and not self.cart_broken() and not clashing)
                if done and again and not self.finished and self.left() > 1:
                    return "the checkout was refused: " + done
                if clashing and not self.finished and self.line_clashes():
                    # the closest-listing pick could not replace the clashing line (too few steps, or that state was picked
                    # once already): a cart known to break a requirement cannot pass, so the ending is without a purchase
                    self.no_fit = True
            if self.finished or self.left() <= 0:
                return ""
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # nothing buyable held and an item with nothing buyable: the ending without a purchase is the one that can pass
        # (so is a held cart that still contradicts a requirement by a printed code when no closest listing was left)
        hopeless = self.no_fit and (not [i for i in self.cart_ids() if not self.sold_out(i)] or self.cart_broken() or bool(self.line_clashes()))
        # after the fill above, a cart still missing an item the store showed is a certain fail: end without a purchase
        # (cart_ids() reads the fresh cart view, else Kyo's record of the cart calls, the same source as `holding`)
        gap = self.missing_items() if refuse and (holding or self.cart_touched) else []
        if gap:
            self.trace.append({"kind": "gap_finish", "missing": gap})
        if buy and not hopeless and not gap and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults and cart:
                self.final_review({i for i in self.cart_ids() if i not in start})
                if self.finished or self.left() <= 0:
                    return ""
            for attempt in range(0 if faults else 2):
                if attempt and refuse and (self.cart_broken() or time.time() - self.t0 + 2 * self.store_lat > HARD_END - 5):
                    # the same cart holding a sold-out line (or over the budget) cannot pass, and a retry that leaves the
                    # ending no time loses the episode: end without a purchase instead
                    self.trace.append({"kind": "forced_retry_skip", "broken": self.cart_broken()})
                    break
                # the pages of lines never opened, keeping the finish's steps (and the first checkout's retry step)
                self.page_first(self.cart_ids(), self.tail() + (1 if attempt == 0 else 0))
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                self.near_before = self.near_state()
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0 or reply is None:
                    return ""
                self.buys += 1
                self.refused_at = self.step
                self.observe(buy, reply.get("observation"))
                if again:
                    return "the checkout was refused: " + self.show(reply.get("observation")) + self.carried(reply, "checkout")
                # the last round: a refusal that changed something (the task, the budget, a sold-out line) gets one re-pick
                if attempt == 0 and cart and self.near_state() != self.near_before:
                    same_needs = self.near_state()[-1] == self.near_before[-1]
                    bought = self.buys
                    self.near_buy(fill=same_needs and not self.cart_broken())
                    if self.finished or self.left() <= 0:
                        return ""
                    if self.buys > bought:
                        # the re-pick checked out and was turned down: the same cart is not sent once more
                        break
        if refuse and not self.finished:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200), "no_fit": self.no_fit})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation")) + self.carried(reply, "ending")
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
        self.soft, self.hard = SPEND_BY_ITEMS.get(len(self.items), SPEND_MANY)
        self.trace.append({"kind": "brief", "brief": self.brief, "spend": [self.soft, self.hard]})
        opening = _read_records(self.view.get("opening"))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(self.loop_cap, self.hard)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.hard + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls,
                           "by_role": {k: round(v, 5) for k, v in self.spent_by.items()}, "sonnet": dict(self.sonnet_used),
                           "fails": dict(self.fails), "dead": sorted(self.dead)})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        self.walk_caps = (loop_cap, spend_cap)
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and time.time() - self.t0 < DEADLINE):
            if self.commit_due():
                done = self.near_buy(fill=bool(self.cart_ids()) and not self.cart_broken())
                if done and not self.finished:
                    self.chat.append({"role": "user", "content": "[Kyo] Only the steps a purchase takes were left, so Kyo bought the closest listings itself: " + done + "\n" + self.status()})
                    continue
                # a held, unbroken cart stays the walker's to check out; with nothing buyable held the episode ends now
                if self.finished or not self.cart_ids() or self.cart_broken():
                    break
            if not self.chain_started:
                self.chain_started = True
                if self.chain_due():
                    text = self.resolve_refs()
                    if text and self.chat and self.chat[-1].get("role") == "user" and isinstance(self.chat[-1].get("content"), str):
                        self.chat[-1]["content"] += "\n\n" + text + "\n" + self.status()
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            # the turn, its resends and the spare model return by LATE, so the endgame after it still fits before HARD_END
            self.call_until = self.t0 + LATE
            try:
                message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            finally:
                self.call_until = 0.0
            if message is None:
                self.misses += 1
                if self.no_credit or {LIGHT, SPARE} <= self.dead or self.calls >= self.all_calls or self.misses > 2:
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
            if (not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls)
                    and self.left() > 8 and time.time() - self.t0 < DEADLINE - 60):
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
