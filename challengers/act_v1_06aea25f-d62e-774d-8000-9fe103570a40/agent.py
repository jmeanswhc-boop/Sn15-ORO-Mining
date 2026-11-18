from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

RUN_TARGET = 3.6
RUN_TASKS = 90
TASK_SECONDS = 300
CLOSE_SECONDS = 55
RESERVE_SECONDS = 30
JUDGE_ITEMS = 3
JUDGE_ROOM = 1.25
CEILING_SHARE = 3.75
CLOSE_OUT = True

WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 250000
GATE_ROUNDS = 3

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

GUIDE = """You are Act, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Act knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by the interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Act carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through the interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: the required words must appear in the value in the same order (case, spaces, hyphens and underscores aside, singular or plural fine), so a value that contains them counts ("deep teal" meets teal, "matte black steel" meets black steel), while words split apart or in another order do not ("black matte steel" is not black steel). A different word does not count, even a near synonym or another form of the word (an abbreviation or code is not the word it stands for, "teal green" is not "sea green", "woollen" is not "wool"). Parts joined by a plus sign name one combination, not a list. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Act adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title counts; the description alone does not, except for what a listing says it fits or works with. A requirement no field of the listing states is not met: buy such a listing only when no listing shows the value. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). A fit statement can name a whole family by a pattern, such as a code whose last digits are written as letters ("every AB 12xx model" covers AB 1200 to AB 1299): compare the pattern's fixed part with the other listing's code. When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Act reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
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

REVIEW = """A shopping agent's first reading of a shopper's request is shown as JSON after the request. Check that reading against the request's own words and return the corrected reading as ONE JSON object (compact, nothing else) with exactly these keys: "items", "lookups", "process", "pick", "pick_measure", "changes". Keep every part of the first reading that passes the tests below, in its own wording and shape; change only what a test shows wrong.
TESTS
- items: one entry per product the shopper wants bought. A product the request gives one quantity is one item even when its description joins words with "and" or names a whole store department; split an entry only where the request gives each product its own quantity. "name" is the kind of product in the request's own words taken whole: when the request names a store category or department, the name is that whole label and never one part of it or a narrower kind than the request states. "shelf" stays as read unless the store's labels listed below hold the exact label for that item. "quantity" is the number of units of that one listing.
- musts: each must is a property requirement the request states on that product: a value, a limit, an exclusion, a closed list, a rule that depends on another item of this order, or a value taken from another listing. Remove a must that only restates the story around the request, the budget, a price, stock or availability, the quantity, or that only describes the other listing a pointing must refers to (those words belong inside the pointing must). Keep the rest word for word.
- lookups: a must belongs here only when its value is to be read off the page of another listing in the store that the request points at, picked out by its shop, kind or properties and not bought in this order. Fitting or matching another item of this order, and a condition on another item's property, are plain musts and never lookups.
- process: a step belongs here only when the request obliges the agent to do something besides finding and buying the items, such as getting approval before the order, telling or showing the shopper something, reading something first or using a particular tool. The story, the shopper's own promises and remarks, an offer to answer questions, the limits on messages and steps, and the way to end when nothing fits are not steps. "messages" counts only the messages to the shopper that the step itself requires, 0 when it needs none.
- pick: "cheapest" only when the request tells the agent to choose among the fitting listings by the lowest price or total; "most" or "least" only when it tells the agent to choose by the extreme of a stated measure of the product, named in "pick_measure"; otherwise "any". The rule for what to put forward when nothing fits, and remarks that money is tight, are not a pick rule.
"changes": one short note per change made, each quoting the request words it rests on; [] when nothing changed."""

HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number in the unit the request uses for money (a limit given in cents stays a count of cents), or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value.
- A reply that rules out everything other than, apart from or except one value requires that value ("anything with processor apart from tensor gets binned" means processor: tensor); only a reply that rules out the value itself excludes it ("not wicker", "anything but led"). When the words after "except", "other than" or "apart from" themselves begin with "anything other than", "anything but" or "not", the reply excludes what follows those words, and the need keeps them ("nothing else will do except anything other than the value shown on the page for X" means gender: anything other than the value shown on the page for X)."""

NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""

CHECK = """You are the last check before a shopping agent checks out. Grade every requirement of each cart line from that line's own record, shown below, and nothing else, with one exception: a requirement that the item fit, suit or work with another item of this order is graded from two records, this line's statement of what it fits (the models, names or part series it works with; a family pattern such as "every AB 12xx model" covers every code sharing the fixed part) and the record of the cart line bought for that other item (its model number, the names it is also sold as, the part series it says it fits, the codes in its title): "exact" only when a code or series named by one record appears in the other, "contradicted" when the other item's codes are named but excluded, "absent" when either record names nothing; give that grade six elements, adding the other line's id and the words quoted from its record. Quantities, the number of lines, stock, the budget, the steps the request asks for and the messages to the shopper are checked by code elsewhere: do not judge them, and never treat a sentence of the request's story as a requirement; the numbered requirements are the only ones.
Grades, one per requirement per line: "exact" (a field for that property states the required value, limit or list member; units converted, limits include the edge, a range must lie wholly inside); "words" (the required words appear together, in that order and unbroken, inside the value of a field for that property, case and singular/plural aside: "deep teal" for teal, "modern classic" for modern, "faux suede" for suede, one element of a comma list is enough, and a value standing among others in a title or list such as "Cushion and Round Cut" for cushion counts; words broken apart by other words such as "wheel cylinder cup" for wheel cup, a different word or a near synonym are not "words"); "category" (the category label itself states it); "absent" (no field for that property states it, and neither the title nor another field holds the wanted value); "contradicted" (a field for that property states a different value: quote it). Rules: a field whose key names the same property decides, even when the title or another field says otherwise; a key for a different property, even a related one (a handle colour is not the body's colour, a maximum speed is not the speed), decides nothing; when no field names the property, the wanted value in the title or under another field counts, the description alone does not, except for what the listing says it fits (a family pattern such as "every AB 12xx model" covers every code sharing the pattern's fixed part); a bare figure that does not say what it measures counts for nothing, a figure inside a model code or product name states no measurement, and a count written as a figure is not the word naming that count. A requirement that applies only in some case (a condition on this listing or on another item of the order) is met when that case does not hold: read the condition first, from the record it is about (for another item, the cart line bought for it), and when it does not hold grade the requirement "exact" quoting the words that show it. A requirement whose value is read from another page is graded against the value the requirements list gives for it; without that value it is "absent". A rule of being unlike another listing's value is "exact" unless the record shows that same value. The newest record of a listing wins.
"label": "names" when the line's category label names the requested product itself; "neighbouring" when another label seen in the store names it better and this one names a broader, narrower or neighbouring kind.
CHEAPER LISTINGS, when shown, stand under the cart line of the requested item they could replace and cost less than it: grade each one against every requirement of that item, from its own record, reading a requirement that names another item against the cart's line for that item as it is now: "fits" when its record meets every requirement (it should then replace the line), else "fails" with the requirement it fails and the quote that shows it. When the pick rule is most or least, name in "better" any other listing shown whose record meets every requirement and beats the cart's line on that rule.
Return ONE JSON object and nothing else: {"lines": [{"id": listing id, "item": the number of the requested item this line is for, "label": "names" or "neighbouring", "grades": [[requirement number such as "R1.2", grade, field name or "title", quote of at most 12 words from the record, and for a fit requirement the other cart line's id and the words quoted from its record]]}], "cheaper": [[listing id, "fits" or "fails", requirement number or "", quote]], "better": [ids of other listings shown that should replace a wrong line, best first]}"""

READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a comma list is enough, the required words appearing in the value in the same order count, such as "deep teal" for teal, a different word or a near synonym does not, unit conversions count, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title counts, the description alone does not except for what the listing says it fits (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a fit statement naming a family by a pattern ("every AB 12xx model") covers every code sharing the pattern's fixed part; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. That ending is right only when the store truly has no listing that meets the requirements, which is rare. Check the listing records shown against the requirements in force, with these reading rules: """ + READING + """
The records are grouped under the requested item they belong to (ITEM n). Grade every listing of a requested kind against each of that item's requirements: stated as met (the record's own words show the value), not stated, or contradicted. A requirement is contradicted only when the listing's own record states a different value for that same property: quote those words. A requirement the record does not mention is not stated, never contradicted. Records marked ROW were seen only as a search result row, so most requirements are not stated for them.
Return ONE JSON object and nothing else: {"items": [{"item": item number, "fit": [ids of that item's OPENED listings whose records state every requirement in force as met, cheapest first, at most 4], "near": [ids of that item's OPENED listings with no contradicted requirement but with some requirement their records do not state, fewest unstated first, then cheapest, at most 4], "contradicted": [{"id": listing id, "requirement": "...", "quote": "the record's own words"}]}], "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "fallback": true when the request itself says to put forward the closest or nearest listing when nothing fits, else false, "proven": true only when some requested item has listings shown below and every one of them has a contradicted requirement, else false, "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
A link is confirmed only when an OPENED record states every detail the description gives for it (its kind, its shop, each named property value); a record of that shop that differs on a named detail or leaves it unstated is not that link, and another listing of the same shop may be. When a page states a rule that turns its own value into the wanted one (pairs with, goes with, the opposite of), the value is what that rule gives.
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when an OPENED record is confirmed as that link, else "", "about": how the requirement or the previous page describes that listing, "candidates": [ids of ROW or OPENED records that could be this link but are not confirmed, likeliest first, at most 4], "search": a short search phrase that would list that listing (its shop's name and its kind), or ""}], "done": true when the last link is a confirmed OPENED page that states the property's value itself, else false, "value": the property's value as that last page gives it, or ""}]}
A detail counts as stated when the record gives it in other units, spelling or word order (10 tb for 10000 gb; the shop named in a title, note or shop field), and a search row showing that shop and kind is a candidate to open.
List each chain's links in order and stop at the first link that is not confirmed, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

BASKET = """A shopping agent buys one listing for each requested item, and some requirements of one item name another item of the same order: it must fit, suit or work with the listing bought for that other item, or it holds only when that other listing has some property. Grade the candidate listings below from their own records, with these reading rules: """ + READING + """
"own": grade every candidate against the requirements of its own item that do not name another item: "fits" when its record meets each of them, else "fails" with the requirement it fails and the quote that shows it.
"pairs": for every candidate of an item with a requirement that names another item, once per such requirement, list every candidate of the named item with which the requirement is met when both are bought. Read it from both records: a fit holds only when a code, model or part series that one record says it fits, works with or is sold as appears in the other record and no exception the record names excludes it (a family pattern such as "every AB 12xx model" covers every code sharing the fixed part); a requirement that applies only in some case holds with every candidate of the named item for which that case does not hold, and with the others only when the record meets it. Give an empty list when it holds with none.
Return ONE JSON object and nothing else: {"own": [[listing id, "fits" or "fails", requirement number or "", quote of at most 12 words]], "pairs": [[listing id, requirement number, [ids of the named item's candidates with which it holds]]]}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null, "query": the tool parameter that takes the search words (free text), or null, "cap": the tool parameter that keeps out listings priced above a number it is given, or null}"""

ASK_PLAN = """You decide which properties a shopping agent asks a shopper about. The shopper holds a few firm preferences the request never mentions, each about one property of one requested product. A question reveals a preference only when it names that exact property of that product. A question about a property the shopper does not care about, about two things at once, or about the same property again earns nothing, and every question uses one of a few messages.

You get the request, each item with what the request already settles for it, and for each item the attribute keys its opened listings carry: how many of those listings carry the key, how many different values they show, and a few of the values. Only a few listings are opened, so a key that shows one value there may differ across the store and can still be the shopper's firm preference.

For each item, rank the keys worth asking, best first:
- Use only keys from that item's own table, written exactly as the key reads (every word of it, plain spelling). For an item with no opened listing yet, name the properties a buyer of that kind of product chooses by, in plain words.
- Leave out a key whose value the request already settles for that item, a second name for a key already chosen, identifiers, codes, model or part numbers, warranty, shipping, stock, and keys whose values list purchase variants or bundle several properties.
- Prefer the keys that the most listings carry (the table lists them that way, most carried first); among those, the properties a buyer of that kind of product chooses by and states as a wish. A single value across the opened listings is no reason to leave a key out.
- When the request's story leaves a property open that the item's table carries (who will use it, where it goes, its looks, what it holds), rank that key first.
- What the request explicitly asks to check with the shopper (ASK FIRST) comes before everything, named by the key it is about.
Rank PICKS keys per item when its table has that many, fewer only when the table is shorter.
Return ONE JSON object and nothing else: {"items": [{"item": item number, "ask": ["key", ...]}]}"""

class Text:
    @staticmethod
    def words(key: Any) -> str:
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
        return " ".join(re.split(r"[\s_.\-]+", spaced)).strip().lower()

    @staticmethod
    def single(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") else word

    @staticmethod
    def stem(text: Any) -> frozenset[str]:
        if isinstance(text, frozenset):
            return text
        return frozenset(map(Text.single, Text.words(text).split()))

    @staticmethod
    def bag(text: Any, ordered: bool = False) -> Any:
        found = [Text.single(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]
        return found if ordered else frozenset(found)

    @staticmethod
    def phraseIn(value: str, quote: str) -> bool:
        quoted = Text.bag(quote, True)
        for part in re.split(r"\bor\b|,|/|;", str(value).lower()):
            want = Text.bag(part, True)
            if not want or len(want) > 4 or any(w.isdigit() for w in want):
                return True
            if any(quoted[i:i + len(want)] == want for i in range(len(quoted) - len(want) + 1)):
                return True
        return False

    @staticmethod
    def names(prop: Any, key: Any) -> bool:
        prop, key = Text.stem(prop), Text.stem(key)
        if not prop or not key:
            return False
        return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

    @staticmethod
    def items(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        return [value] if isinstance(value, (str, dict)) and value else []

    @staticmethod
    def count(value: Any, default: Any = None) -> Any:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else default
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
            return int(float(value))
        return default

    @staticmethod
    def clip(value: Any, limit: int, depth: int = 0) -> Any:
        if depth > 7:
            return "..."
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + "..."
        if isinstance(value, (dict, list)):
            pairs = value.items() if isinstance(value, dict) else enumerate(value)
            clipped = [(k, Text.clip(v, limit, depth + 1)) for k, v in pairs]
            return dict(clipped) if isinstance(value, dict) else [v for _, v in clipped]
        return value

    @staticmethod
    def prose(observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    found.append(node)
            elif isinstance(node, (dict, list)):
                children = list(node.values()) if isinstance(node, dict) else node
                stack.extend((child, depth + 1) for child in reversed(children))
        return " ".join(found)[:800] if found else render(observation)[:400]

def render(value: Any, limit: int | None = None) -> str:
    if limit is not None:
        value = Text.clip(value, limit)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)

class Budgets:
    def __init__(self, maxSteps: int, items: int, seed: str):
        self.started = time.monotonic()
        self.seed = sum(ord(c) for c in seed) if seed else 0
        self.modelTimes: list[float] = []
        self.envTimes: list[float] = []
        self.resize(maxSteps, items)

    def resize(self, maxSteps: int, items: int) -> None:
        size = max(0.8, min(1.5, (max(10, maxSteps) / 45.0) ** 0.5 * (1.0 + 0.1 * max(0, items - 1))))
        base = RUN_TARGET / RUN_TASKS
        self.spendSoft = round(base * size * 1.05, 4)
        self.spendHard = round(base * size * 1.6, 4)
        self.allCalls = int(TASK_SECONDS / 2.5)
        self.autoPages = int(max(4, min(12, maxSteps // 9)))
        self.cheaperOpens = 3 if maxSteps < 60 else 4
        self.coverPages = int(max(2, min(8, maxSteps // 7)))

    @staticmethod
    def held(host: Any) -> int:
        return len([i for i in host.shelf.cart_ids() if not host.shelf.sold_out(i)])

    @staticmethod
    def auditRoom(hard: float, wide: bool, items: int, held: int) -> float:
        room = hard * (1.3 if wide else 1.0)
        return min(room, JUDGE_ROOM * RUN_TARGET / RUN_TASKS) if items >= JUDGE_ITEMS and held < items else room

    @staticmethod
    def closeOut(cost: float, items: int, held: int) -> bool:
        return CLOSE_OUT and items >= JUDGE_ITEMS and held < items and cost >= JUDGE_ROOM * RUN_TARGET / RUN_TASKS

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def timeLeft(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def noteModel(self, seconds: float) -> None:
        self.modelTimes = (self.modelTimes + [seconds])[-12:]

    def noteEnv(self, seconds: float) -> None:
        self.envTimes = (self.envTimes + [seconds])[-20:]

    def modelSeconds(self) -> float:
        return sum(self.modelTimes) / len(self.modelTimes) if self.modelTimes else 6.0

    def envSeconds(self) -> float:
        return sum(self.envTimes) / len(self.envTimes) if self.envTimes else 1.0

    def reserve(self) -> float:
        return max(float(RESERVE_SECONDS), min(float(CLOSE_SECONDS), 10.0 + 2.5 * self.modelSeconds() + 4.0 * self.envSeconds()))

    def closing(self) -> bool:
        return self.timeLeft() < self.reserve()

    def canCall(self) -> bool:
        return self.timeLeft() >= 0.5 * self.reserve() + self.modelSeconds()

    def turnsLeft(self, stepsPerTurn: float) -> int:
        each = self.modelSeconds() + max(1.0, stepsPerTurn) * self.envSeconds()
        return max(0, int((self.timeLeft() - self.reserve()) / max(1.0, each)))

    def callRoom(self, cost: float, cap: float) -> int:
        if cost >= cap:
            return 0
        return max(0, int((self.timeLeft() - self.reserve()) / max(1.0, self.modelSeconds())))

    def jitter(self, tries: int) -> float:
        return 0.8 + 0.4 * (((self.seed * 31 + tries * 17) % 97) / 96.0)

class Item:
    def __init__(self, name: str, shelf: str = "", qty: int = 1, musts: list[str] | None = None):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts if musts is not None else []
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

HEADS = (BRIEF, REVIEW, HEAR, CHECK, AUDIT, TRAIL, ASK_PLAN)
HEAD_CHARS = 1800
TRIM_TO = 0.7
TRIM_KEEP = 4

class ModelGate:
    def __init__(self, host: Any):
        self.host = host
        self.plain_models: set[str] = set()
        self.no_credit = False
        self.waited = 0
        self.cost = 0.0
        self.calls = 0

    @staticmethod
    def parse(text: Any) -> dict[str, Any] | None:
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
            for piece in (text[left:right + 1], re.sub(r",(\s*[\]}])", r"\1", text[left:right + 1])):
                try:
                    got = json.loads(piece)
                    return got if isinstance(got, dict) else None
                except ValueError:
                    pass
        decoder = json.JSONDecoder()
        found = None
        position = 0
        while 0 <= left < len(text):
            try:
                got, end = decoder.raw_decode(text, left)
                if isinstance(got, dict):
                    found = got
                position = max(end, left + 1)
            except ValueError:
                position = left + 1
            left = text.find("{", position)
        return found

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.no_credit or self.cost >= CEILING_SHARE * RUN_TARGET / RUN_TASKS or self.calls >= self.host.budgets.allCalls or not self.host.budgets.canCall():
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, effort)
                if message is not None:
                    return message
                if status in (400, 422) and effort and model not in self.plain_models:
                    self.plain_models.add(model)
                    continue
                if status == 0 and tries == 0 and effort != "low":
                    effort = "low"
                    tries += 1
                    continue

                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)

                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = min(WAITS[tries] * self.host.budgets.jitter(tries), max(0.0, self.host.budgets.timeLeft() - self.host.budgets.reserve()))
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        elif model.startswith("anthropic/") and len(messages) == 1 and isinstance(messages[0].get("content"), str):
            text = messages[0]["content"]
            cut = ModelGate.headLength(text)
            if cut:
                sent = [dict(messages[0], content=[{"type": "text", "text": text[:cut], "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": text[cut:]}])]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if effort and model not in self.plain_models:
            body["reasoning"] = {"effort": effort}
        began = time.monotonic()
        result = _llm.post_verbose("/inference/chat/completions", body)
        self.host.budgets.noteModel(time.monotonic() - began)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
            paid = [float(v) for v in (usage.get("cost"), details.get("upstream_inference_cost")) if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if paid:
                self.cost += max(paid)
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            cut = data["choices"][0].get("finish_reason") == "length"
            self.host.trace.append({"kind": "llm_fail", "model": model, "status": "cut" if cut else 0})
            return None, ("cut" if cut else 0)
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):

            status = Text.count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        if status == 403 and "limit" in str(error.get("body") or "").lower():
            self.no_credit = True
        self.host.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    @staticmethod
    def headLength(text: str) -> int:
        best = 0
        for head in HEADS:
            limit = min(len(head), len(text) - 1)
            if limit <= best or text[:HEAD_CHARS] != head[:HEAD_CHARS]:
                continue
            same = next((i for i in range(HEAD_CHARS, limit) if text[i] != head[i]), limit)
            best = max(best, same)
        return best if best >= HEAD_CHARS else 0

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, effort: str = "low") -> dict[str, Any] | None:
        answers = self.__dict__.setdefault("answers", {})
        asked = (prompt, tuple(models), max_tokens, effort)
        if asked in answers:
            self.host.trace.append({"kind": "reused", "chars": len(prompt)})
            return json.loads(answers[asked])
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max(max_tokens, 1600), effort=effort)
            got = ModelGate.parse((message or {}).get("content"))
            if got is not None:
                answers[asked] = json.dumps(got)
                return got
            if self.host.broke or self.no_credit or self.calls >= self.host.budgets.allCalls or not self.host.budgets.canCall():
                return None
        return None

class StoreMap:
    def __init__(self, host: Any, problem: dict[str, Any]):
        self.host = host
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.id_args: set[tuple[str, str]] = set()
        self.facts_from: dict[str, bool] = {}
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None

    @staticmethod
    def kinds(schema: dict[str, Any]) -> list[str]:
        kind = schema.get("type")
        return [kind] if isinstance(kind, str) else [k for k in kind or [] if isinstance(k, str)]

    @staticmethod
    def fitsKind(value: Any, kind: str) -> bool:
        plain = {"string": str, "boolean": bool, "array": list, "object": dict}
        if kind in plain:
            return isinstance(value, plain[kind])
        if kind in ("integer", "number"):
            return isinstance(value, int if kind == "integer" else (int, float)) and not isinstance(value, bool)
        if kind == "null":
            return value is None
        return True

    @staticmethod
    def shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
        if not isinstance(schema, dict):
            return value
        kinds = StoreMap.kinds(schema)
        if kinds and not any(StoreMap.fitsKind(value, k) for k in kinds):
            value = StoreMap.coerce(value, kinds)
            if not any(StoreMap.fitsKind(value, k) for k in kinds):
                faults.append(f"{where} should be {'/'.join(kinds)}")
                return value
        value = StoreMap.choice(value, schema, where, faults)
        if isinstance(value, str):
            value = StoreMap.checkText(value, schema, where, faults)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            StoreMap.checkNumber(value, schema, where, faults)
        elif isinstance(value, list):
            value = StoreMap.checkList(value, schema, where, faults)
        elif isinstance(value, dict):
            value = StoreMap.checkObject(value, schema, where, faults)
        return value

    @staticmethod
    def coerce(value: Any, kinds: list[str]) -> Any:
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                return int(value)
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                return int(number) if kind == "integer" or number.is_integer() else number
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            if kind == "array" and not isinstance(value, list) and value is not None:
                return [value]
        return value

    @staticmethod
    def choice(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
        options = schema.get("enum")
        if not isinstance(options, list) or value in options:
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            value = next((o for o in options if isinstance(o, str) and o.lower() == low), value)
        if value not in options:
            faults.append(f"{where}: allowed values are {options}")
        return value

    @staticmethod
    def checkText(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            faults.append(f"{where} is too short")
        return value

    @staticmethod
    def checkNumber(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> None:
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")

    @staticmethod
    def checkList(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            return [StoreMap.shape(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
        return value

    @staticmethod
    def checkObject(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            faults.extend(f"{where}.{name} is not an allowed argument" for name in value if name not in props)
        faults.extend(f"{where}.{name} is required" for name in schema.get("required") or [] if name not in value)
        return {k: StoreMap.shape(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}

    @staticmethod
    def textFields(props: dict[str, Any]) -> list[str]:
        return [k for k, v in props.items() if isinstance(v, dict) and "string" in StoreMap.kinds(v) and "enum" not in v]

    @staticmethod
    def arrayField(props: dict[str, Any]) -> str | None:
        return next((k for k, v in props.items() if isinstance(v, dict) and "array" in StoreMap.kinds(v)), None)

    def shapedArgs(self, tool: Any, args: dict[str, Any]) -> dict[str, Any] | None:
        faults: list[str] = []
        shaped = StoreMap.shape(args, self.schemas.get(tool), "args", faults)
        return None if faults else shaped

    def page_tools(self) -> set[str]:
        return {t for t in (self.offer_tool(), self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        tool = self.roles.get("open")
        return known or (tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None)

    def compare_facts(self) -> bool | None:
        tool = self.roles.get("compare")
        return self.facts_from.get(tool or "")

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        if seen is not None:
            return seen is False
        return self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))

    def readBrief(self) -> None:
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {render(t['function'].get('parameters') or {})}" for t in self.tools]
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + render(self.view.get("opening"), 1500)
        models = self.host.models
        brief = models.ask_json(prompt, [LIGHT], 4000) or {}
        if not models.no_credit and (not Text.items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = models.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = self.review_reading(brief)
        self.readRoles()

    def readRoles(self) -> None:
        brief = self.brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + render(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.host.models.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]

    def review_reading(self, brief: dict[str, Any]) -> dict[str, Any]:
        if self.host.models.no_credit or self.host.broke or not Text.items(brief.get("items")):
            return brief
        shown = {key: brief.get(key) for key in ("items", "lookups", "process", "pick", "pick_measure")}
        labels = sorted(self.labels())
        prompt = REVIEW + "\n\nREQUEST:\n" + self.request
        if labels and len(labels) <= 80:
            prompt += "\n\nSTORE LABELS LISTED BY THE TOOLS:\n" + ", ".join(labels)
        prompt += "\n\nFIRST READING:\n" + render(shown)
        got = self.host.models.ask_json(prompt, [LIGHT, HEAVY], 2200)
        if not isinstance(got, dict):
            return brief
        fixed = dict(brief)
        items = [e for e in Text.items(got.get("items")) if isinstance(e, dict) and str(e.get("name") or "").strip()]
        if items:
            fixed["items"] = items
        for key in ("lookups", "process"):
            if isinstance(got.get(key), list):
                fixed[key] = got[key]
        if got.get("pick") in ("cheapest", "most", "least", "any"):
            fixed["pick"] = got["pick"]
            fixed["pick_measure"] = str(got.get("pick_measure") or "") if fixed["pick"] in ("most", "least") else ""
        self.host.trace.append({"kind": "review", "before": Text.clip(shown, 200), "changes": [str(c)[:200] for c in Text.items(got.get("changes"))][:10]})
        return fixed

    def labels(self) -> set[str]:
        found: set[str] = set()
        stack: list[tuple[Any, int]] = [(schema, 0) for schema in self.schemas.values()]
        while stack:
            schema, depth = stack.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            found.update(option.strip().lower() for option in schema.get("enum") or [] if isinstance(option, str))
            stack.extend((value, depth + 1) for value in schema.values() if isinstance(value, dict))
        return found

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        texts = StoreMap.textFields(schema.get("properties") or {})
        required = schema.get("required") or []
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        return target if tool and target and all(k == target for k in required) else None

    def note_view(self, name: str, observation: Any) -> None:
        now = self.host.book.version
        for ident, record in self.host.shelf.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (now, (have if version == now else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        keys = self.tool_keys.get(name)
        if not keys or ident in self.host.shelf.gone:
            return False
        version, have = self.view_keys.get(ident, (-1, set()))
        return version == self.host.book.version and keys <= have

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = SHAPE + "\n\nTOOL: " + name + " | parameters: " + render(self.schemas.get(name) or {}) + "\n\nRESULT (first rows only):\n" + render(sample, 120)[:4000]
        got = self.host.models.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        props = (self.schemas.get(name) or {}).get("properties") or {}
        places = {"count": observation, "next": observation, "next_arg": props, "price": observation[rows][0], "stock": observation[rows][0], "query": props, "cap": props}
        shape = {"rows": rows, **{field: StoreMap.namedKey(got.get(field), where) for field, where in places.items()}}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.host.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    @staticmethod
    def namedKey(key: Any, where: dict[str, Any]) -> str | None:
        return key if isinstance(key, str) and key in where else None

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not (shape and shape["count"] and shape["next"] and shape["next_arg"] and isinstance(observation, dict)):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        good = isinstance(rows, list) and bool(rows) and isinstance(count, int) and not isinstance(count, bool)
        good = good and (value is None or (isinstance(value, (str, int)) and not isinstance(value, bool)))
        return (count, len(rows), shape["next_arg"], value) if good else None

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get("ask") or "") or {}
        props = schema.get("properties") or {}
        texts = StoreMap.textFields(props)
        required = schema.get("required") or []
        target = next((k for k in required if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in required:
            if key in args:
                continue
            options = (props.get(key) or {}).get("enum")
            if not (isinstance(options, list) and options):
                return None
            args[key] = options[0]
        return self.shapedArgs(self.roles.get("ask"), args)

    def offer_tool(self) -> str | None:
        for role in ("offer", "open"):
            if self.roles.get(role):
                return self.roles[role]
        return None

    def isEnding(self, name: str, args: dict[str, Any], role: str) -> bool:
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(role + "_args")
        return all(args.get(k) == v for k, v in (fixed if isinstance(fixed, dict) else {}).items())

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self.isEnding(name, args, "buy")

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self.isEnding(name, args, "refuse")

    def offered(self) -> list[dict[str, Any]]:
        host = self.host
        left = host.left()
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        limits = ((host.book.tail(), ()), (host.book.tail() + 3, ("cart", "cart_view")), (len(host.book.items) + 5 if host.buys == 0 else -1, ("cart", "cart_view", "offer")))
        narrow = next((extra for limit, extra in limits if left <= limit), None)
        tools = [t for t in self.tools if t["function"]["name"] != self.roles.get("ask")]
        if narrow is None or not ends:
            return tools
        keep = ends | {self.roles.get(r) for r in narrow}
        return [t for t in tools if t["function"]["name"] in keep] or tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        out = dict(args)
        for key in StoreMap.textFields(props):
            value = out.get(key)
            if value is None and key not in required:
                continue
            text = str(value or "")
            if not text or any(ident in text for ident in self.host.shelf.rows) or self.brief.get("quiet_decline") is not False:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: props[key].get("maxLength", 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.host.shelf.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in StoreMap.kinds(v) or "array" in StoreMap.kinds(v)) and "enum" not in v]
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""
        key = free[0]
        if key not in args:
            return dict(args), ""
        listed = isinstance(args[key], list)
        fixed = [self.resolveOne(v) for v in (args[key] if listed else [args[key]])]
        problem = next((p for _, p in fixed if p), "")
        if problem:
            return args, problem
        return {**args, key: [v for v, _ in fixed] if listed else fixed[0][0]}, ""

    def resolveOne(self, value: Any) -> tuple[Any, str]:
        shelf = self.host.shelf
        if isinstance(value, str) and value.strip() in shelf.rows:
            return value.strip(), ""
        if not isinstance(value, str) or Shelf.idLike(value, True) or len(value.strip()) < 8:
            return value, ""
        low = value.strip().lower()
        hits = [i for i, row in shelf.rows.items() if any(k != shelf.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
        if len(hits) == 1:
            return hits[0], ""
        if hits:
            return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
        return value, f"{value!r} is not a listing id; pass the id from the result rows (their {shelf.id_key} field)"

    def cart_sets(self, name: str) -> bool:
        specs = [s for s in (self.schemas.get(name, {}).get("properties") or {}).values() if isinstance(s, dict)]
        if any(isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in specs):
            return False
        return any("integer" in StoreMap.kinds(s) and s.get("minimum") == 0 for s in specs)

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        target = StoreMap.arrayField(schema.get("properties") or {})
        return target if target and all(k == target for k in schema.get("required") or []) else None

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        top = spec["maxItems"] if isinstance(spec.get("maxItems"), int) and spec["maxItems"] >= 1 else max(4, self.compare_ok)
        if not self.compare_bad:
            return top
        return max(1, min(top, max(self.compare_ok, (self.compare_bad + 1) // 2)))

class Shelf:
    def __init__(self, host: Any):
        self.host = host
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.id_key: str | None = None
        self.opened: set[str] = set()
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.offer_seen: dict[str, int] = {}
        self.refused_at = -1
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.cart_said = ""
        self.built: list[str] = []
        self.dropped: set[str] = set()

    @staticmethod
    def idLike(value: Any, named: bool = False) -> bool:
        text = value.strip() if isinstance(value, str) else ""
        if not 3 <= len(text) <= 64 or " " in text:
            return False
        return named or re.search(r"\d", text) is not None

    @staticmethod
    def learnId(rows: list[dict[str, Any]]) -> str | None:
        found = []
        for key in rows[0]:
            values = [row.get(key) for row in rows]
            if not all(Shelf.idLike(v, "id" in str(key).lower()) for v in values):
                continue
            values = [v.strip() for v in values]
            if len(rows) > 1 and len(set(values)) < len(values):
                continue
            found.append((("id" in str(key).lower(), sum(len(v) for v in values) / len(values)), key))
        return max(found, key=lambda f: f[0])[1] if found else None

    @staticmethod
    def inRows(node: Any, ident: str, depth: int = 0) -> bool:
        if depth > 5 or not isinstance(node, (list, dict)):
            return False
        if isinstance(node, list) and any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node):
            return True
        children = node if isinstance(node, list) else list(node.values())
        return any(Shelf.inRows(v, ident, depth + 1) for v in children)

    @staticmethod
    def factBlock(node: Any) -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        blocks = [scalars for value in node.values() if isinstance(value, dict) and len(value) >= 2
                  for scalars in [{k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}] if len(scalars) >= 2]
        return max(blocks, key=len) if blocks else None

    def cart_mark(self) -> str:
        return render(sorted(self.cart_lines.items()))

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if isinstance(node, dict) and len(node) >= 3:
            if self.id_key and Shelf.idLike(node.get(self.id_key), True):
                return node[self.id_key].strip()
            if in_list or depth == 0 or any(isinstance(v, dict) for v in node.values()):
                found = Shelf.learnId([node])
                if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
                    return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or 2 * len(rows) < len(node):
            return False
        known = self.id_key and all(Shelf.idLike(r.get(self.id_key), True) for r in rows)
        if known:
            return True
        found = Shelf.learnId(rows)
        self.id_key = found or self.id_key
        return bool(found)

    def absorb(self, observation: Any) -> None:
        self.absorbNode(observation, 0, False)

    def absorbNode(self, node: Any, depth: int, in_list: bool) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            rows = self.learn_rows(node)
            for value in node:
                self.absorbNode(value, depth + 1, rows)
            return
        if not isinstance(node, dict):
            return
        ident = self.ident(node, in_list, depth)
        if not ident:
            for value in node.values():
                self.absorbNode(value, depth + 1, False)
            return
        if ident not in self.rows:
            self.seen_order.append(ident)
            self.rows[ident] = {}
        old = self.rows[ident]
        if any(old.get(k) is True and v is False for k, v in node.items()):
            self.unavailable.add(ident)
        elif any(old.get(k) is False and v is True for k, v in node.items()):
            self.unavailable.discard(ident)
        old.update(node)
        if Shelf.factBlock(node):
            self.full[ident] = dict(old)
        elif ident in self.full:
            self.full[ident].update(node)

    def factIds(self, observation: Any) -> list[str]:
        ids: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and Shelf.factBlock(node) and ident not in ids:
                    ids.append(ident)
                    continue
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                continue
            stack.extend((child, depth + 1) for child in reversed(children))
        return ids

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.host.store.labels()
        scored = [((-(sum(1 for v in values if v.strip().lower() in known) / len(values)), len(values), -present[key], -(sum(len(v) for v in values) / len(values))), key)
                  for key, values in counts.items() if present[key] >= 0.8 * total and len(values) <= 40]
        return min(scored, key=lambda s: s[0])[1] if scored else None

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if self.wantedLabel(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=len, reverse=True)[: max(1, len(self.host.book.items))]

    def wantedLabel(self, label: str) -> bool:
        name = Text.words(label)
        long = {w for w in Text.stem(label) if len(w) >= 4}
        for item in self.host.book.items:
            if name in (Text.words(item.shelf or ""), Text.words(item.name)) or long & {w for w in Text.stem(item.shelf or "") | Text.stem(item.name) if len(w) >= 4}:
                return True
        return False

    def price_of(self, ident: str) -> float | None:
        key = self.host.store.price_key
        value = (self.rows.get(ident) or {}).get(key) if key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        elif isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        key = self.host.store.stock_key
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return value is not False and not (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            lists = [w for v in observation.values() for w in ([v] if isinstance(v, list) else [x for x in v.values() if isinstance(x, list)] if isinstance(v, dict) else [])]
        elif isinstance(observation, list):
            lists = [observation]
        else:
            return []
        return [(i, r) for rows in lists for r in rows if isinstance(r, dict) for i in [self.ident(r, True)] if i]

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))

    def openOne(self, tool: str, args: dict[str, Any], mode: str) -> tuple[dict[str, Any], Any] | None:
        host = self.host
        reply = host.act(tool, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        host.book.observe(tool, seen)
        if mode != "plain":
            host.store.note_view(tool, seen)
            shown = [Shelf.factBlock(seen)] if mode == "single" and isinstance(seen, dict) and self.ident(seen) else [Shelf.factBlock(r) for _, r in self.records_in(seen)] if mode == "many" else []
            if shown:
                host.store.facts_from[tool] = host.store.facts_from.get(tool, False) or any(bool(block) for block in shown)
        return reply, seen

    def openMore(self, ids: list[str], keep: int) -> list[str]:
        host = self.host
        store = host.store
        tool = store.facts_tool()
        target = store.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if host.left() <= keep or host.finished or store.facts_from.get(tool) is False:
                break
            args = store.shapedArgs(tool, {target: ident})
            if args is None:
                break
            opened = self.openOne(tool, args, "single")
            if opened is None:
                break
            texts.append(f"{ident}: " + host.show(opened[1])[:1500])
        if texts:
            host.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def byPrice(self, pool: list[str]) -> list[str]:
        if self.host.store.brief.get("pick") == "cheapest":
            pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        return pool

    def kind_of(self, ident: str, key: str | None) -> str:
        return str((self.rows.get(ident) or {}).get(key) or "").strip().lower() if key else ""

    def openWide(self) -> None:
        host = self.host
        store = host.store
        tool = store.roles.get("compare")
        schema = store.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = StoreMap.arrayField(props)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or store.compare_bare():
            self.openLanes()
            return
        if host.left() <= 12:
            return
        cap = store.compare_room()
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks = [i for ids in ranked for i in [i for i in ids if i not in self.full][:share]][:cap]
        if len(picks) < 2:
            return
        args = store.shapedArgs(tool, {target: picks})
        if args is None:
            return
        before = len(self.full)
        opened = self.openOne(tool, args, "many")
        if opened is None:
            return
        self.compareSeen(picks, opened[1])
        rows = self.records_in(opened[1])
        host.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
        if rows and store.facts_from.get(tool) is False:
            self.openLanes()
            return
        room = store.compare_room()
        if len(self.full) == before and 2 <= room < len(picks) and host.left() > 12:
            args = store.shapedArgs(tool, {target: picks[:room]})
            again = self.openOne(tool, args, "plain") if args is not None else None
            if again is not None:
                self.compareSeen(picks[:room], again[1])
        if any(not any(i in self.full for i in ids) for ids in self.item_groups()):
            self.openLanes()

    def openLanes(self) -> None:
        host = self.host
        tool = host.store.facts_tool()
        target = host.store.single_target(tool)
        if not tool or not target:
            return
        want = max(2, min(4, host.max_steps // 24))
        lanes: list[tuple[int, list[str]]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = self.byPrice([i for i in ids if i not in self.full and self.buyable(i)])
            lanes.append((opened, pool[: max(0, want - opened)]))
        lanes.sort(key=lambda lane: lane[0])
        queue = [lane[k] for k in range(want) for _, lane in lanes if k < len(lane)]
        self.openMore(queue[: want * max(1, len(lanes))], 22)

    def cartFresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def cart_text(self) -> str:
        if self.cartFresh():
            return render(self.cart_view, 300)[:3000]
        return "lines by Act's record of the cart calls (listing: units): " + render(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if not self.cartFresh():
            return list(self.cart_lines)
        text = render(self.cart_view)
        return [i for i in self.rows if i in text]

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if not record:
            return "(no record seen)"
        return render(record, 400)[:3000]

    def recordLines(self, order: Any, skip: Any, limit: int, width: int, cut: int) -> list[str]:
        lines: list[str] = []
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            lines.append(render(self.full.get(ident) or self.rows.get(ident), width)[:cut])
            if len(lines) >= limit:
                break
        return lines

    def others(self, skip: list[str], limit: int = 30) -> str:
        order = list(reversed(self.seen_order))
        if self.host.store.brief.get("pick") == "cheapest":
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        return "\n".join(self.recordLines(order, skip, limit, 160, 900))

    def storeLabels(self) -> list[str]:
        catKey = self.category_key()
        seen: list[str] = []
        for ident in self.seen_order if catKey else []:
            label = (self.rows.get(ident) or {}).get(catKey)
            if isinstance(label, str) and label.strip() and label not in seen:
                seen.append(label)
            if len(seen) >= 40:
                break
        return seen

    def reread_offers(self) -> str:
        host = self.host
        tool = host.store.offer_tool()
        target = host.store.single_target(tool)
        if not tool or not target:
            return ""
        since = max(host.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or host.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            line = self.rereadOne(tool, target, ident)
            if line is None:
                break
            if line:
                lines.append(line)
        host.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            host.book.version += 1
        return "\n".join(lines)

    def rereadOne(self, tool: str, target: str, ident: str) -> str | None:
        host = self.host
        args = host.store.shapedArgs(tool, {target: ident})
        if args is None:
            return None
        before = dict(self.rows.get(ident) or {})
        reply = host.act(tool, args)
        if reply is None or host.finished:
            return None
        observation = reply.get("observation")
        self.offer_seen[ident] = host.step
        host.book.observe(tool, observation)
        after = self.rows.get(ident) or {}
        moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
        return f"{ident} changed ({', '.join(moved[:4])}): {render(observation, 300)[:900]}" if moved else ""

    def readNew(self, ids: list[str]) -> str:
        host = self.host
        tool = host.store.roles.get("offer")
        target = host.store.single_target(tool)
        if not ids or not tool or not target or tool == host.store.facts_tool():
            return ""
        lines: list[str] = []
        read: list[str] = []
        for ident in ids:
            if host.finished or host.left() <= len(ids) + 2:
                break
            if host.store.shown_already(tool, ident):
                continue
            line = self.rereadOne(tool, target, ident)
            if line is None:
                break
            read.append(ident)
            if line:
                lines.append(line)
        if read:
            host.trace.append({"kind": "read_new", "ids": read, "changed": len(lines)})
        if lines:
            host.book.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        return any(ident in pool for pool in (self.unavailable, self.gone))

    def swap_hint(self) -> str:
        host = self.host
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        self.gone.update((i, host.step) for i in lost)
        host.decline.leads = [i for i in host.decline.leads if i not in lost]
        host.book.version += 1
        picks = self.replacements(lost)
        text = f"[Act] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if not picks:
            return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."
        if host.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if host.store.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if not ids or any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) > 1:
            return self.reread_offers()
        self.unavailable.add(ids[0])
        return ""

    def replacements(self, lost: list[str]) -> list[str]:
        host = self.host
        if host.models.cost > host.budgets.spendSoft:
            return []
        records = self.recordLines(reversed(self.seen_order), lost, 25, 200, 1200)
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + host.store.request + "\n\nREQUIREMENTS IN FORCE:\n" + host.book.render()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = host.models.ask_json(prompt, [LIGHT, HEAVY], 2500)
        fits: list[tuple[float, str]] = []
        for entry in Text.items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if host.store.brief.get("pick") == "cheapest" else [i for _, i in fits]
        host.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def item_of(self, ident: str) -> int:
        cat_key = self.category_key()
        label = Text.words(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key else ""
        if not label:
            return 0
        talk = Interviewer(self.host)
        return next((n for n in range(1, len(self.host.book.items) + 1) if Text.words(talk.itemLabel(n)) == label), 0)

    def openRows(self, rows: list[str]) -> bool:
        host = self.host
        store = host.store
        many, joiner = store.compare_target(), store.roles.get("compare")
        tool = store.facts_tool()
        target = store.single_target(tool)
        before = len(self.full)
        pending = [i for i in rows if i not in self.full]
        rounds = 0
        while pending and not host.finished and rounds < 4:
            rounds += 1
            joined = bool(many and joiner and not store.compare_bare())
            chunk = pending[: store.compare_room()] if joined else pending[:2]
            if joined:
                plan = [(joiner, {many: chunk})]
            elif tool and target:
                plan = [(tool, {target: i}) for i in chunk]
            else:
                break
            if host.left() < len(plan) + len(host.book.items) + 2:
                break
            grew = len(self.full)
            if not self.openPlan(plan, joined, chunk):
                break
            retry = joined and len(self.full) == grew and store.compare_room() < len(chunk)
            if not retry:
                pending = pending[len(chunk):]
            if len(self.full) == grew and not retry:
                break
        host.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def openPlan(self, plan: list[tuple[str, dict[str, Any]]], joined: bool, chunk: list[str]) -> bool:
        host = self.host
        for name, args in plan:
            args = host.store.shapedArgs(name, args)
            if args is None:
                return False
            opened = self.openOne(name, args, "plain")
            if opened is None or host.finished:
                return False
            if joined:
                self.compareSeen(chunk, opened[1])
        return True

    def compareSeen(self, chunk: list[str], observation: Any) -> None:
        store = self.host.store
        if len(chunk) > 1 and not any(str(i) in render(observation) for i in chunk):
            store.compare_bad = min(store.compare_bad or 999, len(chunk))
            self.opened.difference_update(i for i in chunk if i not in self.full)
        else:
            store.compare_ok = max(store.compare_ok, len(chunk))

    def cartLineShape(self, props: dict[str, Any]) -> tuple[str, str | None, str | None] | None:
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in StoreMap.kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in StoreMap.kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in StoreMap.kinds(s)), None)
                return key, id_key, count_key
        return None

    def planCart(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        store = self.host.store
        name = store.roles.get("cart") or ""
        schema = store.schemas.get(name) or {}
        props = schema.get("properties") or {}
        listed = self.cartLineShape(props)
        if listed:
            key, id_key, count_key = listed
            if not id_key:
                return None
            return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = StoreMap.textFields(props)
        target = (next((k for t, k in store.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in StoreMap.kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        sets = store.cart_sets(name)
        calls: list[dict[str, Any]] = []
        for ident in drop:
            call = self.dropCall(ident, target, count, mode, props, sets)
            if call is None:
                return None
            calls.append(call)
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not sets:
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def dropCall(self, ident: str, target: str, count: str | None, mode: str | None, props: dict[str, Any], sets: bool) -> dict[str, Any] | None:
        have = self.cart_lines.get(ident, 1)
        if mode:
            return {target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})}
        if count and sets:
            return {target: ident, count: 0}
        if count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
            return {target: ident, count: -have}
        return None

    def setLines(self, calls: list[dict[str, Any]]) -> list[str] | None:
        host = self.host
        cart = host.store.roles.get("cart")
        done: list[str] = []
        for args in calls:
            args = host.store.shapedArgs(cart, args)
            if args is None:
                return None
            reply = host.act(cart, args)
            if reply is None or host.finished:
                return None
            self.applyCart(cart, args)
            host.book.observe(cart, reply.get("observation"))
            done.append(render(args))
        return done

    def applyCart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.host.step
        spec = self.host.store.schemas.get(name, {}).get("properties") or {}
        removal = any(isinstance((spec.get(key) or {}).get("enum"), list) and len((spec.get(key) or {}).get("enum")) >= 2 and value in (spec.get(key) or {}).get("enum")[1:] for key, value in args.items())
        loose_count = next((Text.count(v) for v in args.values() if Text.count(v) is not None and not isinstance(v, str)), None)
        lines, whole = self.cartLines(args, spec, loose_count)
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.host.store.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            else:
                have = (count if count is not None else 1) + (0 if sets else have)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def cartLines(self, args: dict[str, Any], spec: dict[str, Any], loose_count: int | None) -> tuple[list[tuple[str, int | None]], bool]:
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
            if not isinstance(value, list):
                continue
            shape = (spec.get(key) or {}).get("items") or {}
            if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                whole = True
            for line in value:
                if isinstance(line, dict):
                    whole = True
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    count = next((Text.count(v) for v in line.values() if Text.count(v) is not None and not isinstance(v, str)), 1)
                    if ident:
                        lines.append((ident, count))
                elif isinstance(line, str) and line in self.rows:
                    lines.append((line, loose_count))
        return lines, whole

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        host = self.host
        store = host.store
        opener, joiner, target = store.roles.get("open"), store.roles.get("compare"), store.single_target(store.roles.get("open"))
        many = store.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner or store.compare_bare():
            return calls
        picks = self.openerCalls(calls, opener, target)[: store.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        joined = {i for i, _ in picks}
        out = [{"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": render({many: [i for _, i in picks]})}} if index == first
               else {"id": call["id"], "joined": True} if index in joined else call for index, call in enumerate(calls)]
        host.trace.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def openerCalls(self, calls: list[dict[str, Any]], opener: str, target: str) -> list[tuple[int, str]]:
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = ModelGate.parse(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        return picks

    def partial_cart(self) -> bool:
        ids = [i for i in self.cart_ids() if not self.sold_out(i)]
        cat_key = self.category_key()
        if len(self.host.book.items) < 2 or len(ids) >= len(self.host.book.items) or not cat_key:
            return False
        kinds = {Text.words(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in ids} - {""}
        return bool(kinds) and len(kinds) < len(self.host.book.items)

class Ledger:
    def __init__(self, host: Any):
        self.host = host
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
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.queued: list[str] = []

    def fromBrief(self) -> None:
        host = self.host
        brief = host.store.brief
        for entry in Text.items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = Text.count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in Text.items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        host.budgets.resize(host.max_steps, len(self.items))
        host.lookups.fromBrief(brief)
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        limit = Text.count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif host.store.roles.get("ask"):
            self.msg_limit = 3
        steps = Text.count(brief.get("steps"))
        if steps is not None and 10 <= steps < host.max_steps:
            host.max_steps = steps
        for entry in Text.items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, Text.count(entry.get("messages")) or 0)})
        if host.store.roles.get("ask"):
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def msgs_free(self) -> int:
        held = max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - (self.msg_used + held)

    def gate_due(self, stage: str) -> bool:
        mark = (stage, self.host.shelf.cart_mark() + f"#{self.version}")
        return mark not in self.gate_ok and any(p["when"] == stage for p in self.process)

    def tail(self) -> int:
        if any(p["when"] == "order" and p["messages"] > 0 for p in self.process) and self.msg_used < self.msg_limit and self.gate_due("order"):
            return 2
        return 1

    def itemIndex(self, number: Any) -> int:
        return min(max(1, Text.count(number) or 1), len(self.items))

    def itemAt(self, number: Any) -> Item:
        return self.items[self.itemIndex(number) - 1]

    def itemLine(self, number: int, item: Item, many: bool) -> str:
        refs = self.host.lookups.ref_values
        needs = [m + (f" (the page it points at gives: {refs[m]})" if m in refs else "") for m in item.musts + item.told if m not in item.dropped]
        head = (f"item {number}: {item.name}" if many else f"item: {item.name}") + (f" (store label: {item.shelf})" if item.shelf else "") + f", quantity {item.qty}"
        text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
        return text + ("; shopper does not mind: " + ", ".join(item.free) if item.free else "")

    def pickLine(self, brief: dict[str, Any]) -> str:
        pick = brief.get("pick")
        if pick == "cheapest":
            return "pick: the cheapest listing that fits (lowest price)"
        if pick in ("most", "least"):
            return f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace("  ", " ")
        return "pick: any listing that fits everything"

    def factsLine(self) -> str:
        store = self.host.store
        facts = store.facts_tool()
        if not facts or not (store.facts_from.get(facts) is True or store.roles.get("offer")):
            return ""
        bare = sorted(t for t in store.page_tools() | {store.roles.get("offer") or ""} if t and store.facts_from.get(t) is False)
        pages = sorted(t for t in store.page_tools() if t != facts)
        return (f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))

    def sheet(self) -> list[str]:
        brief = self.host.store.brief
        many = len(self.items) > 1
        parts = [self.itemLine(number, item, many) for number, item in enumerate(self.items, 1)]
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        parts.append(self.pickLine(brief))
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        parts += [f"wish (tie-breaker only): {wish}" for wish in Text.items(brief.get("wishes"))]
        parts += [f"rule from the request: {rule}" for rule in Text.items(brief.get("rules"))]
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        stages = {"choose": "before choosing", "order": "at the order (Act does it when you check out)", "end": "at an ending without a purchase (Act does it then)"}
        parts += [f"step the request asks for, {stages[entry['when']]}: {entry['step']}" for entry in self.process]
        facts = self.factsLine()
        return parts + [facts] if facts else parts

    def render(self) -> str:
        return "\n".join("- " + p for p in self.sheet())

    def observe(self, name: str, observation: Any) -> str:
        self.host.shelf.absorb(observation)
        meaning = self.hearNote(observation)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def logStep(self, what: str) -> None:
        host = self.host
        self.done_log.append(f"step {host.step}: " + what + f" (cart then: {host.shelf.cart_mark()})")
        self.done_steps.append(host.step)

    def sendOne(self, tool: str, text: str) -> tuple[str, Any] | None:
        host = self.host
        args = host.store.message_args(text)
        if not args:
            return None
        reply = host.act(tool, args)
        if reply is None:
            return None
        self.msg_used += 1
        answer = reply.get("observation")
        words = Text.prose(answer)
        self.logStep(f"message to the shopper {text!r} -> reply {words[:300]!r}")
        host.trace.append({"kind": "ask", "q": text, "a": words[:300]})
        return words, answer

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        host = self.host
        tool = host.store.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if host.finished or host.left() <= floor or self.msg_used >= self.msg_limit:
                break
            heard = self.sendOne(tool, text)
            if heard is None:
                break
            sent.append((number, text, heard[0]))
            if isinstance(heard[1], dict) and heard[1].get("error"):
                break
        return sent

    def heardNeeds(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in Text.items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = self.itemIndex(entry.get("item"))
            need = str(entry["need"]).strip()
            item = self.items[number - 1]
            if need not in item.told:
                item.told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.host.lookups.refs:
                self.host.lookups.refs.append((number, need))

    def heardDrops(self, heard: dict[str, Any], told: list[str]) -> None:
        for entry in Text.items(heard.get("drop")):
            if not isinstance(entry, dict) or not str(entry.get("property") or "").strip():
                continue
            item = self.itemAt(entry.get("item"))
            gone = Text.words(str(entry["property"]).split(":")[0])
            hit = [m for m in item.musts + item.told if m not in item.dropped and Text.words(m.split(":")[0]) == gone]
            item.dropped.extend(hit)
            told.extend(f"dropped {m}" for m in hit)

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        self.heardNeeds(heard, told)
        self.heardDrops(heard, told)
        for entry in Text.items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                self.itemAt(entry.get("item")).free.append(str(entry["property"]))
        budget = heard.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        items = "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        pairs = "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.host.models.ask_json(HEAR + "\n\nITEMS:\n" + items + "\n\nQUESTIONS AND ANSWERS:\n" + pairs, list(models), 1500) or {}

    def looseLines(self, observation: Any) -> list[str]:
        shelf = self.host.shelf
        if not isinstance(observation, dict) or shelf.ident(observation):
            return []
        lines: list[str] = []
        for value in observation.values():
            if isinstance(value, str):
                lines += [value] if len(value.split()) >= 5 else []
            elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                lines += [v for v in value if len(v.split()) >= 5]
            elif isinstance(value, dict) and not shelf.ident(value, False, 1):
                lines += [inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return [line for line in lines if line not in self.heard]

    def noteEntries(self, value: Any) -> list[dict[str, Any]]:
        return [e for e in Text.items(value) if isinstance(e, dict) and str(e.get("need") or "").strip()]

    def noteBudget(self, got: dict[str, Any]) -> list[str]:
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            return [f"budget now {budget}"]
        return []

    def noteDrops(self, got: dict[str, Any]) -> list[str]:
        changed = []
        for entry in self.noteEntries(got.get("drop")):
            item = self.itemAt(entry.get("item"))
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target] or [m for m in live if Text.words(m.split(":")[0]) == Text.words(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        return changed

    def noteAdds(self, got: dict[str, Any]) -> list[str]:
        changed = []
        for entry in self.noteEntries(got.get("add")):
            item = self.itemAt(entry.get("item"))
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f"added {need}")
        return changed

    def hearNote(self, observation: Any) -> str:
        lines = self.looseLines(observation)
        if not lines:
            return ""
        shelf = self.host.shelf
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.render() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.host.models.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in Text.items(got.get("gone")) if isinstance(g, (str, int))]
        shelf.unavailable.update(g for g in gone if g in shelf.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = self.noteBudget(got) + self.noteDrops(got) + self.noteAdds(got)
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.host.recent = []
            self.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.musts + item.told if need not in item.dropped]

    def evidenceLine(self, ident: str, needs: list[tuple[int, str]], many: bool) -> str:
        block = Shelf.factBlock(self.host.shelf.full.get(ident) or {}) or {}
        parts = []
        for number, need in needs:
            prop = need.split(":", 1)[0].strip()
            hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if Text.names(prop, k)]
            if hits:
                parts.append((f"(item {number}) " if many else "") + f"{prop} -> " + ", ".join(hits[:3]))
        return f"{ident}: " + "; ".join(parts) if parts else ""

    def evidence(self, observation: Any) -> str:
        ids = self.host.shelf.factIds(observation)
        needs = self.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.items) > 1
        lines = [line for ident in ids[:12] for line in [self.evidenceLine(ident, needs[:8], many)] if line]
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

class Lookups:
    def __init__(self, host: Any):
        self.host = host
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.ref_holds = 0
        self.shop_tried: set[str] = set()

    def fromBrief(self, brief: dict[str, Any]) -> None:
        items = self.host.book.items
        for lookup in Text.items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in items):
                key = Text.words(text.split(":")[0])
                text = next((m for item in items for m in item.musts if Text.words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()

    def unmixCall(self, item: Item, pointers: list[str], rest: list[str]) -> list[str]:
        host = self.host
        prompt = (UNMIX + "\n\nREQUEST:\n" + host.store.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                  + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
        got = host.models.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
        said = [str(x).strip() for x in Text.items(got.get("describe_other"))]
        return [m for m in rest if m in said]

    def unmix(self) -> None:
        host = self.host
        for number, item in enumerate(host.book.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            moved = self.unmixCall(item, pointers, rest)
            if not moved:
                continue
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts if m not in moved]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            host.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def trailRecords(self, pending: list[str]) -> list[str]:
        order = list(reversed(self.host.shelf.seen_order))
        place = {ident: k for k, ident in enumerate(order)}
        bags: dict[str, frozenset[str]] = {}
        for ident in order:
            bags[ident] = Text.bag(render(self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident) or {}))
        counts: dict[str, int] = {}
        for bag in bags.values():
            for word in bag:
                counts[word] = counts.get(word, 0) + 1
        rare = max(2, len(bags) // 5)
        wanted = Text.bag(" ".join(pending))

        def score(ident: str) -> int:
            return sum(1 for w in wanted & bags[ident] if counts.get(w, 0) <= rare)

        ranked = sorted(order, key=lambda i: (-score(i), place[i]))
        records: list[str] = []
        for ident in ranked[:60]:
            if ident in self.host.shelf.opened:
                record = self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident)
                records.append(f"OPENED {ident}: " + render(record, 300)[:1400])
            else:
                records.append(f"ROW {ident}: " + render(self.host.shelf.rows.get(ident), 120)[:600])
        return records

    def shopRows(self, pending: list[str], tool: str | None, target: str | None) -> list[str]:
        host = self.host
        shelf = host.shelf
        catKey = shelf.category_key()
        if not pending or not tool or not target or not catKey or host.finished or host.budgets.closing() or len(self.shop_tried) >= 4 or host.left() < host.book.tail() + 6:
            return []
        bags = {ident: Text.bag(render(shelf.rows.get(ident) or {})) for ident in shelf.seen_order}
        counts: dict[str, int] = {}
        for bag in bags.values():
            for word in bag:
                counts[word] = counts.get(word, 0) + 1
        rare = max(2, len(bags) // 5)
        wanted: list[str] = []
        for need in pending:
            said = Text.bag(need, True)
            within = frozenset(said)
            ranked: list[tuple[int, int, str]] = []
            for place, ident in enumerate(shelf.seen_order):
                row = shelf.rows.get(ident) or {}
                label = row.get(catKey)
                kinds = {w for w in Text.bag(label) if len(w) >= 3} if isinstance(label, str) else set()
                if not kinds or not kinds <= within:
                    continue
                for key, value in row.items():
                    if key in (shelf.id_key, catKey) or not isinstance(value, str):
                        continue
                    name = Text.bag(value, True)
                    if 3 <= len(value.strip()) <= 40 and 1 <= len(name) <= 4 and max(len(w) for w in name) >= 4 and any(said[k:k + len(name)] == name for k in range(len(said) - len(name) + 1)):
                        ranked.append((-sum(1 for w in within & bags[ident] if counts.get(w, 0) <= rare), place, ident))
                        break
            ranked.sort()
            wanted += [ident for _, _, ident in ranked[:2] if ident not in shelf.opened and ident not in self.shop_tried and ident not in wanted]
        opened: list[str] = []
        for ident in wanted[:3]:
            if host.finished or len(self.shop_tried) >= 4 or host.left() < host.book.tail() + 6:
                break
            self.shop_tried.add(ident)
            faults: list[str] = []
            args = StoreMap.shape({target: ident}, host.store.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = host.act(tool, args)
            if reply is None:
                break
            host.book.observe(tool, reply.get("observation"))
            opened.append(f"{ident}: " + host.show(reply.get("observation"))[:1500])
        if opened:
            host.trace.append({"kind": "shop_rows", "pending": [p[:160] for p in pending[:3]], "opened": wanted[:len(opened)]})
        return opened

    def trail(self, rounds: int = 6) -> tuple[str, bool]:
        live = {need for _, need in self.host.book.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.host.store.roles.get("offer") or self.host.store.roles.get("open")
        target = self.host.store.single_target(tool)
        finder = self.host.store.roles.get("search")
        seen: list[str] = []
        lost: list[str] = []
        turns = 0
        searched = 0
        seen += self.shopRows(pending, tool, target)
        while pending and turns < rounds and self.trails < 10 and self.host.models.cost < self.host.budgets.spendHard * 1.1 and not self.host.finished:
            turns += 1
            self.trails += 1
            records = self.trailRecords(pending)
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (most relevant first):\n" + "\n".join(records)
            got = self.host.models.ask_json(prompt, [LIGHT, HEAVY], 3000) or {}
            self.host.trace.append({"kind": "trail", "pending": pending, "got": Text.clip(got, 150)})
            reach: list[str] = []
            queries: list[str] = []
            lost = []
            chains = [c for c in Text.items(got.get("chains")) if isinstance(c, dict)]
            if not chains and isinstance(got, dict) and any(k in got for k in ("links", "id", "candidates", "search")):
                chains = [{"need": got.get("need") or (pending[0] if len(pending) == 1 else ""), "links": got.get("links") or [got], "done": got.get("done"), "value": got.get("value")}]
            for chain in chains:
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in Text.items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.host.shelf.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.host.book.version += 1
                    continue
                closed = [i for i in ids if i in self.host.shelf.rows and i not in self.host.shelf.opened]
                loose = next((l for l, i in zip(links, ids) if not i or i not in self.host.shelf.opened), None) or (links[-1] if links else {})
                picks = [str(c).strip() for c in Text.items(loose.get("candidates")) if str(c).strip() in self.host.shelf.rows and str(c).strip() not in self.host.shelf.opened]
                for ident in closed + picks:
                    if ident not in reach:
                        reach.append(ident)
                if not closed and not picks:
                    lost.append(f"{need} -> find " + (str(loose.get("about") or "") or "the listing it points at"))
                    words = str(loose.get("search") or "").strip()
                    if words and words not in queries:
                        queries.append(words)
            pending = [p for p in pending if p not in self.ref_values]
            if not pending or self.host.finished:
                break
            acted = False
            if reach and tool and target and self.host.left() >= 5:
                for ident in reach[: max(0, min(3, 10 - len(seen), self.host.left() - 4))]:
                    faults: list[str] = []
                    args = StoreMap.shape({target: ident}, self.host.store.schemas.get(tool), "args", faults)
                    if faults:
                        break
                    reply = self.host.act(tool, args)
                    if reply is None or self.host.finished:
                        break
                    acted = True
                    self.host.book.observe(tool, reply.get("observation"))
                    seen.append(f"{ident}: " + self.host.show(reply.get("observation"))[:1500])
            elif queries and finder and searched < 3 and self.host.left() >= 8:
                call = self.host.coverage.search_args(finder, queries[0])
                if call is not None and finder + render(call) not in self.host.read_cache:
                    reply = self.host.act(finder, call)
                    searched += 1
                    if reply is None or self.host.finished:
                        break
                    acted = True
                    found = reply.get("observation")
                    self.host.book.observe(finder, found)
                    self.host.read_cache[finder + render(call)] = (self.host.book.version, self.host.step, self.host.show(found))
                    seen.append(f"search {render(call)}: " + self.host.show(found)[:2500])
            if not acted:
                break
        text = ""
        if seen:
            text += "[Act opened the pages a requirement points at, or searched for them]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 4 and self.host.left() > 3 * len(lost) + 4
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

class Coverage:
    def __init__(self, host: Any):
        self.host = host
        self.pages_read = 0
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.found: list[dict[str, Any]] = []
        self.cover_marks: set[str] = set()
        self.browseMarks: set[str] = set()
        self.rank: dict[str, int] = {}
        self.recheck_marks: set[str] = set()
        self.recent_search: list[tuple[dict[str, Any], Any]] = []

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        host = self.host
        self.paged_obs = []
        pick = host.store.brief.get("pick")
        plan = host.store.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown or (pick == "cheapest" and host.shelf.price_sorted(observation)):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 3 or self.pages_read + pages > host.budgets.autoPages or host.left() - host.book.tail() - pages < 12:
            return (f"\n[Act] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={render(value)}.")
        texts = []
        for _ in range(pages):
            page = self.readPage(name, {**args, param: value})
            if page is None:
                break
            seen, shown_text = page
            texts.append(f"{param}={render(value)}: {shown_text}")
            after = host.store.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        host.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Act read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def readPage(self, name: str, call: dict[str, Any]) -> tuple[Any, str] | None:
        host = self.host
        call = host.store.shapedArgs(name, call)
        if call is None:
            return None
        reply = host.act(name, call)
        if reply is None or host.finished:
            return None
        self.pages_read += 1
        seen = reply.get("observation")
        host.book.observe(name, seen)
        self.paged_obs.append(seen)
        shown_text = host.show(seen)
        host.read_cache[name + render(call)] = (host.book.version, host.step, shown_text)
        return seen, shown_text

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.host.store.brief.get("pick") != "cheapest" or not self.host.store.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.host.shelf.records_in(seen)}
        rows = [(self.host.shelf.price_of(i), i) for i in found if self.host.shelf.buyable(i) and self.host.shelf.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Act] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        if self.wider_held or self.host.store.brief.get("pick") != "cheapest" or self.host.left() < 14 + 2 * len(self.host.book.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.host.shelf.price_of(i):g}" for i in self.host.shelf.cart_ids() if self.host.shelf.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.host.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def note_search(self, name: str, args: dict[str, Any], observation: Any) -> None:
        shape = self.host.store.read_shape(name, observation)
        if not shape or not shape.get("next_arg") or not shape.get("next"):
            return
        views = [observation] + [o for o in self.paged_obs if isinstance(o, dict)]
        last = views[-1]
        value = last.get(shape["next"]) if isinstance(last, dict) else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            value = None
        ids = [i for view in views for i, _ in self.host.shelf.records_in(view)]
        if not ids:
            return
        prices = [p for p in (self.host.shelf.price_of(i) for i in ids) if p is not None]
        base = {k: v for k, v in args.items() if k != shape["next_arg"]}
        prior = next((e for e in reversed(self.found) if e["tool"] == name and e["args"] == base), None) if shape["next_arg"] in args else None
        self.noteRank(ids, len(prior["ids"]) if prior else 0)
        count = last.get(shape["count"]) if shape.get("count") and isinstance(last, dict) else None
        self.found.append({"tool": name, "args": base, "next": value, "ids": ids, "step": self.host.step, "sorted": self.host.shelf.price_sorted(last),
                           "top": max(prices) if prices else None, "count": count if isinstance(count, int) and not isinstance(count, bool) else None})

    def best_found(self, kind: str, key: str | None) -> dict[str, Any] | None:
        best = None
        for entry in reversed(self.found):
            score = sum(1 for i in entry["ids"] if self.host.shelf.kind_of(i, key) == kind)
            if score and (best is None or score > best[1]):
                best = (entry, score)
        return best[0] if best else None

    def search_args(self, finder: str, words: str) -> dict[str, Any] | None:
        for args, observation in self.recent_search:
            shape = self.host.store.read_shape(finder, observation) or {}
            key = shape.get("query")
            if not key:
                strings = [k for k, v in args.items() if isinstance(v, str) and v.strip() and k != shape.get("next_arg")]
                key = max(strings, key=lambda k: len(args[k])) if strings else None
            if not key or not str(args.get(key) or "").strip():
                continue
            base = {k: v for k, v in args.items() if k != shape.get("next_arg")}
            base[key] = words
            faults: list[str] = []
            call = StoreMap.shape(base, self.host.store.schemas.get(finder), "args", faults)
            if not faults:
                return call
        return None

    def cover_pages(self) -> list[str]:
        if self.host.store.brief.get("pick") != "cheapest" or not self.host.store.price_key or not self.found:
            return []
        mark = self.host.shelf.cart_mark() + f"#{self.host.book.version}"
        key = self.host.shelf.category_key()
        if mark in self.cover_marks or not key:
            return []
        self.cover_marks.add(mark)
        cart = self.host.shelf.cart_ids()
        room = min(self.host.budgets.coverPages, self.host.left() - self.host.book.tail() - 3 * len(cart) - 4)
        texts: list[str] = []
        read = 0
        for line in cart:
            price = self.host.shelf.price_of(line)
            kind = self.host.shelf.kind_of(line, key)
            if price is None or not kind:
                continue
            entry = self.best_found(kind, key)
            if not entry and room > 0 and not self.host.finished and self.host.shelf.item_of(line):
                _, used = self.browsePass([self.host.shelf.item_of(line)], min(room, 4), price)
                room -= used
                read += used
                entry = self.best_found(kind, key)
            if not entry or room <= 0 or self.host.finished:
                continue
            tool = entry["tool"]
            shape = self.host.store.shapes.get(tool) or {}
            if shape.get("cap") and not entry.get("capped"):
                entry["capped"] = True
                faults: list[str] = []
                call = StoreMap.shape({**entry["args"], shape["cap"]: price - 1}, self.host.store.schemas.get(tool), "args", faults)
                if not faults and tool + render(call) not in self.host.read_cache:
                    reply = self.host.act(tool, call)
                    room -= 1
                    if reply is None or self.host.finished:
                        break
                    seen = reply.get("observation")
                    self.host.book.observe(tool, seen)
                    self.host.read_cache[tool + render(call)] = (self.host.book.version, self.host.step, self.host.show(seen))
                    self.paged_obs = []
                    self.note_search(tool, call, seen)
                    read += 1
                    ids = [i for i, _ in self.host.shelf.records_in(seen)]
                    if ids and all(self.host.shelf.price_of(i) is None or self.host.shelf.price_of(i) < price for i in ids) and self.found and self.found[-1]["step"] == self.host.step:
                        entry = self.found[-1]
                        entry["capped"] = True
            value = entry.get("next")
            if entry.get("sorted") and entry.get("top") is not None and entry["top"] >= price:
                value = None
            while value is not None and room > 0 and not self.host.finished and shape.get("next_arg"):
                faults = []
                call = StoreMap.shape({**entry["args"], shape["next_arg"]: value}, self.host.store.schemas.get(tool), "args", faults)
                if faults or tool + render(call) in self.host.read_cache:
                    break
                reply = self.host.act(tool, call)
                room -= 1
                if reply is None:
                    break
                seen = reply.get("observation")
                self.host.book.observe(tool, seen)
                self.host.read_cache[tool + render(call)] = (self.host.book.version, self.host.step, self.host.show(seen))
                read += 1
                ids = [i for i, _ in self.host.shelf.records_in(seen)]
                self.noteRank(ids, len(entry["ids"]))
                entry["ids"] = entry["ids"] + ids
                after = self.host.store.paging(tool, seen)
                value = after[3] if after else None
                entry["next"] = value
                here = [p for p in (self.host.shelf.price_of(i) for i in ids if self.host.shelf.kind_of(i, key) == kind) if p is not None]
                if not ids or not here:
                    break
                if self.host.shelf.price_sorted(seen) and min(here) >= price:
                    break
            pool = self.cheaperPool(cart, key, kind, price)
            if not [i for i in pool if i not in self.host.shelf.full] and room > 0 and not self.host.finished and self.host.shelf.item_of(line):
                _, used = self.browsePass([self.host.shelf.item_of(line)], min(room, 4), price)
                room -= used
                read += used
                pool = self.cheaperPool(cart, key, kind, price)
            if pool:
                texts.append(f"buyable rows of the same kind as {line} ({price:g}) priced below it, cheapest first: "
                             + ", ".join(f"{i} ({self.host.shelf.price_of(i):g}" + (", attributes read)" if i in self.host.shelf.full else ")") for i in pool[:8]))
        if read:
            self.host.trace.append({"kind": "cover", "pages": read, "cart": cart})
        return texts

    def cheaperPool(self, cart: list[str], key: str, kind: str, price: float) -> list[str]:
        shelf = self.host.shelf
        pool = [i for i in shelf.seen_order if i not in cart and shelf.buyable(i) and shelf.kind_of(i, key) == kind
                and shelf.price_of(i) is not None and shelf.price_of(i) < price]
        pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
        return pool

    def vocabDf(self) -> tuple[dict[str, int], int]:
        df: dict[str, int] = {}
        rows = self.host.shelf.rows
        for record in rows.values():
            for word in set(re.findall(r"[a-z0-9]+", render(record).lower())):
                df[word] = df.get(word, 0) + 1
        return df, len(rows)

    def valueWords(self, need: str, df: dict[str, int], total: int) -> list[str]:
        value = need.split(":", 1)[1] if ":" in need else need
        words: list[str] = []
        for word in re.findall(r"[a-z0-9]+", value.lower()):
            if word not in words and 1 <= df.get(word, 0) <= 0.5 * total:
                words.append(word)
        words.sort(key=lambda w: df[w])
        return words[:3]

    def ownEntries(self, number: int) -> list[dict[str, Any]]:
        ranked = []
        for entry in self.found:
            if entry.get("next") is None or not entry.get("ids"):
                continue
            mine = sum(1 for i in entry["ids"] if self.host.shelf.item_of(i) == number)
            if mine:
                count = entry.get("count")
                ranked.append((count if isinstance(count, int) else 10 ** 6, -mine / len(entry["ids"]), -entry["step"], entry))
        ranked.sort(key=lambda r: r[:3])
        return [r[3] for r in ranked[:3]]

    def browsePass(self, numbers: list[int], room: int, below: float | None = None) -> tuple[list[str], int]:
        host = self.host
        shelf = host.shelf
        finder = host.store.roles.get("search")
        numbers = [n for n in numbers if 1 <= n <= len(host.book.items)]
        mark = f"{sorted(numbers)}#{host.book.version}"
        if not finder or room <= 0 or not numbers or mark in self.browseMarks or host.finished:
            return [], 0
        self.browseMarks.add(mark)
        talk = Interviewer(host)
        df, total = self.vocabDf()
        shape = host.store.shapes.get(finder) or {}
        pointing = {need for _, need in host.lookups.refs}
        plan: list[tuple[int, str, Any]] = []
        for number in numbers:
            label = re.findall(r"[a-z0-9]+", talk.itemLabel(number).lower())
            needs = [need for n, need in host.book.needs_now() if n == number and need not in pointing]
            for need in needs[:4]:
                words = self.valueWords(need, df, total)
                if not words:
                    continue
                call = self.search_args(finder, " ".join(words + label))
                if call is not None:
                    plan.append((number, "value", call))
            for need in sorted(needs[:4], key=lambda text: bool(self.valueWords(text, df, total)))[:1]:
                key = re.findall(r"[a-z0-9]+", need.split(":", 1)[0].lower()) if ":" in need else []
                call = self.search_args(finder, " ".join(key + [w for w in label if w not in key])) if 1 <= len(key) <= 3 else None
                if call is not None and all(call != spec for _, _, spec in plan):
                    plan.append((number, "key", call))
        for _ in range(2):
            for number in numbers:
                for entry in self.ownEntries(number):
                    plan.append((number, "page", entry))
        fresh: dict[int, list[str]] = {n: [] for n in numbers}
        used = 0
        queries: list[str] = []
        for number, kind, spec in plan:
            if used >= room or host.finished or len(fresh[number]) >= 6:
                continue
            if kind == "page":
                value = spec.get("next")
                if value is None or not shape.get("next_arg"):
                    continue
                faults: list[str] = []
                call = StoreMap.shape({**spec["args"], shape["next_arg"]: value}, host.store.schemas.get(finder), "args", faults)
                if faults:
                    continue
            else:
                call = spec
            key = finder + render(call)
            if key in host.read_cache:
                continue
            reply = host.act(finder, call)
            used += 1
            if reply is None or host.finished:
                break
            seen = reply.get("observation")
            host.book.observe(finder, seen)
            host.read_cache[key] = (host.book.version, host.step, host.show(seen))
            self.paged_obs = []
            if kind == "page":
                got = [i for i, _ in shelf.records_in(seen)]
                self.noteRank(got, len(spec["ids"]))
                spec["ids"] = spec["ids"] + got
                after = host.store.paging(finder, seen)
                spec["next"] = after[3] if after else None
            else:
                self.note_search(finder, call, seen)
            queries.append(render(call))
            for ident, _ in shelf.records_in(seen):
                if ident in shelf.full or ident in shelf.opened or ident in fresh[number] or not shelf.buyable(ident) or shelf.item_of(ident) != number:
                    continue
                if below is None or (shelf.price_of(ident) is not None and shelf.price_of(ident) < below):
                    fresh[number].append(ident)
        out = [i for n in numbers for i in fresh[n]]
        if host.store.brief.get("pick") == "cheapest":
            out.sort(key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))
        if used:
            host.trace.append({"kind": "browse", "items": numbers, "pages": used, "queries": queries[:8], "fresh": out[:12]})
        return out, used

    def noteRank(self, ids: list[str], start: int) -> None:
        for place, ident in enumerate(ids, start):
            if place < self.rank.get(ident, place + 1):
                self.rank[ident] = place

    def bestRows(self, pool: list[str]) -> list[str]:
        shelf = self.host.shelf
        order = {ident: k for k, ident in enumerate(pool)}
        near = sorted(pool, key=lambda i: (self.rank.get(i, len(order) + len(self.rank)), order[i]))
        cheap = sorted(pool, key=lambda i: (shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"), order[i]))
        out: list[str] = []
        for pair in zip(near, cheap):
            out += [i for i in dict.fromkeys(pair) if i not in out]
        return out

    def flaggedRows(self, verdict: str) -> list[str]:
        host = self.host
        shelf = host.shelf
        if not verdict or host.finished or host.budgets.closing() or len(self.recheck_marks) >= 2:
            return []
        said = " ".join(host.checker.problems)
        cart = shelf.cart_ids()
        numbers = sorted({shelf.item_of(i) for i in cart if re.search(r"(?<![\w-])" + re.escape(i) + r"(?![\w-])", said)} - {0})
        room = min(host.budgets.coverPages, (host.left() - host.book.tail() - len(host.book.items) - 6) // 2)
        mark = f"{numbers}#{shelf.cart_mark()}#{host.book.version}"
        if not numbers or room < 2 or mark in self.recheck_marks:
            return []
        self.recheck_marks.add(mark)
        _, used = self.browsePass(numbers, min(room, 2 + len(numbers)))
        picks = [i for i in host.decline.unread_rows(numbers, 2) if i not in cart]
        opened = shelf.openMore(picks[: 2 * len(numbers)], host.book.tail() + len(host.book.items) + 3)
        if not used and not opened:
            return []
        host.trace.append({"kind": "recheck_rows", "items": numbers, "pages": used, "opened": [t.split(":", 1)[0] for t in opened]})
        return [f"Act searched again for item(s) {', '.join(map(str, numbers))} the review faulted" + (" and read the best unread listings:" if opened else "; nothing new to read.")] + opened

    def cheaper_rows(self) -> list[str]:
        host = self.host
        shelf = host.shelf
        if host.store.brief.get("pick") != "cheapest":
            return []
        mark = shelf.cart_mark() + f"#{host.book.version}"
        cat_key = shelf.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = shelf.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = shelf.price_of(line)
            kind = shelf.kind_of(line, cat_key)
            if price is None or not kind:
                continue
            pool = [i for i in shelf.seen_order if i not in cart and i not in shelf.full and shelf.buyable(i) and shelf.kind_of(i, cat_key) == kind
                    and (shelf.price_of(i) if shelf.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return shelf.openMore(picks[:host.budgets.cheaperOpens], host.book.tail() + 4)

class Checker:
    def __init__(self, host: Any):
        self.host = host
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.problems: list[str] = []
        self.better: list[str] = []
        self.checkHard: dict[str, bool] = {}
        self.checkHolds: dict[str, int] = {}
        self.checkText: dict[str, str] = {}
        self.altFor: dict[str, str] = {}
        self.notes: dict[str, dict[str, Any]] = {}
        self.joint_tried: set[str] = set()

    def check_key(self, ids: list[str]) -> str:
        return f"{'|'.join(sorted(ids))}#{self.host.book.version}"

    def checkLedger(self) -> tuple[str, dict[str, str]]:
        parts = []
        names: dict[str, str] = {}
        for number, item in enumerate(self.host.book.items, 1):
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            texts = []
            for k, need in enumerate(needs, 1):
                tag = f"R{number}.{k}"
                names[tag] = need
                texts.append(tag + ": " + need + (f" (the page it points at gives: {self.host.lookups.ref_values[need]})" if need in self.host.lookups.ref_values else ""))
            text = head + "; requirements: " + ("; ".join(texts) if texts else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        pick = self.host.store.brief.get("pick")
        if pick == "cheapest":
            parts.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            parts.append(f"pick: the {pick} {self.host.store.brief.get('pick_measure') or ''} among fitting listings".replace("  ", " "))
        else:
            parts.append("pick: any listing that fits everything")
        if self.host.store.brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        return "\n".join("- " + p for p in parts), names

    def lineItem(self, ident: str, label: str, used: set[int]) -> tuple[int, bool]:
        if len(self.host.book.items) == 1:
            return 1, True
        number = next((Text.count(n) for n, members in self.host.decline.lead_items.items() if ident in members and Text.count(n)), None)
        if number and 1 <= number <= len(self.host.book.items):
            return number, True
        if label:
            exact = next((n for n, it in enumerate(self.host.book.items, 1) if n not in used and it.shelf and it.shelf.strip().lower() == label.lower()), None)
            if exact:
                return exact, True
            want = Text.bag(label)
            scored: list[tuple[float, int]] = []
            for n, it in enumerate(self.host.book.items, 1):
                if n in used:
                    continue
                best = 0.0
                for name in (it.shelf, it.name):
                    have = Text.bag(name or "")
                    if have and want:
                        best = max(best, len(want & have) / len(want | have))
                scored.append((best, n))
            scored.sort(key=lambda s: (-s[0], s[1]))
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1], True
        return 0, False

    def checkLines(self, ids: list[str], lead: bool) -> list[dict[str, Any]]:
        catKey = self.host.shelf.category_key()
        used: set[int] = set()
        out: list[dict[str, Any]] = []
        for ident in ids:
            label = str((self.host.shelf.rows.get(ident) or {}).get(catKey) or "").strip() if catKey else ""
            number, sure = self.lineItem(ident, label, used)
            if number:
                used.add(number)
            out.append({"id": ident, "item": number, "sure": sure, "label": label, "price": self.host.shelf.price_of(ident)})
        for line in out:
            if not line["item"]:
                line["item"] = next((n for n in range(1, len(self.host.book.items) + 1) if n not in used), 0)
                if line["item"]:
                    used.add(line["item"])
        for line in out:
            line["want"] = self.host.book.items[line["item"] - 1].qty if line["item"] else 1
            line["have"] = line["want"] if lead else self.host.shelf.cart_lines.get(line["id"], line["want"])
        return out

    def placeLines(self, lines: list[dict[str, Any]], got: dict[str, Any] | None, lead: bool) -> None:
        if not isinstance(got, dict) or all(line["sure"] for line in lines):
            return
        taken = {line["item"] for line in lines if line["sure"]}
        said: dict[str, int] = {}
        for entry in Text.items(got.get("lines")):
            if isinstance(entry, dict):
                number = Text.count(entry.get("item"))
                if number and 1 <= number <= len(self.host.book.items):
                    said[str(entry.get("id") or "").strip()] = number
        placed: list[dict[str, Any]] = []
        for line in lines:
            number = said.get(line["id"])
            if not line["sure"] and number and number not in taken:
                taken.add(number)
                line["item"] = number
                placed.append(line)
        for line in lines:
            if not line["sure"] and line not in placed and line["item"] in taken:
                line["item"] = next((n for n in range(1, len(self.host.book.items) + 1) if n not in taken), line["item"])
                taken.add(line["item"])
        for line in lines:
            if not line["sure"] and line["item"]:
                line["want"] = self.host.book.items[line["item"] - 1].qty
                line["have"] = line["want"] if lead else self.host.shelf.cart_lines.get(line["id"], line["want"])

    def codeProblems(self, lines: list[dict[str, Any]], lead: bool) -> list[str]:
        problems: list[str] = []
        for line in lines:
            if self.host.shelf.sold_out(line["id"]):
                problems.append(f"{line['id']} is sold out")
            if line["have"] != line["want"]:
                problems.append(f"{line['id']}: {line['have']} in the cart, the request asks for {line['want']}")
            shelf = self.host.book.items[line["item"] - 1].shelf if line["item"] else ""
            if shelf and line["label"] and shelf.strip().lower() != line["label"].lower():
                problems.append(f"{line['id']}: its label '{line['label']}' is not the item's store label '{shelf}'")
        if not lead:
            covered = {line["item"] for line in lines}
            for number, item in enumerate(self.host.book.items, 1):
                if number not in covered:
                    problems.append(f"no cart line for item {number} ({item.name})")
        budget = self.host.book.budget
        if lines and all(line["price"] is not None for line in lines) and isinstance(budget, (int, float)) and not isinstance(budget, bool):
            if self.host.store.brief.get("budget_scope") == "each":
                for line in lines:
                    if line["price"] * line["have"] > budget:
                        problems.append(f"{line['id']}: {line['price'] * line['have']:g} is over the budget of {budget:g} for each item")
            else:
                total = sum(line["price"] * line["have"] for line in lines)
                if total > budget:
                    problems.append(f"the total {total:g} is over the budget of {budget:g}")
        return problems

    def itemLabels(self) -> dict[int, str]:
        if not self.host.shelf.category_key():
            return {}
        talk = Interviewer(self.host)
        return {n: Text.words(talk.itemLabel(n)) for n in range(1, len(self.host.book.items) + 1)}

    def sameItem(self, ident: str, line: dict[str, Any], labels: dict[int, str], known: set[str], catKey: str | None) -> bool:
        number = line["item"]
        if number and ident in self.host.decline.lead_items.get(str(number), []):
            return True
        if not catKey:
            return len(self.host.book.items) == 1
        label = Text.words(str((self.host.shelf.rows.get(ident) or {}).get(catKey) or ""))
        want = labels.get(number, "") if number else ""
        if want and want in known:
            return label == want
        return bool(label) and bool(line["label"]) and label == Text.words(line["label"])

    def cheaperSeen(self, lines: list[dict[str, Any]], flagged: dict[str, bool] | None = None) -> list[str]:
        shelf = self.host.shelf
        cheapest = self.host.store.brief.get("pick") == "cheapest"
        flagged = flagged or {}
        catKey = shelf.category_key()
        labels = self.itemLabels()
        known = {Text.words(str(row.get(catKey) or "")) for row in shelf.rows.values()} if catKey else set()
        cart = {line["id"] for line in lines}
        found: list[str] = []
        room = 24000
        for line in lines:
            if (cheapest and line["price"] is None) or (not cheapest and line["id"] not in flagged):
                continue
            pool = [i for i in shelf.seen_order if i not in cart and i not in found and shelf.buyable(i) and shelf.price_of(i) is not None
                    and (not cheapest or (shelf.price_of(i) or 0.0) < line["price"]) and self.sameItem(i, line, labels, known, catKey)]
            pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
            rows = [i for i in pool if i not in shelf.full][:2] if cheapest else []
            kept = 0
            for i in pool:
                if i not in shelf.full and i not in rows:
                    continue
                size = len(render(shelf.full.get(i) or shelf.rows.get(i), 240)[:1500])
                if kept >= 2 and size > room:
                    break
                room -= size
                kept += 1
                self.altFor[i] = line["id"]
                found.append(i)
        return found

    def altText(self, lines: list[dict[str, Any]], alts: list[str]) -> str:
        shelf = self.host.shelf
        parts = []
        for line in lines:
            mine = [i for i in alts if self.altFor.get(i) == line["id"]]
            if not mine:
                continue
            head = f"for item {line['item'] or '?'}, in place of {line['id']}" + (f" (price {line['price']:g})" if line["price"] is not None else "") + ":"
            parts.append(head + "\n" + "\n".join(f"{i}" + (f" (price {shelf.price_of(i):g})" if shelf.price_of(i) is not None else "") + ": " + render(shelf.full.get(i) or shelf.rows.get(i), 240)[:1500] for i in mine))
        return "\n".join(parts)

    def altProblem(self, alt: str, lines: list[dict[str, Any]]) -> str:
        line = next((x for x in lines if x["id"] == self.altFor.get(alt)), None)
        if line is None:
            return f"{alt} costs less than the cart's line of the same kind and its record meets every requirement"
        price = self.host.shelf.price_of(alt)
        if self.host.store.brief.get("pick") == "cheapest" and line["price"] is not None and price is not None and price < line["price"]:
            return f"{alt} meets every requirement of item {line['item']} and costs less than {line['id']}, so it should replace that line"
        return f"{alt} meets every requirement of item {line['item']}, which {line['id']} fails, so it should replace that line"

    def codeHard(self, line: dict[str, Any]) -> bool:
        shelf = self.host.book.items[line["item"] - 1].shelf if line["item"] else ""
        return self.host.shelf.sold_out(line["id"]) or bool(shelf and line["label"] and shelf.strip().lower() != line["label"].lower())

    def lineFlags(self, lines: list[dict[str, Any]], said: list[str], got: dict[str, Any] | None) -> dict[str, bool]:
        hard: set[str] = set()
        for entry in Text.items(got.get("lines") if isinstance(got, dict) else None):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            grades = [self.gradeParts(g) for g in Text.items(entry.get("grades"))]
            if str(entry.get("label") or "").strip().lower().startswith("neighbour") or any(len(g) >= 2 and g[1].lower().startswith("contradict") for g in grades):
                hard.add(ident)
        flags: dict[str, bool] = {}
        for line in lines:
            code = self.codeHard(line)
            if code or any(p.startswith(line["id"] + ":") for p in said):
                flags[line["id"]] = code or line["id"] in hard
        return flags

    def otherFits(self, got: dict[str, Any] | None, shown: list[str]) -> list[str]:
        fits: list[str] = []
        if not isinstance(got, dict):
            return fits
        for entry in Text.items(got.get("others")) + Text.items(got.get("cheaper")):
            parts = self.gradeParts(entry)
            if len(parts) >= 2 and parts[0] in shown and parts[1].lower().startswith("fit") and parts[0] not in fits:
                fits.append(parts[0])
        return fits

    def recall(self, key: str) -> None:
        note = self.notes.get(key)
        if note:
            self.problems, self.better, self.altFor = list(note["problems"]), list(note["better"]), dict(note["alt"])

    def gradeParts(self, grade: Any) -> list[str]:
        if isinstance(grade, list):
            parts = grade
        elif isinstance(grade, dict):
            parts = [grade.get(k) for k in ("requirement", "grade", "field", "quote", "other", "other_quote")]
            while parts and parts[-1] is None:
                parts.pop()
        else:
            return []
        return [str(p).strip() if p is not None else "" for p in parts]

    def quoteIn(self, ident: str, quote: str) -> bool:
        if not quote:
            return False
        text = re.sub(r"\s+", " ", render(self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident) or {}).lower())
        return re.sub(r"\s+", " ", quote.lower()) in text

    def gradeVerdict(self, got: dict[str, Any] | None, lines: list[dict[str, Any]], names: dict[str, str], cheaper: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]], bool]:
        problems: list[str] = []
        better: list[str] = []
        disputed: list[tuple[str, str]] = []
        hard = False
        if not isinstance(got, dict):
            return problems, better, disputed, hard
        ids = {line["id"] for line in lines}
        for entry in Text.items(got.get("lines")):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            if ident not in ids:
                continue
            if str(entry.get("label") or "").strip().lower().startswith("neighbour"):
                problems.append(f"{ident}: its category label names a neighbouring kind, not the requested product itself")
            for grade in Text.items(entry.get("grades")):
                parts = self.gradeParts(grade)
                if len(parts) < 2:
                    continue
                tag, verdict = parts[0], parts[1].lower()
                field = parts[2] if len(parts) > 2 else ""
                quote = parts[3] if len(parts) > 3 else ""
                need = names.get(tag, tag)
                value = need.split(":", 1)[1] if ":" in need else need
                where = f" ({field}: '{quote}')" if quote else (f" (field {field})" if field else "")
                if verdict.startswith("absent") or verdict.startswith("contradict"):
                    problems.append(f"{ident}: '{need}' is {verdict}{where}")
                    disputed.append((ident, tag))
                    if verdict.startswith("contradict") and self.quoteIn(ident, quote):
                        hard = True
                    continue
                if len(parts) >= 6 and verdict.startswith(("exact", "words")) and (str(parts[4]).strip() or str(parts[5]).strip()):
                    other = str(parts[4]).strip()
                    if other not in ids:
                        problems.append(f"{ident}: '{need}' was graded against {other or 'no listing'}, which is not a cart line")
                        disputed.append((ident, tag))
                    elif not (self.quoteIn(ident, quote) and self.quoteIn(other, parts[5])):
                        problems.append(f"{ident}: '{need}' is not shown: the words quoted for the fit were not found in both records{where}")
                        disputed.append((ident, tag))
                    continue
                if verdict.startswith("words") and not Text.phraseIn(value, quote):
                    problems.append(f"{ident}: '{need}' is not stated as one unbroken phrase{where}")
                    disputed.append((ident, tag))
        for alt in self.otherFits(got, cheaper):
            problems.append(self.altProblem(alt, lines))
            if alt not in better:
                better.append(alt)
        for b in Text.items(got.get("better")):
            b = str(b).strip()
            if b in self.host.shelf.rows and not self.host.shelf.sold_out(b) and b not in better and b not in ids:
                better.append(b)
        return problems, better, disputed, hard

    def resolved(self, got: dict[str, Any], ident: str, tag: str, need: str = "") -> bool:
        if not isinstance(got, dict):
            return False
        for entry in Text.items(got.get("lines")):
            if not isinstance(entry, dict) or str(entry.get("id") or "").strip() != ident:
                continue
            for grade in Text.items(entry.get("grades")):
                parts = self.gradeParts(grade)
                if len(parts) >= 2 and parts[0] == tag:
                    verdict = parts[1].lower()
                    quote = parts[3] if len(parts) > 3 else ""
                    if not verdict.startswith(("exact", "words", "category")) or not self.quoteIn(ident, quote):
                        return False
                    if verdict.startswith("words") and need and not Text.phraseIn(need.split(":", 1)[1] if ":" in need else need, quote):
                        return False
                    if len(parts) >= 6 and (str(parts[4]).strip() or str(parts[5]).strip()):
                        other = str(parts[4]).strip()
                        return other in self.host.shelf.rows and self.quoteIn(other, parts[5])
                    return True
        return False

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None, wide: bool = True) -> str:
        lead = ids is not None
        ids = ids if lead else self.host.shelf.cart_ids()
        key = self.check_key(ids)
        limit = self.host.budgets.spendHard if limit is None else limit
        if key in self.checked:
            self.recall(key)
            if not lead and self.verdicts.get(key) is False and self.checkHard.get(key) and self.checkHolds.get(key, 0) < 1 and self.host.left() > self.host.book.tail() + 3:
                self.checkHolds[key] = 1
                return self.checkText.get(key, "")
            return ""
        if self.host.models.cost > limit or not ids:
            return ""
        self.checked.add(key)
        self.problems, self.better, self.altFor = [], [], {}
        lines = self.checkLines(ids, lead)
        ledger, names = self.checkLedger()
        cheaper = self.cheaperSeen(lines)
        base = CHECK + "\n\nREQUEST:\n" + self.host.store.request + "\n\nREQUIREMENTS IN FORCE:\n" + ledger
        base += "\n\nCART LINES" + (" (planned; only these lines are judged, the other items are handled separately)" if lead else "") + ":\n"
        base += "\n".join(f"{line['id']}: " + (f"item {line['item']}" if line["sure"] else "item to decide (say which requested item this line is for)") + f", quantity {line['have']}" + (f", label {line['label']}" if line["label"] else "") + (f", price {line['price']:g}" if line["price"] is not None else "") for line in lines)
        base += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.host.shelf.sold_out(i) else "") + self.host.shelf.record_of(i) for i in ids)
        labels = self.host.shelf.storeLabels()
        if len(labels) > 1:
            base += "\n\nCATEGORY LABELS SEEN IN THIS STORE: " + "; ".join(labels)
        base += self.codeLines(lines)
        prompt = base
        if cheaper:
            prompt += "\n\nCHEAPER LISTINGS OF THE CART'S ITEMS SEEN (grade each in \"cheaper\" against the requirements of the item it stands under; cheapest first):\n" + self.altText(lines, cheaper)
        if lead or self.host.store.brief.get("pick") in ("most", "least") or (self.host.store.brief.get("pick") == "cheapest" and not cheaper):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.host.shelf.others(ids + cheaper, 12)
        got = self.host.models.ask_json(prompt, [LIGHT], 6000, "medium")
        if not isinstance(got, dict) or not isinstance(got.get("lines"), list) or not got.get("lines"):
            got = self.host.models.ask_json(prompt, [HEAVY], 3500, "medium")
        self.placeLines(lines, got, lead)
        coded = self.codeProblems(lines, lead)
        said, better, disputed, hard = self.gradeVerdict(got, lines, names, cheaper)
        second = None
        if not coded and said and len(said) == len(disputed):
            asked = ", ".join(f"{tag} on {ident}" for ident, tag in disputed)
            second = self.host.models.ask_json(base + "\n\nA first reading graded " + asked + " as absent, contradicted or not shown as one unbroken phrase. Re-read those records in full before grading, and quote the field that states each value when one does.", [HEAVY, LIGHT], 3500, "medium")
            if second and all(self.resolved(second, ident, tag, names.get(tag, "")) for ident, tag in disputed):
                said = self.labelTurned(got, second, lines)
                hard = False
        problems = coded + said
        flags = self.lineFlags(lines, said, got) if problems else {}
        third = None
        wider: list[str] = []
        if wide and flags and self.host.models.cost <= limit and self.host.left() > len(ids) + 4 and len({line["item"] for line in lines}) >= len(self.host.book.items):
            wider = [i for i in self.cheaperSeen(lines, flags) if i not in cheaper]
            if wider and not self.host.budgets.closing():
                faults = [p for p in problems if any(p.startswith(i + ":") for i in flags)]
                third = self.host.models.ask_json(base + "\n\nThis check found these cart lines failing: " + "; ".join(faults[:6]) + "\n\nOTHER LISTINGS OF THE FAILING LINES' ITEMS (cheapest first):\n" + self.altText(lines, wider)
                                                  + "\n\nGrade only these other listings now, each against every requirement of the item it stands under, from its own record, with the cart's other lines as they are. Return ONE JSON object and nothing else: {\"others\": [[listing id, \"fits\" or \"fails\", requirement number or \"\", quote]]}", [LIGHT], 6000, "medium")
                for alt in self.otherFits(third, wider):
                    problems.append(self.altProblem(alt, lines))
                    if alt not in better:
                        better.append(alt)
        ok = not problems
        self.host.trace.append({"kind": "check", "cart": ids, "items": [line["item"] for line in lines], "got": got, "second": second, "ok": ok, "problems": problems[:6], "shown": cheaper + wider, "third": third})
        self.verdicts[key] = ok
        self.notes[key] = {"problems": problems[:8], "better": better[:6], "alt": dict(self.altFor), "flags": flags, "items": {line["id"]: line["item"] for line in lines}}
        if ok:
            return ""
        self.host.decline.leads = [i for i in self.host.decline.leads if i not in ids]
        self.problems, self.better = problems[:8], better[:6]
        self.checkHard[key] = bool(coded) or hard
        text = "A careful review says this cart is not right yet: " + "; ".join(problems[:6])
        if better:
            text += ". Listings that look better: " + ", ".join(better[:4])
        text += ". Fix the cart (or show the review is wrong by re-reading the records), then check out."
        self.checkText[key] = text
        return text

    def labelTurned(self, got: dict[str, Any] | None, second: dict[str, Any] | None, lines: list[dict[str, Any]]) -> list[str]:
        ids = {line["id"] for line in lines}
        before = {str(e.get("id") or "").strip(): str(e.get("label") or "").strip().lower() for e in Text.items(got.get("lines") if isinstance(got, dict) else None) if isinstance(e, dict)}
        turned: list[str] = []
        for entry in Text.items(second.get("lines") if isinstance(second, dict) else None):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            if ident in ids and str(entry.get("label") or "").strip().lower().startswith("neighbour") and not before.get(ident, "").startswith("neighbour"):
                turned.append(f"{ident}: its category label names a neighbouring kind, not the requested product itself")
        return turned

    @staticmethod
    def nameBag(text: Any) -> frozenset[str]:
        return frozenset(Text.single(w)[:5] for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2)

    def linkedNeeds(self) -> dict[int, list[tuple[int, str]]]:
        items = self.host.book.items
        if len(items) < 2:
            return {}
        pointing = {need for _, need in self.host.lookups.refs}
        names = {m: [Checker.nameBag(it.name), Checker.nameBag(it.shelf)] for m, it in enumerate(items, 1)}
        out: dict[int, list[tuple[int, str]]] = {}
        for number, item in enumerate(items, 1):
            for need in item.musts + item.told:
                if need in item.dropped or need in pointing:
                    continue
                have = Checker.nameBag(need)
                other = next((m for m, bags in names.items() if m != number and any(b and b <= have for b in bags)), 0)
                if other and (other, need) not in out.get(number, []):
                    out.setdefault(number, []).append((other, need))
        return out

    @staticmethod
    def codeSet(text: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for hit in re.finditer(r"(?<![a-z0-9])[a-z][a-z0-9]*\d[a-z0-9]*(?:-[a-z0-9]+)*|(?<![a-z0-9])[a-z]{2,} \d{2,}(?![a-z0-9])", text):
            if len(hit.group(0)) >= 3:
                found.setdefault(hit.group(0), hit.start())
        return found

    @staticmethod
    def family(pattern: str, code: str) -> bool:
        fixed = re.fullmatch(r"([a-z0-9-]*\d)(x+)", pattern)
        return bool(fixed) and code != pattern and len(code) == len(pattern) and code.startswith(fixed.group(1))

    def codeText(self, ident: str) -> str:
        record = self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident) or {}
        return re.sub(r"\s+", " ", render({k: v for k, v in record.items() if v != ident} if isinstance(record, dict) else record).lower())

    def sharedCodes(self, one: str, two: str) -> str:
        full = self.host.shelf.full
        if not one or not two or one == two:
            return ""
        texts = [self.codeText(one), self.codeText(two)]
        mine, theirs = Checker.codeSet(texts[0]), Checker.codeSet(texts[1])
        common = len(full) >= 5
        hits: list[str] = []
        for x, i in mine.items():
            for y, j in theirs.items():
                if not (x == y or y.startswith(x + "-") or x.startswith(y + "-") or Checker.family(x, y) or Checker.family(y, x)):
                    continue
                code = x if len(x) <= len(y) else y
                if common and sum(1 for record in full.values() if code in render(record).lower()) > max(3, 0.4 * len(full)):
                    continue
                hits.append(f"{code} ({one}: '{texts[0][max(0, i - 40):i + len(x) + 40]}' / {two}: '{texts[1][max(0, j - 40):j + len(y) + 40]}')")
                if len(hits) >= 3:
                    return "; ".join(hits)
        return "; ".join(hits)

    def codeLines(self, lines: list[dict[str, Any]]) -> str:
        links = self.linkedNeeds()
        if not links:
            return ""
        held = {line["item"]: line["id"] for line in lines if line["item"]}
        out: list[str] = []
        for line in lines:
            for other, _ in links.get(line["item"], []):
                pair = f"{line['id']} and {held.get(other)}"
                shared = self.sharedCodes(line["id"], held.get(other, ""))
                if shared and not any(x.startswith(pair) for x in out):
                    out.append(pair + ": " + shared)
        if not out:
            return ""
        return ("\n\nCODES FOUND IN BOTH RECORDS (a string match made by code, shown for the requirements that name another item: such a requirement is met only when a record states the fit,"
                " the model or the condition with that code and no exception it names excludes the other line):\n" + "\n".join(out[:8]))

    def keepsKinds(self, old: Any, new: Any) -> bool:
        catKey = self.host.shelf.category_key()
        if not catKey:
            return True
        kinds = [{self.host.shelf.kind_of(i, catKey) for i in group} for group in (old, new)]
        return len(kinds[1]) >= len(kinds[0])

    def jointPools(self, group: list[int], held: dict[int, str], lines: dict[str, int]) -> dict[int, list[str]]:
        shelf = self.host.shelf
        catKey = shelf.category_key()
        labels = self.itemLabels()
        known = {Text.words(str(row.get(catKey) or "")) for row in shelf.rows.values()} if catKey else set()
        pools: dict[int, list[str]] = {}
        for number in group:
            line = {"id": held[number], "item": number, "label": str((shelf.rows.get(held[number]) or {}).get(catKey) or "").strip() if catKey else ""}
            pool = [i for i in shelf.seen_order if i in shelf.full and i not in lines and shelf.buyable(i) and shelf.price_of(i) is not None
                    and (self.sameItem(i, line, labels, known, catKey) or i in self.host.decline.lead_items.get(str(number), []))]
            pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
            pools[number] = [held[number]] + [i for i in dict.fromkeys(pool) if i not in pools.get(number, [])][:4]
        return pools

    def jointLines(self, lines: dict[str, int], kind: Any) -> dict[str, int] | None:
        host = self.host
        shelf = host.shelf
        tried = self.__dict__.setdefault("joint_tried", set())
        first = self.check_key(list(lines))
        note = self.notes.get(first) or {}
        if (first in tried or host.finished or host.budgets.closing() or host.store.brief.get("pick") in ("most", "least")
                or host.models.cost > host.budgets.spendHard * 1.3 or host.left() <= len(lines) + 6):
            return None
        links = self.linkedNeeds()
        if not links:
            return None
        tried.add(first)
        held = {number: ident for ident, number in (note.get("items") or {}).items() if ident in lines and number}
        group = sorted(set(links) | {m for pairs in links.values() for m, _ in pairs})
        if any(n not in held for n in group):
            return None
        pools = self.jointPools(group, held, lines)
        if all(len(pool) < 2 for pool in pools.values()):
            return None
        ledger, names = self.checkLedger()
        tags = {need: tag for tag, need in names.items()}
        room = 22000
        parts: list[str] = []
        for number in group:
            parts.append(f"item {number} ({host.book.items[number - 1].name}):")
            for ident in pools[number]:
                text = render(shelf.full.get(ident) or shelf.rows.get(ident), 240)[:1500]
                room -= len(text)
                if room < 0 and ident != held[number]:
                    continue
                parts.append(f"{ident} (price {shelf.price_of(ident) or 0:g}" + (", in the cart now" if ident == held[number] else "") + "): " + text)
        named: list[str] = []
        codes: list[str] = []
        for number, pairs in links.items():
            for other, need in pairs:
                named.append(f"{tags.get(need, need)} (item {number}) names item {other}: {need}")
                for one in pools[number]:
                    for two in pools[other]:
                        shared = self.sharedCodes(one, two)
                        if shared and len(codes) < 30:
                            codes.append(f"{one} and {two}: {shared}")
        prompt = (BASKET + "\n\nREQUEST:\n" + host.store.request + "\n\nREQUIREMENTS IN FORCE:\n" + ledger + "\n\nREQUIREMENTS THAT NAME ANOTHER ITEM:\n" + "\n".join(named)
                  + "\n\nCANDIDATES (cheapest first after the cart's line):\n" + "\n".join(parts))
        if codes:
            prompt += "\n\nCODES FOUND IN BOTH RECORDS (a string match made by code; a code counts only where a record states what it fits, works with or is sold as):\n" + "\n".join(codes)
        got = host.models.ask_json(prompt, [LIGHT], 5000, "medium")
        if not isinstance(got, dict) or not isinstance(got.get("own"), list):
            got = host.models.ask_json(prompt, [HEAVY], 3500, "medium")
        if not isinstance(got, dict):
            return None
        own: dict[str, bool] = {}
        for entry in Text.items(got.get("own")):
            grade = self.gradeParts(entry)
            if len(grade) >= 2 and grade[0] not in own:
                own[grade[0]] = grade[1].lower().startswith("fit")
        holds: dict[tuple[str, str], set[str]] = {}
        for entry in Text.items(got.get("pairs")):
            if isinstance(entry, dict):
                entry = [entry.get("id"), entry.get("requirement"), entry.get("with")]
            if isinstance(entry, list) and len(entry) >= 3:
                holds.setdefault((str(entry[0]).strip(), str(entry[1]).strip()), set()).update(str(x).strip() for x in Text.items(entry[2]))
        choices = {number: [i for i in pools[number] if own.get(i)] for number in group}
        plans: list[dict[int, str]] = [{}]
        for number in group:
            plans = [{**plan, number: i} for plan in plans for i in choices[number]][:5000]

        def fitting(plan: dict[int, str]) -> bool:
            return all(plan[other] in holds.get((plan[number], tags.get(need, "")), set()) for number, pairs in links.items() for other, need in pairs)

        cheapest = host.store.brief.get("pick") == "cheapest"
        best = None
        for plan in plans:
            if len(set(plan.values())) < len(plan) or not fitting(plan):
                continue
            trial = {(plan[n] if n in plan else i): q for i, q in lines.items() for n in [next((m for m in group if held[m] == i), 0)]}
            if not self.inBudget(trial) or not self.keepsKinds(lines, trial):
                continue
            total = sum((shelf.price_of(i) or 0.0) * q for i, q in trial.items())
            score = (total,) if cheapest else (sum(1 for i in trial if i not in lines), total)
            if best is None or score < best[0]:
                best = (score, trial)
        verdict = None
        if best and best[1] != lines:
            self.check_cart(list(best[1]), host.budgets.spendHard * 1.3, False)
            verdict = self.verdicts.get(self.check_key(list(best[1])))
        host.trace.append({"kind": "joint_try", "pools": {str(n): pools[n] for n in group}, "fit": [i for i in own if own[i]], "plan": best[1] if best else None, "ok": verdict})
        return best[1] if best and verdict and best[1] != lines else None

    def swapBetter(self) -> str:
        cart = self.host.store.roles.get("cart")
        lines = {i: self.host.shelf.cart_lines.get(i, 1) for i in self.host.shelf.cart_ids() if not self.host.shelf.sold_out(i)}
        key = self.check_key(list(lines))
        if not cart or not lines or self.verdicts.get(key) is not False or not ((self.notes.get(key) or {}).get("better") or self.linkedNeeds()):
            return ""
        catKey = self.host.shelf.category_key()

        def kind(ident: str) -> str:
            return str((self.host.shelf.rows.get(ident) or {}).get(catKey) or "").strip().lower() if catKey else ident

        swapped = self.betterLines(lines, kind, bool((self.notes.get(key) or {}).get("better")))
        if not swapped or self.host.finished:
            return ""
        calls = self.host.shelf.planCart(swapped, [i for i in lines if i not in swapped])
        if not calls or len(calls) + 2 > self.host.left():
            return ""
        done = self.host.shelf.setLines(calls)
        if done is None:
            return ""
        self.host.trace.append({"kind": "swap_better", "lines": swapped, "done": done})
        fresh = self.host.shelf.readNew([i for i in swapped if i not in lines])
        return self.swapText(lines, swapped, "a second check of the whole new cart found every requirement met") + (("\nAct read the new line(s) again: " + fresh) if fresh else "")

    def swapText(self, lines: dict[str, int], swapped: dict[str, int], why: str) -> str:
        gone = [i for i in lines if i not in swapped]
        came = [i for i in swapped if i not in lines]
        return ("Act changed the cart itself: " + ", ".join(came) + " now stands in for " + ", ".join(gone) + "; the check found the old line(s) failing or dearer and the new one(s) meeting every requirement, and "
                + why + ". The cart now holds " + ", ".join(f"{i} x{n}" for i, n in swapped.items()) + ". Check out with this cart as it is; do not put the old line back.")

    def swapPairs(self, lines: dict[str, int], kind: Any) -> list[tuple[str, str]]:
        note = self.notes.get(self.check_key(list(lines))) or {}
        catKey = self.host.shelf.category_key()
        flags = note.get("flags") or {}
        alt = note.get("alt") or {}
        pairs: list[tuple[str, str]] = []
        for pick in note.get("better") or []:
            bad = alt.get(pick)
            if bad not in lines:
                bad = next((i for i in lines if (i in flags or not flags) and (not catKey or kind(i) == kind(pick))), None)
            if bad and pick not in lines and not self.host.shelf.sold_out(pick) and (bad, pick) not in pairs:
                pairs.append((bad, pick))
        for bad in list(flags) or list(lines):
            number = next((n for n, ids in self.host.decline.lead_items.items() if bad in ids), "")
            for pick in self.host.decline.lead_items.get(number, []):
                if pick != bad and pick not in lines and not self.host.shelf.sold_out(pick) and (not catKey or kind(pick) == kind(bad)) and (bad, pick) not in pairs:
                    pairs.append((bad, pick))
        return pairs

    def betterLines(self, lines: dict[str, int], kind: Any, single: bool = True) -> dict[str, int] | None:
        first = self.check_key(list(lines))
        if self.verdicts.get(first) is not False or self.host.models.cost > self.host.budgets.spendHard * 1.3 or self.host.finished:
            return None
        base = dict(lines)
        done: set[str] = set()
        found = None
        for _ in range(3 if single else 0):
            pairs = [p for p in self.swapPairs(base, kind) if p[1] not in done and p[0] in base]
            if not pairs and base != lines:
                base = dict(lines)
                pairs = [p for p in self.swapPairs(base, kind) if p[1] not in done]
            if not pairs or self.host.finished:
                break
            bad, pick = pairs[0]
            done.add(pick)
            trial = {(pick if i == bad else i): n for i, n in base.items()}
            if not self.keepsKinds(base, trial):
                continue
            if pick not in self.host.shelf.full and self.host.left() > len(trial) + 4:
                self.host.shelf.openMore([pick], len(trial) + 3)
            self.check_cart(list(trial), self.host.budgets.spendHard * 1.3, False)
            verdict = self.verdicts.get(self.check_key(list(trial)))
            self.host.trace.append({"kind": "better_try", "bad": bad, "pick": pick, "ok": verdict})
            if verdict:
                found = trial
                break
            if verdict is None:
                break
            if [p for p in self.swapPairs(trial, kind) if p[1] not in done and p[0] != pick]:
                base = trial
        if found is None and not self.host.finished:
            found = self.jointLines(lines, kind)
        self.recall(first)
        return found

    def auditFits(self) -> dict[int, list[str]]:
        fits: dict[int, list[str]] = {}
        for entry in self.host.trace:
            if entry.get("kind") != "audit" or not isinstance(entry.get("got"), dict):
                continue
            for item in Text.items(entry["got"].get("items")):
                number = Text.count(item.get("item")) if isinstance(item, dict) else None
                named = [str(i) for key in ("fit", "best") for i in Text.items(item.get(key)) if str(i) in self.host.shelf.rows and not self.host.shelf.sold_out(str(i))] if number else []
                if named:
                    fits[number or 0] = named
        return fits

    def inBudget(self, lines: dict[str, int]) -> bool:
        budget = self.host.book.budget
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return True
        prices = [(self.host.shelf.price_of(i), n) for i, n in lines.items()]
        if any(p is None for p, _ in prices):
            return False
        if self.host.store.brief.get("budget_scope") == "each":
            return all((p or 0.0) * n <= budget for p, n in prices)
        return sum((p or 0.0) * n for p, n in prices) <= budget

    def leadSwap(self) -> str:
        host = self.host
        shelf = host.shelf
        lines = {i: shelf.cart_lines.get(i, 1) for i in shelf.cart_ids() if not shelf.sold_out(i)}
        key = self.check_key(list(lines))
        note = self.notes.get(key) or {}
        flags = note.get("flags") or {}
        if not host.store.roles.get("cart") or not lines or not flags or self.verdicts.get(key) is not False or host.finished or host.decline.audit_version != host.book.version:
            return ""
        fits = self.auditFits()
        cheapest = host.store.brief.get("pick") == "cheapest"
        plan = dict(lines)
        moved: list[tuple[str, str]] = []
        for bad in sorted(flags, key=lambda i: not flags[i]):
            number = (note.get("items") or {}).get(bad)
            named = fits.get(number, []) if number else []
            if bad not in plan or not named or bad in named:
                continue
            price = shelf.price_of(bad)
            leads = [i for i in named if i not in plan and shelf.buyable(i) and (flags[bad] or not cheapest or (price is not None and shelf.price_of(i) is not None and (shelf.price_of(i) or 0.0) < price))]
            if cheapest:
                leads.sort(key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))
            for pick in leads:
                trial = {(pick if i == bad else i): n for i, n in plan.items()}
                if self.inBudget(trial) and self.keepsKinds(plan, trial):
                    plan = trial
                    moved.append((bad, pick))
                    break
        if not moved:
            return ""
        calls = shelf.planCart(plan, [i for i in lines if i not in plan])
        if not calls or len(calls) + 2 > host.left():
            return ""
        done = shelf.setLines(calls)
        if done is None:
            return ""
        host.trace.append({"kind": "lead_swap", "moved": moved, "done": done})
        fresh = shelf.readNew([i for i in plan if i not in lines])
        return self.swapText(lines, plan, "the review of every listing named the new one(s) as meeting every requirement") + (("\nAct read the new line(s) again: " + fresh) if fresh else "")

class Decline:
    loose: frozenset[str] = frozenset()

    def __init__(self, host: Any):
        self.host = host
        self.audit_runs = 0
        self.audit_seen = -1
        self.audit_version = -1
        self.decline_proven = False
        self.force_buy = False
        self.decline_reads = 0
        self.audited = False
        self.rowsHeld = False
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.confirmed: str | None = None
        self.completed = False

    def audit(self, near: bool = False, late: bool = False) -> str:
        if self.audit_runs >= 2 or (self.audit_runs and len(self.host.shelf.full) == self.audit_seen and self.host.book.version == self.audit_version) or self.host.models.cost > Budgets.auditRoom(self.host.budgets.spendHard, near or late, len(self.host.book.items), Budgets.held(self.host)):
            return ""
        self.audit_runs += 1
        self.audited = True
        got = self.audit_call()
        near = near or bool((got or {}).get("fallback") is True)
        named = self.audit_fits(got, near)
        fresh: list[str] = []
        items = self.host.book.items
        lacking = [n for n in range(1, len(items) + 1) if str(n) not in named]
        if lacking and self.host.left() > len(items) + 3:
            fresh, _ = self.host.coverage.browsePass(lacking, min(self.host.budgets.coverPages, (self.host.left() - self.host.book.tail() - len(items) - len(lacking) - 1) // 2))
        for _ in range(2):
            unread = [i for ids in named.values() for i in ids if i not in self.host.shelf.full]
            short = [str(n) for n in range(1, len(items) + 1) if str(n) not in named]
            if not (short or unread) or self.host.left() <= len(items) + 3 or self.host.models.cost > Budgets.auditRoom(self.host.budgets.spendHard, near or late, len(self.host.book.items), Budgets.held(self.host)):
                break
            found = [i for i in fresh if i not in unread and i not in self.host.shelf.full]
            rows = self.auditRows(got, short, unread + (self.rowMix(got, short, found) if found else [])[:6])[: max(2, (self.host.left() - len(items) - 3) // 3)]
            fresh = []
            if not rows or not self.host.shelf.openRows(rows):
                break
            focus = sorted({int(n) for n in short} | {int(n) for n, ids in named.items() if any(i not in self.host.shelf.full for i in ids)})
            got = self.audit_call(focus, named)
            near = near or bool((got or {}).get("fallback") is True)
            regraded = self.audit_fits(got, near)
            named = {**{k: v for k, v in named.items() if int(k) not in focus}, **regraded}
        self.audit_seen = len(self.host.shelf.full)
        self.audit_version = self.host.book.version
        remaining = [str(n) for n in range(1, len(self.host.book.items) + 1) if str(n) not in named and self.unread_rows([n], 1)]
        self.decline_proven = bool((got or {}).get("proven") is True and not named and not remaining)
        if not named:
            if remaining and not self.rowsHeld and self.host.left() > len(self.host.book.items) + 5:
                self.rowsHeld = True
                rows = self.unread_rows([int(n) for n in remaining], 2)
                return ("Listings of item(s) " + ", ".join(remaining) + " are still unread: " + ", ".join(rows[:6])
                        + ". Open them and judge them against every requirement; end without a purchase only when each listing of some item is contradicted by its own record.")
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        missing = [str(n) for n in range(1, len(self.host.book.items) + 1) if str(n) not in named]
        text = "A careful review found listings with no requirement contradicted by their own records: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
        if got and got.get("reason"):
            text += f" ({got.get('reason')})"
        if missing:
            text += f". Item(s) {', '.join(missing)} still need a candidate: search their category again with other words and open what might fit"
        return text + ". Open them, put the best one per item in the cart and check out; a requirement their records do not mention is unknown, not failed, and ending without a purchase is right only when every listing of an item is contradicted by its own record."

    def decline_hold(self) -> str:
        if self.decline_reads < 2 and self.host.left() > 10:
            self.decline_reads += 1
            opened = self.read_candidates(3, min(6, self.host.left() - 8))
            if opened:
                return "Before ending without a purchase, Act opened more listings of each requested item: " + ", ".join(opened) + ". Check them against every requirement; buy what fits."
        live = {need for _, need in self.host.book.needs_now()}
        pending = [need for _, need in self.host.lookups.refs if need in live and need not in self.host.lookups.ref_values]
        if pending and self.host.left() > 8:
            chain, hold = self.host.lookups.trail(3)
            pending = [need for _, need in self.host.lookups.refs if need in live and need not in self.host.lookups.ref_values]
            if (hold or pending) and self.host.lookups.ref_holds < 3 and self.host.left() > 8:
                self.host.lookups.ref_holds += 1
                return (chain + "A requirement still takes its value from another listing that has not been read: " + " | ".join(pending[:3])
                        + ". An unread reference never shows that nothing fits: find that listing, open its page and every page it points to, then judge the candidates with that value.")
            if chain:
                return chain + "Judge each candidate again with that value; buy the one that fits, and end without a purchase only if every candidate fails a requirement by its own record."
        return ""

    def together(self) -> bool:
        return bool(self.host.store.compare_target()) and not self.host.store.compare_bare()

    def read_candidates(self, per_item: int, room: int) -> list[str]:
        host = self.host
        shelf, store = host.shelf, host.store
        if room <= 0:
            return []
        talk = Interviewer(host)
        cat_key = shelf.category_key()
        wanted: list[str] = []
        for number in range(1, len(host.book.items) + 1):
            if str(number) in self.lead_items:
                continue
            label = Text.words(talk.itemLabel(number))
            ids = [i for i in shelf.seen_order if (not cat_key or Text.words(str((shelf.rows.get(i) or {}).get(cat_key) or "")) == label)]
            have = sum(1 for i in ids if i in shelf.full)
            pool = host.coverage.bestRows([i for i in ids if i not in shelf.full and i not in shelf.opened and shelf.buyable(i)])
            wanted += pool[: max(0, per_item - have) + 1]
        wanted = wanted[: max(2, room * max(1, store.compare_room()))]
        before = set(shelf.full)
        done: list[str] = []
        while wanted and room > 0 and not host.finished:
            chunk = wanted[: max(1, store.compare_room())] if self.together() else wanted[:2]
            wanted = wanted[len(chunk):]
            if not shelf.openRows(chunk):
                break
            room -= 1 if self.together() else len(chunk)
            done += [i for i in chunk if i in shelf.full and i not in before]
        return done

    def unread_rows(self, numbers: list[int], per_item: int) -> list[str]:
        shelf = self.host.shelf
        out: list[str] = []
        for number in numbers:
            pool = self.host.coverage.bestRows([i for i in shelf.seen_order if i not in shelf.full and i not in shelf.opened and shelf.buyable(i) and shelf.item_of(i) == number])
            out += pool[:per_item]
        return out

    def auditRows(self, got: dict[str, Any] | None, short: list[str], unread: list[str]) -> list[str]:
        suggested = [str(i) for i in Text.items((got or {}).get("open")) if str(i) in self.host.shelf.rows and str(i) not in self.host.shelf.full and not self.host.shelf.sold_out(str(i))]
        rows = list(unread)
        numbers = [int(n) for n in short]
        if not self.host.shelf.category_key():
            return (rows + [i for i in suggested if i not in rows])[:8]
        per = max(2, min(4, (8 - len(rows)) // max(1, len(numbers))))
        for number in numbers:
            mine = [i for i in suggested if self.host.shelf.item_of(i) == number]
            mine += [i for i in self.unread_rows([number], per) if i not in mine]
            rows += [i for i in mine[:per] if i not in rows]
        rows += [i for i in suggested if i not in rows and self.host.shelf.item_of(i) == 0][:2]
        return rows[:8]

    def rowMix(self, got: dict[str, Any] | None, short: list[str], found: list[str]) -> list[str]:
        shelf = self.host.shelf
        keyed = bool(shelf.category_key())
        wanted = {int(n) for n in short}
        suggested = [str(i) for i in Text.items((got or {}).get("open")) if str(i) in shelf.rows and str(i) not in shelf.full and not shelf.sold_out(str(i))
                     and (not keyed or shelf.item_of(str(i)) in wanted | {0})]
        mixed: list[str] = []
        for place in range(max(len(suggested), len(found))):
            for ident in suggested[place:place + 1] + found[place:place + 1]:
                if ident not in mixed:
                    mixed.append(ident)
        return mixed

    def audit_call(self, focus: list[int] | None = None, known: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
        records: list[str] = []
        groups: dict[int, list[str]] = {n: [] for n in range(0, len(self.host.book.items) + 1)}
        for ident in reversed(self.host.shelf.seen_order):
            if not self.host.shelf.sold_out(ident):
                groups[self.host.shelf.item_of(ident)].append(ident)

        def line(ident: str) -> str:
            record = self.host.shelf.full.get(ident) if ident in self.host.shelf.full else None
            return ("OPENED " + render(record, 200)[:1100]) if record else ("ROW " + render(self.host.shelf.rows.get(ident), 120)[:400])

        for number in range(1, len(self.host.book.items) + 1):
            if focus is not None and number not in focus:
                leads = (known or {}).get(str(number)) or []
                records.append(f"ITEM {number} ({self.host.book.items[number - 1].name}): judged in an earlier review, listings found to fit: " + (", ".join(leads) if leads else "none") + "; grade it again only if a record below contradicts that.")
                continue
            opened = [i for i in groups[number] if i in self.host.shelf.full]
            rows = [i for i in groups[number] if i not in self.host.shelf.full]
            if self.host.store.brief.get("pick") == "cheapest":
                rows.sort(key=lambda i: self.host.shelf.price_of(i) if self.host.shelf.price_of(i) is not None else float("inf"))
            picked = opened[:6] + rows[:4]
            if picked:
                records.append(f"ITEM {number} ({self.host.book.items[number - 1].name}):")
                records += [line(i) for i in picked]
        if not self.host.shelf.category_key() or any(not groups[n] for n in range(1, len(self.host.book.items) + 1)):
            for ident in groups[0][: max(0, 36 - len(records))]:
                records.append(line(ident))
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.host.store.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.host.book.render()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.host.models.ask_json(prompt, [HEAVY, LIGHT], 4000, "low")
        self.host.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None, near: bool = False) -> dict[str, list[str]]:
        named: dict[str, list[str]] = {}
        for entry in Text.items((got or {}).get("items")):
            if not isinstance(entry, dict):
                continue
            number = Text.count(entry.get("item"))
            if not number or not 1 <= number <= len(self.host.book.items):
                continue
            keys = ["fit", "best"] + (["near"] if near else [])
            ids: list[str] = []
            for key in keys:
                ids += [str(i) for i in Text.items(entry.get(key)) if str(i) in self.host.shelf.rows and not self.host.shelf.sold_out(str(i)) and str(i) not in ids]
            ids = ids[:4]
            if ids:
                named[str(number)] = ids
        fits = (got or {}).get("fits")
        if not named and isinstance(fits, dict):
            for k, v in fits.items():
                ids = [str(i) for i in Text.items(v) if str(i) in self.host.shelf.rows and not self.host.shelf.sold_out(str(i))][:4]
                if ids:
                    named[str(k)] = ids
        firm = {str(i) for entry in Text.items((got or {}).get("items")) if isinstance(entry, dict) for key in ("fit", "best") for i in Text.items(entry.get(key))}
        firm |= {str(i) for given in (fits.values() if isinstance(fits, dict) else []) for i in Text.items(given)}
        self.loose = (self.loose - set(named)) | {number for number, ids in named.items() if not firm & set(ids)}
        return named

    def pastBudget(self, lines: dict[str, int]) -> bool:
        budget = self.host.book.budget
        if not lines or not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return False
        prices = [self.host.shelf.price_of(i) for i in lines]
        if any(price is None for price in prices):
            return False
        spent = [float(price or 0.0) * count for price, count in zip(prices, lines.values())]
        return max(spent) > budget if self.host.store.brief.get("budget_scope") == "each" else sum(spent) > budget

    def buy_leads(self) -> str:
        if self.decline_proven or not self.lead_items or self.completed:
            return ""
        held = {self.host.shelf.item_of(i) for i in self.host.shelf.cart_ids() if not self.host.shelf.sold_out(i)}
        if not all(str(n) in self.lead_items or n in held for n in range(1, len(self.host.book.items) + 1)):
            return ""
        first = next((i for n in range(1, len(self.host.book.items) + 1) for i in self.lead_items.get(str(n), []) if not self.host.shelf.sold_out(i)), None)
        if not first:
            return ""
        self.confirmed = self.confirmed if self.confirmed and not self.host.shelf.sold_out(self.confirmed) else first
        self.force_buy = True
        self.host.trace.append({"kind": "buy_leads", "leads": self.lead_items})
        return self.complete_cart()

    def complete_cart(self) -> str:
        cart, buy = self.host.store.roles.get("cart"), self.host.store.roles.get("buy")
        if self.completed or not self.confirmed or self.host.shelf.sold_out(self.confirmed) or not cart or not buy:
            return ""
        loose = sorted(n for n in self.lead_items if n in self.loose)
        if self.force_buy and loose:
            self.host.trace.append({"kind": "lead_hold", "why": "near", "items": loose})
            return ""
        self.completed = True
        cat_key = self.host.shelf.category_key()

        def kind(ident: str) -> str:
            return str((self.host.shelf.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = Text.count(number)
            pick = next((i for i in ids if not self.host.shelf.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.host.book.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.host.book.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.host.shelf.cart_ids()
        keep = {i: self.host.shelf.cart_lines.get(i, 1) for i in now if i not in plan and not self.host.shelf.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.host.book.items):
            return ""
        lines = {**keep, **plan}
        if self.pastBudget(lines):
            self.completed = False
            self.host.trace.append({"kind": "lead_hold", "why": "budget", "lines": lines})
            return ""
        calls = self.host.shelf.planCart(lines, drop)
        if not calls or len(calls) + 1 > self.host.left():
            return ""

        if self.host.left() > len(calls) + 4:
            self.host.shelf.openMore(list(lines), len(calls) + 3)
        self.host.checker.check_cart(list(lines), self.host.budgets.spendHard * 1.3)
        swapped = self.host.checker.betterLines(lines, kind)
        if swapped:
            drop = drop + [i for i in lines if i not in swapped and i in now]
            lines = swapped
            self.confirmed = next((i for i in lines if i in self.leads), next(iter(lines)))
            calls = self.host.shelf.planCart(lines, drop)
            if not calls or len(calls) + 1 > self.host.left():
                return ""
        if not self.host.checker.verdicts.get(self.host.checker.check_key(list(lines))) and not self.force_buy:
            return ""
        if self.pastBudget(lines):
            self.host.trace.append({"kind": "lead_hold", "why": "budget", "lines": lines})
            return ""
        done = self.host.shelf.setLines(calls)
        if done is None:
            return ""
        faults = []
        args = StoreMap.shape(dict(self.host.store.brief.get("buy_args") or {}), self.host.store.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []

        for attempt in range(2):
            if attempt:
                self.host.shelf.reread_offers()
                self.host.checker.check_cart(list(lines), self.host.budgets.spendHard * 1.3)
                swapped = self.host.checker.betterLines(lines, kind)
                if swapped and self.host.left() > 2:
                    more = self.host.shelf.planCart(swapped, [i for i in lines if i not in swapped])
                    if more and len(more) + 1 <= self.host.left():
                        extra = self.host.shelf.setLines(more)
                        if extra is None:
                            return ""
                        done += extra
                        lines = swapped
                if ((not self.host.checker.verdicts.get(self.host.checker.check_key(list(lines))) and not self.force_buy) or self.host.left() < 1 or self.host.finished
                        or any(self.host.shelf.sold_out(i) for i in lines) or self.pastBudget(lines)):
                    break
            go, said = self.host.procedures.run("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.host.act(buy, args)
            self.host.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": Text.clip(reply, 200)})
            if reply is None:
                break
            self.host.buys += 1
            if not reply.get("finished"):
                self.host.shelf.refused_at = self.host.step
            self.host.book.observe(buy, reply.get("observation"))
            replies.append(self.host.show(reply.get("observation")))
            if self.host.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Act acted instead): the independent check confirmed " + self.confirmed + ", so Act set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

class Interviewer:
    def __init__(self, host: Any):
        self.host = host
        self.tried: dict[int, set[str]] = {}
        self.revealed: dict[int, int] = {}
        self.tabled: dict[int, bool] = {}

    def itemLabel(self, number: int) -> str:
        host = self.host
        item = host.book.items[number - 1]
        catKey = host.shelf.category_key()
        if not catKey:
            return item.shelf or item.name
        sides = [side for side in (Text.stem(item.shelf or ""), Text.stem(item.name)) if side]
        wanted = frozenset().union(*sides) if sides else frozenset()
        longWanted = {w for w in wanted if len(w) >= 4}

        def alike(a: str, b: str) -> bool:
            return a == b or (len(a) >= 4 and len(b) >= 4 and abs(len(a) - len(b)) <= 2 and (a.startswith(b) or b.startswith(a)))

        def covers(have: frozenset[str], side: frozenset[str]) -> bool:
            return all(any(alike(a, b) for b in side) for a in have) and all(any(alike(a, b) for a in have) for b in side)

        best = None
        counts: dict[str, int] = {}
        for ident in host.shelf.seen_order:
            label = (host.shelf.rows.get(ident) or {}).get(catKey)
            if isinstance(label, str) and label.strip():
                counts[label] = counts.get(label, 0) + 1
        for label, seen in counts.items():
            have = Text.stem(label)
            exact = any(covers(have, side) for side in sides)
            overlap = sum(1 for w in have if len(w) >= 4 and any(alike(w, x) for x in longWanted))
            if not exact and not overlap:
                continue
            extra = sum(1 for w in have if not any(alike(w, x) for x in wanted))
            score = (exact, overlap, -extra, seen)
            if best is None or score > best[0]:
                best = (score, label)
        return best[1] if best else (item.shelf or item.name)

    def keyStats(self, number: int) -> list[tuple[str, int, int, list[str]]]:
        host = self.host
        label = self.itemLabel(number)
        catKey = host.shelf.category_key()
        stats: dict[str, dict[str, Any]] = {}
        listings = 0
        for ident in host.shelf.seen_order:
            record = host.shelf.full.get(ident)
            if not record:
                continue
            if catKey and Text.words(str(record.get(catKey) or "")) != Text.words(label):
                continue
            block = Shelf.factBlock(record) or {}
            if not block:
                continue
            listings += 1
            done: set[str] = set()
            for key, value in block.items():
                name = Text.words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = stats.setdefault(name, {"n": 0, "values": [], "per": []})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"].append(text)
                if text and text not in slot["values"]:
                    slot["values"].append(text)
        table = []
        ranked = []
        for name, slot in stats.items():
            values = slot["values"]
            per = slot["per"]
            if not values or not per:
                continue
            mean = sum(len(v) for v in per) / len(per)
            if mean > 40:
                continue
            listy = sum(1 for v in per if len([p for p in re.split(r"[,;/|]", v) if p.strip()]) >= 3)
            if listy * 2 > len(per):
                continue
            if len(per) >= 4 and len(set(per)) == len(per) and all(Shelf.idLike(v) for v in per):
                continue
            top = max(per.count(v) for v in set(per)) / len(per)
            weight = (slot["n"] / max(1, listings)) * (0.65 + 0.35 * (1.0 - top))
            ranked.append((weight, name, slot["n"], len(values), [v[:40] for v in values[:4]]))
        merged: list[tuple[float, str, int, int, list[str]]] = []
        for row in sorted(ranked, key=lambda r: (-r[0], -r[2], r[1])):
            if any(Text.names(row[1], kept[1]) and set(row[4]) & set(kept[4]) for kept in merged):
                continue
            merged.append(row)
        for weight, name, carried, distinct, values in merged[:18]:
            table.append((name, carried, distinct, values))
        return table

    def plan(self, count: int) -> dict[int, list[str]]:
        host = self.host
        blocks = []
        tables: dict[int, list[tuple[str, int, int, list[str]]]] = {}
        for number, item in enumerate(host.book.items, 1):
            tables[number] = self.keyStats(number)
            settled = "; ".join(m for m in item.musts + item.told if m not in item.dropped) or "nothing"
            lines = [f"item {number}: {item.name} (store category: {self.itemLabel(number)}) - settled by the request: {settled}"]
            if tables[number]:
                for name, carried, distinct, values in tables[number]:
                    lines.append(f"  - {name}: on {carried} listings, {distinct} values: " + " | ".join(values))
            else:
                lines.append("  (no opened listing of this item yet)")
            blocks.append("\n".join(lines))
        picks = max(4, count + 2)
        prompt = ASK_PLAN.replace("PICKS", str(picks)) + "\n\nREQUEST:\n" + host.store.request + "\n\nITEMS AND THEIR KEYS:\n" + "\n".join(blocks)
        first = [str(x) for x in Text.items(host.store.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        got = host.models.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ranked: dict[int, list[str]] = {}
        for entry in Text.items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = Text.count(entry.get("item"))
            if not number or not 1 <= number <= len(host.book.items):
                continue
            known = {name for name, _, _, _ in tables[number]}
            item = host.book.items[number - 1]
            settled = [Text.stem(m.split(":")[0]) for m in item.musts + item.told if ":" in m and m not in item.dropped]
            keep: list[str] = []
            for key in Text.items(entry.get("ask")):
                name = Text.words(re.split(r"[/,;(]", str(key))[0]).strip().rstrip("?.")
                if not name or len(name.split()) > 5 or name in keep:
                    continue
                if known and name not in known and not any(Text.names(name, k) for k in known):
                    continue
                if any(Text.names(name, done) for done in settled):
                    continue
                keep.append(name)
            ranked[number] = keep
        for number in range(1, len(host.book.items) + 1):
            item = host.book.items[number - 1]
            settled = [Text.stem(m.split(":")[0]) for m in item.musts + item.told if ":" in m and m not in item.dropped]
            queue = list(ranked.get(number) or [])
            for name, carried, distinct, _ in tables[number]:
                if name in queue or any(Text.stem(name) == Text.stem(k) for k in queue) or any(Text.names(name, done) for done in settled):
                    continue
                queue.append(name)
            ranked[number] = queue
            self.tabled[number] = bool(tables[number])
        host.trace.append({"kind": "ask_plan", "ranked": {n: list(v) for n, v in ranked.items()}, "tables": {n: len(t) for n, t in tables.items()}})
        return ranked

    def nextPick(self, ranked: dict[int, list[str]]) -> tuple[int, str] | None:
        order = sorted(ranked, key=lambda n: (self.revealed.get(n, 0), 0 if self.tabled.get(n, True) else 1, len(self.tried.get(n, set())), n))
        for number in order:
            queue = ranked[number]
            while queue:
                name = queue.pop(0)
                if name not in self.tried.setdefault(number, set()):
                    return number, name
        return None

    def question(self, number: int, name: str) -> str:
        return f"What {name} would you like for the {Text.words(self.itemLabel(number))}?"

    def run(self) -> str:
        host = self.host
        count = host.book.msgs_free()
        if count <= 0:
            return ""
        ranked = self.plan(count)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while host.book.msgs_free() > 0 and not host.finished:
            pick = self.nextPick(ranked)
            if pick is None:
                break
            number, name = pick
            self.tried.setdefault(number, set()).add(name)
            batch = host.book.send([(number, name, self.question(number, name))])
            if not batch:
                break
            sent += batch
            before = len(told)
            heard = host.book.hear(batch, (LIGHT, HEAVY))
            host.book.take_heard(heard, told)
            if len(told) > before:
                self.revealed[number] = self.revealed.get(number, 0) + 1
        if told:
            host.book.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "The interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in host.lookups.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        live = {need for _, need in host.book.needs_now()}
        if any(need in live and need not in host.lookups.ref_values for _, need in host.lookups.refs) and host.left() > 12 and not host.finished:
            chain, _ = host.lookups.trail(2)
            if chain:
                summary += "\n" + chain
        if host.book.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(host.book.queued)
            host.book.queued = []
        return summary

class Pending:
    def __init__(self, name: Any, raw: Any):
        self.name = name
        self.raw = raw
        self.args: dict[str, Any] = {}
        self.trimmed = ""
        self.many: str | None = None
        self.target: str | None = None
        self.ending = False
        self.mark = ""
        self.readers: set[str] = set()
        self.plain: set[str] = set()

class EndingGuard:
    def __init__(self, host: Any):
        self.host = host
        self.held = 0
        self.late_held = False
        self.refuse_blocks = 0
        self.pushed = False

    def hold(self, job: Pending) -> str | None:
        host = self.host
        if job.ending and host.store.is_refuse(job.name, job.args) and Budgets.closeOut(host.models.cost, len(host.book.items), Budgets.held(host)):
            host.trace.append({"kind": "refuse_closeout", "cost": round(host.models.cost, 4), "left": host.left()})
            job.args = host.store.quiet(job.name, job.args)
            return None
        if job.ending and host.left() > 2:
            text = self.buyHold() if host.store.is_buy(job.name, job.args) else self.refuseHold()
            if text is not None:
                return text
        if host.store.is_refuse(job.name, job.args):
            return self.refuseLast(job)
        return None

    def viewCart(self) -> str | None:
        host = self.host
        store, shelf = host.store, host.shelf
        if shelf.cart_touched and shelf.cart_seen_at < shelf.cart_at and store.roles.get("cart_view") and host.left() > 3:
            faults: list[str] = []
            view_args = StoreMap.shape({}, store.schemas.get(store.roles["cart_view"]), "args", faults)
            view = host.act(store.roles["cart_view"], view_args) if not faults else None
            if view is not None:
                shelf.cart_view = view.get("observation")
                shelf.cart_seen_at = host.step
                change = host.book.observe(store.roles["cart_view"], shelf.cart_view)
                if change:
                    host.book.queued = []
                    return "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + shelf.cart_text() + "\nApply the change, fix the cart if needed, then check out."
        return None

    def buyHold(self) -> str | None:
        host = self.host
        shelf = host.shelf
        wider = host.coverage.wider_hold()
        if wider:
            return wider + "\nCart now: " + shelf.cart_text()
        reread = shelf.reread_offers()
        changed = self.viewCart()
        if changed is not None:
            return changed
        swap = shelf.swap_hint()
        held = [i for i in shelf.cart_ids() if i in shelf.gone] if self.held < 2 else []
        if swap or held:
            self.held += 0 if swap else 1
            return "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + shelf.cart_text()
        chain, hold = host.lookups.trail()
        if hold:
            return "Not run yet (no step used). " + chain + "\nCart now: " + shelf.cart_text()
        read = shelf.openMore(shelf.cart_ids(), host.book.tail() + 3) + host.coverage.cover_pages() + host.coverage.cheaper_rows()
        verdict = host.checker.check_cart()
        if verdict and (host.checker.better or host.checker.linkedNeeds()) and host.left() > len(shelf.cart_ids()) + 5:
            version = host.book.version
            swapped = host.checker.swapBetter()
            if swapped:
                if host.book.version != version:
                    return "Not run yet (no step used). " + swapped + "\nCart now: " + shelf.cart_text()
                host.book.queued.append(swapped)
                return None
        read += host.coverage.flaggedRows(verdict)
        if verdict:
            seen = ("Act read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
            return "Not run yet. " + chain + seen + verdict + "\nCart now: " + shelf.cart_text()
        if reread and host.models.cost > host.budgets.spendSoft:
            return "Not run yet (no step used). Act re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."
        return None

    def refuseHold(self) -> str | None:
        host = self.host
        decline = host.decline
        if host.left() <= 5:
            return None
        hold = decline.decline_hold()
        if hold:
            return "Not run yet (no step used). " + hold
        review = decline.audit()
        if review:
            return "Not run yet (no step used). " + review
        if not decline.leads or host.left() <= 4:
            return None
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            host.trace.append({"kind": "refuse_blocked", "leads": list(decline.leads)})
            return ("Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(decline.leads)
                    + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.")
        if self.refuse_blocks == 1:
            self.refuse_blocks = 2
            first = decline.leads[0]
            host.checker.check_cart([first], host.budgets.spendHard)
            passed = bool(host.checker.verdicts.get(host.checker.check_key([first])))
            covered = all(str(n) in decline.lead_items for n in range(1, len(host.book.items) + 1))
            if first in decline.leads and (passed or (covered and not decline.decline_proven)):
                decline.confirmed = first
                decline.force_buy = not passed
                host.trace.append({"kind": "refuse_blocked", "confirmed": first})
                said = "says it meets every requirement" if passed else "found no requirement its record contradicts, and no listing of any item is proven to fail"
                return f"Not run yet (no step used). An independent check of {first}'s records {said}. Put the best listing for each item in the cart at the requested quantity and check out; ending without a purchase is right only when every listing of some item is contradicted by its own record."
        return None

    def lateHold(self) -> str | None:
        host = self.host
        shelf = host.shelf
        lines = [i for i in shelf.built if not shelf.sold_out(i) and i not in shelf.dropped]
        fresh = shelf.cart_view is not None and shelf.cart_seen_at >= shelf.cart_at
        missing = [i for i in lines if i not in shelf.cart_ids() or (i in shelf.dropped and not fresh)]
        if len(lines) >= len(host.book.items) and len(missing) + 1 <= host.left():
            self.late_held = True
            host.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
            redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
            return ("Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                    + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.")
        return None

    def refuseLast(self, job: Pending) -> str | None:
        host = self.host
        decline = host.decline
        if not decline.audit_runs and host.left() > 2:
            decline.audit()
        if not decline.decline_proven and decline.lead_items:
            done = decline.buy_leads()
            if done:
                return done + (" [episode finished]" if host.finished else "")
        if host.left() <= 5 and not self.late_held:
            text = self.lateHold()
            if text is not None:
                return text
        if not decline.leads and not self.pushed and host.left() > host.max_steps // 3:
            self.pushed = True
            return f"Not run yet (no step used). {host.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        done = decline.complete_cart()
        if done:
            return done + (" [episode finished]" if host.finished else "")
        job.args = host.store.quiet(job.name, job.args)
        return None

class Procedures:
    def __init__(self, host: Any):
        self.host = host

    def toolLines(self, ending: set[str]) -> list[str]:
        return [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {render(t['function'].get('parameters') or {})}"
                for t in self.host.store.tools if t["function"]["name"] not in ending]

    def prompt(self, stage: str, steps: list[dict[str, Any]], tools: list[str]) -> str:
        host = self.host
        book = host.book
        record = "\n".join(f"{i}. {line}" for i, line in enumerate(book.done_log, 1)) or "(nothing yet)"
        text = GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
        text += "\n\nREQUEST:\n" + host.store.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
        text += "\n\nLEDGER:\n" + book.render() + "\n\nCART NOW: " + host.shelf.cart_text()
        text += "\n\nRECORD (messages and step calls so far):\n" + record
        if book.notes:
            text += "\n\nSTORE NOTES SEEN:\n" + "\n".join(book.notes[-4:])
        text += f"\n\nMESSAGES LEFT: {max(0, book.msg_limit - book.msg_used)}   STEPS LEFT: {host.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
        return text + "\n\nTOOLS:\n" + "\n".join(tools)

    def unproven(self, got: dict[str, Any], stage: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        book = self.host.book
        shown = [e for e in Text.items(got.get("steps")) if isinstance(e, dict)]
        talk = stage == "order" and any(p["messages"] > 0 for p in steps)
        proof = {i for i in range(1, len(book.done_log) + 1) if not talk or (i <= len(book.done_steps) and book.done_steps[i - 1] > self.host.shelf.cart_at)}
        return [e for e in shown if e.get("done") is not True or not any(Text.count(n, 0) in proof for n in Text.items(e.get("lines")))]

    def nudge(self, unproven: list[dict[str, Any]]) -> None:
        host = self.host
        book = host.book
        host.trace.append({"kind": "gate_unproven", "steps": Text.clip(unproven, 120)})
        book.done_log.append(f"(Act: no line made since the cart last changed at step {host.shelf.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                             + f" done, and {book.msg_limit - book.msg_used} message(s) and {host.left()} steps are left: give the call that does it)")
        book.done_steps.append(-1)

    def perform(self, stage: str, call: dict[str, Any] | None, ending: set[str], said: list[str]) -> bool:
        host = self.host
        book, store = host.book, host.store
        name = str((call or {}).get("tool") or "")
        if not call or name not in store.schemas or name in ending:
            if not call:
                book.gate_ok.add((stage, host.shelf.cart_mark() + f"#{book.version}"))
            return False
        if host.left() < 2 or (name == store.roles.get("ask") and book.msg_used >= book.msg_limit):
            return False
        faults: list[str] = []
        args = StoreMap.shape(call.get("args") if isinstance(call.get("args"), dict) else {}, store.schemas[name], "args", faults)
        if faults:
            return False
        if name == store.roles.get("ask"):
            text = next((v for v in args.values() if isinstance(v, str)), "")
            batch = book.send([(1, "", text)], 1)
            if not batch:
                return False
            book.gate_msgs += 1
            said.append(f"Q: {text}\nA: {batch[0][2]}")
            return True
        reply = host.act(name, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        host.shelf.absorb(seen)
        book.done_log.append(f"step {host.step}: {name} {render(args)} -> {host.show(seen)[:300]} (cart then: {host.shelf.cart_mark()})")
        book.done_steps.append(host.step)
        said.append(f"{name} {render(args)} -> {host.show(seen)[:400]}")
        return True

    def done(self, said: list[str]) -> str:
        return ("Before going on, Act did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def run(self, stage: str) -> tuple[bool, str]:
        host = self.host
        book, store = host.book, host.store
        steps = [p for p in book.process if p["when"] == stage]
        if not steps or host.finished or not book.gate_due(stage):
            return True, ""
        ending = {store.roles.get("buy"), store.roles.get("refuse")} - {None}
        tools = self.toolLines(ending)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            got = host.models.ask_json(self.prompt(stage, steps, tools), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                book.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            host.trace.append({"kind": "gate", "stage": stage, "call": Text.clip(call, 200), "steps": Text.clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                book.version += 1
                return False, (self.done(said) + "\n" if said else "") + "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again."
            unproven = self.unproven(got, stage, steps)
            if not call and unproven and not said and not nudged and store.roles.get("ask") and book.msg_used < book.msg_limit and host.left() >= 2:
                nudged = True
                self.nudge(unproven)
                continue
            if not self.perform(stage, call, ending, said):
                break
        return True, self.done(said)

class Act:
    def __init__(self, problem: dict[str, Any]):
        self.store = StoreMap(self, problem)
        steps = Text.count(self.store.view.get("max_steps"))
        if steps is None and isinstance(self.store.view.get("max_steps"), float):
            steps = int(self.store.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 30
        self.budgets = Budgets(self.max_steps, 1, str(self.store.binding))
        self.step = 0
        self.finished = bool(self.store.view.get("finished"))
        self.broke = False
        self.misses = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.asked = False
        self.buys = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.models = ModelGate(self)
        self.shelf = Shelf(self)
        self.book = Ledger(self)
        self.lookups = Lookups(self)
        self.coverage = Coverage(self)
        self.checker = Checker(self)
        self.decline = Decline(self)
        self.guard = EndingGuard(self)
        self.procedures = Procedures(self)

    def left(self) -> int:
        return self.max_steps - self.step

    def stepsPerTurn(self) -> float:
        return self.step / self.loop_calls if self.loop_calls else 2.0

    def pace(self) -> str:
        if self.finished:
            return ""
        turns = self.budgets.turnsLeft(self.stepsPerTurn())
        if turns * self.stepsPerTurn() >= self.left() + 2:
            return ""
        seconds = int(self.budgets.timeLeft())
        if turns <= 2:
            return (f" TIME IS NEARLY UP ({seconds} s left): check out in this reply when the cart holds a line for each item; otherwise cart the best fitting listings you have opened"
                    " and check out, since running out of time shows nothing about the store.")
        return (f" Time is shorter than steps: about {seconds} s are left, enough for roughly {turns} more replies at the current pace. Put every independent tool call of the next move"
                " into one reply (open several listings at once), and plan to check out before the time is up.")

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.store.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            began = time.monotonic()
            result = _env.post_verbose("/environment/call", payload)
            self.budgets.noteEnv(time.monotonic() - began)
            if isinstance(result.data, dict):
                return self.envReply(name, args, result.data)
            if not self.envRetry(result.error or {}, payload):
                return None
        return None

    def envReply(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        seen = data.get("observation")
        if name in self.store.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
            for value in args.values():
                self.shelf.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
        return data

    def envRetry(self, error: dict[str, Any], payload: dict[str, Any]) -> bool:
        status = error.get("status")
        body = str(error.get("body") or "")
        self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
        if status == 409:
            low = body.lower()
            if "finish" in low or "quarantin" in low or "finaliz" in low:
                self.finished = True
                return False
            found = re.search(r"expected (\d+)", body)
            if found:
                self.step = int(found.group(1)) - 1
            elif "reuse" in low or "different action" in low:
                self.step += 1
            else:
                return False
            payload["step"] = self.step + 1
            return True
        if status in (400, 401, 402, 403, 404, 422):
            self.broke = self.broke or status == 402
            return False
        return True

    def stepHint(self, left: int) -> str:
        if left <= self.book.tail() + 1:
            return " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        if left <= 6 or (left <= len(self.book.items) + 8 and self.buys == 0):
            return " Few steps left: set the cart to the chosen lines and check out now."
        if self.buys == 0 and left <= 12:
            return " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        if self.buys == 0 and not self.shelf.cart_touched and self.step >= self.max_steps // 2:
            return " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        return ""

    def status(self) -> str:
        left = self.left()
        text = f"[Act] step {self.step} of {self.max_steps} used, {left} left." + self.stepHint(left) + self.pace()
        live = {need for _, need in self.book.needs_now()}
        lookups = [need for _, need in self.lookups.refs if need in live and need not in self.lookups.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.decline.leads:
            text += "\n[review candidates] " + ", ".join(self.decline.leads)
        if self.book.notes:
            text += "\n[changes so far] " + " | ".join(self.book.notes[-4:])
        return text + "\n[ledger]\n" + self.book.render()

    def ready_to_ask(self) -> bool:
        if self.asked or not self.store.roles.get("ask") or self.book.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        slack = self.left() - self.book.msgs_free() - (6 + 2 * len(self.book.items))
        opened = len(self.shelf.full)
        if opened >= 3 * len(self.book.items) + 1:
            return True
        return self.step >= max(6, self.max_steps // 5) or slack <= 0

    def interview(self) -> str:
        self.asked = True
        if self.book.msgs_free() <= 0:
            return ""
        self.shelf.openWide()
        return Interviewer(self).run()

    def show(self, observation: Any) -> str:
        text = render(observation, SHOW_CHARS)
        return text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + "...(cut)"

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        calls = self.shelf.join_opens(calls)
        return [(call["id"], self.runOne(call, len(calls) > 1)) for call in calls]

    def runOne(self, call: dict[str, Any], crowded: bool) -> str:
        if call.get("joined"):
            return "Shown in the side-by-side result above: Act opened these listings together in one step."
        fn = call.get("function") or {}
        job = Pending(fn.get("name"), fn.get("arguments"))
        for guard in (self.guardTool, self.guardArgs, self.guardCompare, self.guardGone, self.guardRemoval, self.guardEnding, self.guardRepeat, self.guardShown):
            text = guard(job, crowded)
            if text is not None:
                return text
        text = self.guard.hold(job)
        if text is not None:
            return text
        if job.ending and self.left() >= 2:
            go, said = self.procedures.run("order" if self.store.is_buy(job.name, job.args) else "end")
            if not go:
                return "Not run yet. " + said + "\nCart now: " + self.shelf.cart_text()
            if said:
                job.trimmed = "\n[Act] " + said + job.trimmed
        reply = self.act(job.name, job.args)
        if reply is None:
            return "The call failed (network or server). You may try again."
        observation = reply.get("observation")
        paged = self.learnReply(job, reply, observation)
        return self.renderReply(job, reply, observation, paged)

    def guardTool(self, job: Pending, crowded: bool) -> str | None:
        if self.finished:
            return "Not run: the episode is over."
        if job.name == self.store.roles.get("ask"):
            return "Not run (no step used): the interviewer handles every message to the shopper."
        allowed = sorted(t["function"]["name"] for t in self.store.offered())
        if job.name not in self.store.schemas:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(allowed)}."
        if job.name not in allowed:
            return "Not run (no step used): too few steps are left for that; use: " + ", ".join(allowed) + "."
        return None

    def guardArgs(self, job: Pending, crowded: bool) -> str | None:
        raw = job.raw
        if isinstance(raw, dict):
            args = raw
        else:
            args = ModelGate.parse(raw) if (raw or "").strip() else {}
            if args is None:
                return "Not run (no step used): the arguments were not one JSON object."
        faults: list[str] = []
        args = StoreMap.shape(args, self.store.schemas[job.name], "args", faults)
        if faults:
            return "Not run (no step used): " + "; ".join(faults[:6])
        args, wrong = self.store.resolve_ids(job.name, args)
        if wrong:
            return "Not run (no step used): " + wrong + "."
        job.args = args
        return None

    def guardCompare(self, job: Pending, crowded: bool) -> str | None:
        job.many = self.store.compare_target() if job.name == self.store.roles.get("compare") else None
        many = job.many
        if many and isinstance(job.args.get(many), list) and len(job.args[many]) > self.store.compare_room() and self.store.compare_bad:
            room = self.store.compare_room()
            job.trimmed = f"\n[Act] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, job.args[many][room:]))}."
            job.args = {**job.args, many: job.args[many][:room]}
        return None

    def guardGone(self, job: Pending, crowded: bool) -> str | None:
        job.target = self.store.single_target(job.name)
        target, args = job.target, job.args
        if target and job.name in (self.store.roles.get("open"), self.store.roles.get("offer")) and str(args.get(target)) in self.shelf.gone:
            return f"Not run (no step used): {args[target]} sold out at step {self.shelf.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."
        return None

    def guardRemoval(self, job: Pending, crowded: bool) -> str | None:
        args = job.args
        if job.name == self.store.roles.get("cart") and any(Text.count(v, 0) < 0 for v in args.values() if not isinstance(v, str)):
            view = render(self.shelf.cart_view) if self.shelf.cart_seen_at >= self.shelf.cart_at else ""
            absent = [v for v in args.values() if isinstance(v, str) and v in self.shelf.rows and v not in self.shelf.cart_lines and v not in view and v not in self.shelf.cart_said]
            if absent:
                return f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.shelf.cart_text()
        return None

    def guardEnding(self, job: Pending, crowded: bool) -> str | None:
        job.ending = self.store.is_buy(job.name, job.args) or self.store.is_refuse(job.name, job.args)
        if job.ending and crowded:
            return "Not run (no step used): send the finishing call on its own, after reading your other results."
        if not job.ending and self.left() <= self.book.tail() and (self.store.roles.get("buy") or self.store.roles.get("refuse")):
            return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
        return None

    def guardRepeat(self, job: Pending, crowded: bool) -> str | None:
        job.mark = job.name + render(job.args)
        if not job.ending and job.name != self.store.roles.get("cart") and job.mark in self.recent:
            return "Not run (no step used): you just made this same call and nothing changed; its result is above."
        job.readers = {self.store.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
        cached = self.read_cache.get(job.mark)
        if job.name in job.readers and cached and cached[0] == self.book.version:
            return f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]
        return None

    def guardShown(self, job: Pending, crowded: bool) -> str | None:
        name, args, many, target = job.name, job.args, job.many, job.target
        if target and name in job.readers and name != self.store.roles.get("compare") and self.store.shown_already(name, str(args.get(target))):
            return f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.shelf.record_of(str(args[target]))
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not self.store.shown_already(name, str(i))]
            if not keep:
                return ("Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                        + "\n".join(f"{i}: {self.shelf.record_of(str(i))}" for i in args[many])[:4000])
            if len(keep) < len(args[many]):
                job.trimmed += f"\n[Act] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                job.args = {**args, many: keep}
                job.mark = name + render(job.args)
        return None

    def learnReply(self, job: Pending, reply: dict[str, Any], observation: Any) -> str:
        name, args, many = job.name, job.args, job.many
        store, shelf = self.store, self.shelf
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            shown = render(observation)
            if not any(str(i) in shown for i in args[many]):
                store.compare_bad = min(store.compare_bad or 999, len(args[many]))
                shelf.opened.difference_update(str(i) for i in args[many] if str(i) not in shelf.full)
                job.trimmed += f"\n[Act] The side-by-side view showed nothing for {len(args[many])} listings: send at most {store.compare_room()} per call."
            else:
                store.compare_ok = max(store.compare_ok, len(args[many]))
        self.recent = (self.recent + [job.mark])[-3:]
        store.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in shelf.rows for x in (v if isinstance(v, list) else [v])))
        shelf.absorb(observation)
        if name == store.roles.get("compare"):
            rows = shelf.records_in(observation)
            if rows:
                store.facts_from[name] = store.facts_from.get(name, False) or any(Shelf.factBlock(r) for _, r in rows)
        if name in job.readers - {store.roles.get("search")}:
            store.note_view(name, observation)
        job.plain = {tool for role, tool in store.roles.items() if role != "search"}
        paged = self.coverage.more_pages(name, args, observation) if name not in job.plain and not job.ending else ""
        if name == store.roles.get("search") and not job.ending and isinstance(observation, dict):
            self.coverage.recent_search = ([(args, observation)] + self.coverage.recent_search)[:6]
            self.coverage.note_search(name, args, observation)
        if store.single_target(name) and name not in (store.roles.get("cart"), store.roles.get("buy"), store.roles.get("refuse"), store.roles.get("ask")):
            listing = shelf.ident(observation) if isinstance(observation, dict) else None
            if listing:
                store.facts_from[name] = store.facts_from.get(name, False) or bool(Shelf.factBlock(observation))
                if name == store.roles.get("open") and not store.facts_from[name]:
                    other = next((t for t in store.schemas if t != name and store.single_target(t) and store.facts_from.get(t) is not False
                                  and t not in (store.roles.get("cart"), store.roles.get("buy"), store.roles.get("refuse"), store.roles.get("ask"), store.roles.get("search"))), None)
                    if other:
                        store.roles["open"] = other
                if not store.facts_from[name] and name != store.roles.get("open") and not store.roles.get("offer"):
                    store.roles["offer"] = name
                if name == store.offer_tool():
                    shelf.offer_seen[listing] = self.step
        if name == store.roles.get("cart"):
            shelf.applyCart(name, args)
            shelf.cart_said = render(observation)
            self.recent = []
            counts = [Text.count(v) for v in args.values() if not isinstance(v, str) and Text.count(v) is not None]
            for value in args.values():
                if counts and isinstance(value, str) and value in shelf.rows:
                    if counts[0] <= 0 and not Shelf.inRows(observation, value):
                        shelf.dropped.add(value)
                    elif counts[0] > 0:
                        shelf.dropped.discard(value)
            lines = [i for i in shelf.cart_ids() if not shelf.sold_out(i) and i not in shelf.dropped]
            if len(lines) >= len(shelf.built):
                shelf.built = lines
        if name == store.roles.get("cart_view"):
            shelf.cart_view = observation
            shelf.cart_seen_at = self.step
        if store.is_buy(name, args):
            self.buys += 1
            if not reply.get("finished"):
                shelf.refused_at = self.step
        return paged

    def renderReply(self, job: Pending, reply: dict[str, Any], observation: Any, paged: str) -> str:
        name, args = job.name, job.args
        store, shelf, book = self.store, self.shelf, self.book
        text = self.show(observation) + paged + job.trimmed
        if name not in job.plain and not job.ending:
            text += self.coverage.cheapest_here([observation] + self.coverage.paged_obs)
        base = text
        proof = book.evidence(observation)
        if proof:
            text += "\n" + proof
        if name == store.roles.get("cart") and any(Text.count(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in shelf.rows and Shelf.inRows(observation, v) for v in args.values()):
                text += "\n[Act] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
        book.note_sold = False
        meaning = book.hearNote(observation)
        if meaning:
            text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
        if store.is_buy(name, args) and not reply.get("finished") and (book.note_sold or any(line in Text.prose(observation) for line in book.sold_lines)):
            seen = shelf.find_sold()
            if seen:
                text += "\n[Act re-read the cart's lines after the refusal]\n" + seen
        swap = shelf.swap_hint()
        if swap:
            text += "\n" + swap
        if book.queued:
            text += "\n[notes seen on Act's own calls] " + " | ".join(book.queued)
            book.queued = []
        if name in job.readers:
            self.read_cache[job.mark] = (book.version, self.step, base)
        if reply.get("finished"):
            text += " [episode finished]"
        return text

    def finish(self, again: bool = False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.store.roles.get("buy"), self.store.roles.get("refuse")
        fresh = self.shelf.cart_view is not None and self.shelf.cart_seen_at >= self.shelf.cart_at
        holding = bool(self.shelf.cart_ids()) if fresh else bool(self.shelf.cart_lines)
        if buy and refuse and holding and self.shelf.partial_cart():
            self.trace.append({"kind": "partial_cart_not_bought", "cart": self.shelf.cart_ids(), "items": len(self.book.items)})
            holding = False
            self.shelf.cart_touched = False
        if buy and not holding and not self.decline.decline_proven and not self.decline.completed and self.left() >= len(self.book.items) + 2:
            stale = self.decline.audit_runs == 0 or self.book.version != self.decline.audit_version
            if not self.budgets.closing() and (stale or (not self.decline.lead_items and self.shelf.cart_ids())):
                self.decline.audit(near=not self.decline.lead_items and bool(self.shelf.cart_ids()), late=True)
            done = self.decline.buy_leads()
            if done or self.finished:
                return ""
        if buy and (holding or (self.shelf.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = StoreMap.shape(dict(self.store.brief.get("buy_args") or {}), self.store.schemas.get(buy), "args", faults)
            if not faults:

                if self.left() >= 2:
                    self.procedures.run("order")
                if self.left() >= 6 and not self.finished:
                    self.lookups.trail(2)
                if self.left() >= 6 and not self.finished and self.shelf.cart_ids():
                    self.checker.check_cart(None, self.budgets.spendHard * 1.3)
                    self.checker.swapBetter() or self.checker.leadSwap()
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": Text.clip(reply, 200), "left": self.left(), "seconds": round(self.budgets.elapsed(), 1), "cost": round(self.models.cost, 4)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.shelf.refused_at = self.step
                    self.book.observe(buy, reply.get("observation"))
                    lines = self.shelf.cart_ids()
                    if lines and self.left() >= 2 and not any(self.shelf.sold_out(i) for i in lines) and not self.shelf.partial_cart():
                        if self.left() >= 3:
                            self.procedures.run("order")
                        if self.finished or self.left() <= 0:
                            return ""
                        second = self.act(buy, args)
                        self.trace.append({"kind": "forced_buy_again", "reply": Text.clip(second, 200)})
                        if self.finished or self.left() <= 0:
                            return ""
                        if second is not None:
                            self.buys += 1
                            self.shelf.refused_at = self.step
                            self.book.observe(buy, second.get("observation"))
                            return "the checkout was refused: " + self.show(second.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            faults = []
            args = StoreMap.shape(self.store.quiet(refuse, dict(self.store.brief.get("refuse_args") or {})), self.store.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.procedures.run("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": Text.clip(reply, 200), "left": self.left(), "seconds": round(self.budgets.elapsed(), 1), "cost": round(self.models.cost, 4)})
                if reply is not None and not self.finished:
                    self.book.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        total = sum(len(render(m.get("content"))) for m in self.chat)
        if total <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) - TRIM_KEEP)]:
            if total <= HISTORY_LIMIT * TRIM_TO:
                break
            content = self.chat[index].get("content") or ""
            if len(content) <= 600:
                continue
            ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.shelf.rows][:20]
            short = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"
            total -= len(render(content)) - len(render(short))
            self.chat[index]["content"] = short

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.store.readBrief()
        self.book.fromBrief()
        self.trace.append({"kind": "brief", "brief": self.store.brief})
        opening = self.store.view.get("opening")
        if opening is not None:
            self.shelf.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.store.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(self.budgets.spendHard)
        if not self.finished:
            turned = self.finish(again=True)

            if turned and not self.finished and not self.broke and not self.models.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Act] Act tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.budgets.spendHard * 1.15)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.models.cost, 5), "calls": self.models.calls, "loop_calls": self.loop_calls,
                           "seconds": round(self.budgets.elapsed(), 1), "model_seconds": round(self.budgets.modelSeconds(), 2), "env_seconds": round(self.budgets.envSeconds(), 2)})
        return self.trace

    def toolCalls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            raw = fn.get("arguments")
            parsed = raw if isinstance(raw, dict) else (ModelGate.parse(raw) if isinstance(raw, str) and raw.strip() else {})
            calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                          "function": {"name": str(fn.get("name") or ""), "arguments": render(parsed if isinstance(parsed, dict) else {})}})
        return calls

    def askFirst(self, calls: list[dict[str, Any]]) -> bool:
        committing = {self.store.roles.get(r) for r in ("cart", "buy", "refuse")} - {None}
        if self.asked or not self.store.roles.get("ask") or self.book.msgs_free() <= 0 or not any(c["function"]["name"] in committing for c in calls) or self.left() <= 8:
            return False
        for call in calls:
            self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
        summary = self.interview()
        self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
        return True

    def answer(self, replies: list[tuple[str, str]]) -> None:
        last = len(replies) - 1
        for index, (cid, text) in enumerate(replies):
            self.chat.append({"role": "tool", "tool_call_id": cid, "content": text + ("\n" + self.status() if index == last else "")})

    def work(self, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.budgets.callRoom(self.models.cost, spend_cap) > 0:
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.models.think(self.chat, [LIGHT, SPARE], 3000, self.store.offered(), "low")
            if message is None:
                self.misses += 1
                if self.models.no_credit or self.models.calls >= self.budgets.allCalls or self.misses > 2:
                    break
                continue
            calls = self.toolCalls(message)
            content = message.get("content")
            if not isinstance(content, str):
                content = render(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            if self.askFirst(calls):
                continue
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            self.answer(self.run_calls(calls))

def main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Act(problem_data)
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

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return main(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]

__all__ = ["agent_main"]
