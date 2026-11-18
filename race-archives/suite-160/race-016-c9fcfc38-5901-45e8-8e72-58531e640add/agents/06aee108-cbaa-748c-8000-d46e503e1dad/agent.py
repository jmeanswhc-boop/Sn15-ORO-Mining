from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
TOP = "anthropic/claude-opus-5.5"
SWEEP = False
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
SMALL_TABLE = 4
PLAN_ROWS = 12
ASK_PAIR = False
PAIR_ASK = """Also return "pair": the first two questions to send, chosen together as the pair most likely to hit the shopper's unstated preferences (they may be on two items or both on one item), best first. Each {"item": item number, "key": that item's table key, copied exactly, "quote": the request's words that point to it, copied word for word, or "" when only the tables point to it}. Never a key the request settles or a key in "skip"."""
ASK_KEYS = 6
ASK_ACROSS = False
ASK_RATES = (0.92, 0.65, 0.50, 0.42, 0.42, 0.32)
ASK_RANK_TOP = 0.22
ASK_RANK_DECAY = 0.8
ASK_ITEM_FEW = 0.8
ASK_ITEM_MANY = 0.45
ASK_ITEM_AGAIN = 0.15
ASK_OUTSIDE = 0.08

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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts; a price limit the request sets for one product alone (not for the whole order) goes in that item's "cap": the most one unit of it may cost, as a number in the request's own unit, else "cap" is null. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false, "quote": the answer's own words that give the value, copied exactly}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number in the unit the request uses for money (a limit given in cents stays a count of cents), or null, "refused": "limit" when an answer is not the shopper's reply at all but a store or system notice that no more messages can be sent (never for a shopper who did not follow the question), else null}
- A message can ask about several properties: report every need its answer states, each on its own item.
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
Return ONE JSON object and nothing else: {"items": [{"item": item number, "fit": [ids of that item's OPENED listings whose records state every requirement in force as met, cheapest first, at most 4], "same": [ids of that item's OPENED listings not in "fit" whose records state every requirement as met, some of them in other words that mean the same value (a synonym such as easy for simple or grey for gray; never a looser, wider, nearby or merely related value), cheapest first, at most 4], "near": [ids of that item's OPENED listings with no contradicted requirement but with some requirement their records do not state, fewest unstated first, then cheapest, at most 4], "contradicted": [{"id": listing id, "requirement": "...", "quote": "the record's own words"}] for at most 3 listings of that item, the clearest}], "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "fallback": true when the request itself says to put forward the closest or nearest listing when nothing fits, else false, "proven": true only when some requested item has listings shown below and every one of them has a contradicted requirement, else false, "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

REFER = """A shopper's requirement takes its value from the page of another listing in the store, which the requirement describes (its seller or shop, its kind of product, some of its property values), or from the page that listing's page sends the reader on to. Read the requirement and the shopper's words below and return ONE JSON object and nothing else:
{"key": the property of the product being bought whose value is wanted, as the shopper names it,
"key_quote": the shopper's words naming that property,
"relation": "pairing" when the wanted value is the one that goes with, pairs with, works alongside or mates up with the value the last page gives; "not" only when the wanted value must differ from the value the page gives (anything other than it); "matching" when it is that same value, including wording that limits the choice to what the page gives (nothing but it, only it, no other than it); "other" for any other relation to it (a range around it, at least or at most it),
"relation_quote": the shopper's words that set the relation,
"relation_words": for "other", the requirement's words for that relation with the page's value left out (such as "within 2 kg either way of", "at least"), else "",
"seller": the seller or shop of the described listing, copied exactly,
"shelf": the kind of product of the described listing in the shopper's words, or "",
"values": [{"key": a property of the described listing, "value": its value as written, "forms": [the value as written, then the same amount or word as a listing page may write it in other units or spellings, at most 6, ending with the bare number when a page may leave the unit out (for 0.5 m: "0.5 m", "50 cm", "1.64 ft", "19.69 in", "0.5"; for 1 1/2 in: "1 1/2 in", "1.5 in", "3.81 cm", "1.5")], "quote": the shopper's words stating it}]}
Copy every quote exactly from the shopper's words. The properties that describe the other listing are never the wanted property."""

PAGE = """A shopping agent follows a chain of listing pages to learn the value of one property, the KEY below. It has just opened the page shown. From the page's own words it must learn whether this page is the listing described, and whether the page sends the reader on to another listing for the KEY. Return ONE JSON object and nothing else:
{"same": true only when this page is the listing described: it is sold by the named seller, it is that kind of product, and it states every described value (the same amount in other units or spellings counts; a different amount, size, colour or material does not); else false,
"shown": [[each described value, the page's words that state it]],
"mismatch": [[a described detail this page does not match, the page's words that show it]] (a note saying that a version with one of the described values exists elsewhere means this page is not the one described: list it here),
"own": this page's own value for the KEY: its attribute for that property; when it has none, the value its title states; else what its seller's note says about this item itself; "" when the page gives none,
"own_quote": the page's words holding that value,
"next": the one sentence that tells the reader to take the KEY from another listing instead of this one (go by, look to, refer to, use, take it from, is to be taken from another listing), copied whole; "" when there is none. A sentence that only says another version or listing exists elsewhere, or that sends the reader elsewhere for a different property, is not it,
"next_key": the property that sentence is about, as it names it,
"next_seller": the seller or shop it names, copied exactly,
"next_shelf": the kind of product it names, or "",
"next_values": [{"key": a property, "value": its value as the sentence writes it, "forms": [as written, then the same amount in other units or spellings, then the bare number when it has one, at most 6]}] for each detail the sentence gives about that listing,
"pairs": [[a value, the value it goes with]] for each rule on the page that pairs values ("A goes with B"),
"pairs_quote": the words of that rule}
Copy every quote exactly from the page."""

FITS = """The listing records below may state product codes: the listing's own model code (for example "Model: AB-1234"), the codes of the models it works with or fits (a list, or a family pattern such as "every AB-12xx model", where each x stands for any character), codes it names as exceptions, its own part code (often in brackets in its title), part codes it points to for the rest of its fits ("see part CD123 for the rest"), and statements of what another part also fits ("Part CD123 also fits: ..."). Read them from each record's own words and return ONE JSON object and nothing else:
{"codes": [{"id": listing id, "model": its own model code or "", "fits": [codes of models it fits, copied exactly], "families": [family patterns, copied exactly], "except": [codes it excludes], "part": its own part code or "", "see": [part codes it points to for more fits], "also": [[a part code, [the codes that part also fits]]]}]}
List every record shown, with empty fields when it states no code."""

HOP_CALLS = 10
HOP_MORE = 6
HOP_OPENS = 5
CHAIN_GUESS = 4

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

ASK_PLAN = """You help a shopping agent choose what to ask a shopper. The shopper holds a few firm preferences the request never states, each on one property of one requested product. Every message to the shopper asks about one property of one product, and messages are few.

You get the request and, for each requested item, what the request already settles and the attribute keys its opened listings carry (most carried first).

Return:
"story": the properties of a requested item that a sentence of the request hints at without settling them: who will use it, where it goes or stands, how it is used, what went wrong before, how it must look, feel, fit, carry or hold. Each {"item": item number, "key": that item's table key naming the property, copied exactly from its table (plain words when the item has no table), "quote": the request's sentence that hints at it, copied word for word}. Strongest hint first, at most PICKS. Leave out properties the request already settles for that item, and words that only name the product.
"ask_me": properties the request says to check with the shopper, each {"item": item number, "key": as for "story", "quote": the request's words that say so}.
"fits": the items the request says must be compatible with, fit or work with another requested item or with a product the shopper already has, each {"item": item number, "quote": the request's words that say so}.
"skip": for each item, the table keys that are not a choice a buyer of that kind of product states: identifiers, codes, model or part numbers, warranty, condition, services, origin, shipping, stock, short descriptions, counts of parts or slots, and keys whose values list purchase options or bundle several properties. Each {"item": item number, "keys": [at most 6 keys, copied exactly]}.
"rank": for each item with a table, up to 4 of its table keys that a buyer of that kind of product most often holds a firm preference on, best first, preferring keys that most of its listings carry. Each {"item": item number, "keys": [...]}.
"plain": only for an item with no opened listing or with fewer than 4 table keys, the properties a buyer of that kind of product chooses by, best first, in plain words. Each {"item": item number, "keys": [...]}.
Return ONE JSON object and nothing else: {"story": [...], "ask_me": [...], "fits": [...], "skip": [...], "rank": [...], "plain": [...]}"""

CAP = """A shopping agent may send messages to the shopper. Read the shopper's request and the store's tools below and return ONE JSON object and nothing else: {"messages": how many messages the agent may send to the shopper in this task (integer), or null when neither the request nor a tool description states a number, "quote": the words of the request or of that tool's description that state it, copied exactly, or "", "meter": the exact name of a tool that reports how many messages are left or used, or null when no tool does}."""

METER = """A store tool reports the messages a shopping agent may still send to the shopper. Read its result and return ONE JSON object and nothing else: {"left": how many more messages may be sent (integer) or null, "used": how many were sent so far (integer) or null, "total": the whole allowance (integer) or null}. Give only numbers the result shows."""

STORE = """A shopping agent reads a store's tools before it searches. Name every way the tools can list several listings at once (a keyword search, a category or shelf listing, a filter by brand or price). Return ONE JSON object and nothing else: {"routes": [{"tool": the tool's exact name, "fixed": {argument: value} that this way needs (for a tool whose mode argument picks between ways, that mode's value), else {}, "words": the argument that takes free search words, or null, "section": the argument that takes a category, shelf, section or department name, or null, "brand": the argument that takes a brand or maker name, or null, "cap": the argument that keeps out listings priced above a number, or null, "page": the argument that asks for a later page (an offset, a page number or a token), or null}]}
List a tool with a mode argument once for each mode that lists listings. Copy tool and argument names exactly. Leave out tools that show one listing, show listings named by id, change or show the cart, message the shopper or end the task."""

VALUES = """A shopping agent searches a store for listings that meet a shopper's requirements. For each requirement below, say how a listing's attribute must read to meet it. Return ONE JSON object and nothing else: {"needs": [{"need": the requirement copied exactly, "kind": one of "equals" (a listing meets it only by stating one of a few named values in words: a colour, a material, a type, a style, a pattern, a named feature), "not" (it rules out one or a few named values and accepts anything else), "range" (a number with a unit: an exact size or amount, a lower or upper limit, or a range), "other" (a condition, a fit with another item, a value taken from another listing, or anything else), "values": for "equals" and "not", each named value copied word for word from the requirement (one entry per alternative), else [], "bounds": for "range", the limit written in the requirement's own unit first and then the same limit in each other unit a listing of this kind may state it in, each {"low": the smallest accepted number or null, "high": the largest accepted number or null, "unit": the unit as a listing would write it, or "" for a count written as a bare number}, else []}]}"""

class TextRows:
    FIELD = re.compile(r"([a-z_][a-z0-9_]*): (.*)", re.S)
    LITERALS = {str(v): v for v in (True, False, None)}

    @staticmethod
    def value(text: str) -> Any:
        bare = text.strip()
        if bare in TextRows.LITERALS:
            return TextRows.LITERALS[bare]
        if re.fullmatch(r"-?\d+", bare):
            return int(bare)
        if re.fullmatch(r"-?\d+\.\d+", bare):
            return float(bare)
        pairs = TextRows.pairs(bare)
        return pairs if pairs is not None else bare

    @staticmethod
    def pairs(text: str) -> dict[str, Any] | None:
        parts = [p for p in text.split("; ") if p.strip()]
        if not parts:
            return None
        out: dict[str, Any] = {}
        for part in parts:
            key, sep, rest = part.partition("=")
            key = key.strip()
            if not sep or not key or len(key) > 48 or "=" in key:
                return None
            out[key] = rest.strip()
        return out

    @staticmethod
    def row(text: str) -> dict[str, Any] | None:
        fields: list[list[str]] = []
        parts = text.split(" | ") if " | " in text else [p.strip() + (" " if p.strip().endswith(":") else "") for p in text.split("\n") if p.strip()]
        for segment in parts:
            found = TextRows.FIELD.fullmatch(segment)
            if found:
                fields.append([found.group(1), found.group(2)])
            elif fields:
                fields[-1][1] += " | " + segment
            else:
                return None
        if len(fields) < 3 or len({k for k, _ in fields}) < len(fields):
            return None
        return {k: TextRows.value(v) for k, v in fields}

    @staticmethod
    def parse(node: Any, depth: int = 0) -> Any:
        if depth > 6:
            return node
        if isinstance(node, str):
            got = TextRows.row(node)
            return got if got is not None else node
        if isinstance(node, list):
            return [TextRows.parse(v, depth + 1) for v in node]
        if isinstance(node, dict):
            return {k: TextRows.parse(v, depth + 1) for k, v in node.items()}
        return node

    @staticmethod
    def observation(node: Any) -> Any:
        got = TextRows.parse(node)
        if isinstance(got, dict) and len(got) == 1 and isinstance(node, dict):
            only = next(iter(node.values()))
            inner = next(iter(got.values()))
            if isinstance(only, str) and isinstance(inner, dict):
                return inner
        return got


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
        self.cap: float | None = None
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
        keys = StoreMap.textFields(props)
        if not keys:
            return out
        decline = self.host.decline
        decline.muteNotice()
        for key in keys:
            limit = props[key].get("maxLength") if isinstance(props[key].get("maxLength"), int) and props[key]["maxLength"] > 0 else 200
            said = decline.declineText(limit)
            if out.get(key) is None and key not in required and not decline.named:
                continue
            out[key] = said
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
        since = max(host.step - 2, self.refused_at, self.cart_at + 1)
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

    def unitsIn(self, seen: Any) -> dict[str, int]:
        found: dict[str, int] = {}
        price_key = self.host.store.price_key
        stack: list[tuple[Any, int]] = [(seen, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ids = [v for v in node.values() if isinstance(v, str) and v in self.rows]
                if len(ids) == 1:
                    price = self.price_of(ids[0])
                    ints = [v for k, v in node.items() if isinstance(v, int) and not isinstance(v, bool) and k != price_key and (price is None or v != price) and 0 <= v <= 99]
                    if len(ints) == 1:
                        found[ids[0]] = ints[0]
                stack.extend((v, depth + 1) for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend((v, depth + 1) for v in node)
        return found

    def unitCalls(self, fix: dict[str, tuple[int, int]], lines: dict[str, int]) -> list[dict[str, Any]] | None:
        store = self.host.store
        name = store.roles.get("cart") or ""
        schema = store.schemas.get(name) or {}
        props = schema.get("properties") or {}
        if self.cartLineShape(props):
            return self.planCart(lines, [])
        texts = StoreMap.textFields(props)
        target = (next((k for t, k in store.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in StoreMap.kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target or not count:
            return None
        sets = store.cart_sets(name)
        calls: list[dict[str, Any]] = []
        for ident, (have, want) in fix.items():
            if sets:
                calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), count: want})
            elif mode:
                calls.append({target: ident, mode: props[mode]["enum"][0 if want > have else 1], count: abs(want - have)})
            elif isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= want - have:
                calls.append({target: ident, count: want - have})
            else:
                return None
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
            self.cart_said = render(reply.get("observation"))
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
        self.cap_from = ""
        self.meter: str | None = None
        self.meter_read = False
        self.pending_refs: list[dict[str, Any]] = []

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
            cap = entry.get("cap")
            if isinstance(cap, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", cap):
                cap = float(cap)
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
                self.items[-1].cap = float(cap)
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
            self.cap_from = "brief"
        elif host.store.roles.get("ask"):
            self.msg_limit = 1
            self.cap_from = "assumed"
        steps = Text.count(brief.get("steps"))
        if steps is not None and 10 <= steps < host.max_steps:
            host.max_steps = steps
        for entry in Text.items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                talks = max(0, Text.count(entry.get("messages")) or 0)
                if talks and when not in ("choose", "order"):
                    continue
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order", "messages": talks})
        if host.store.roles.get("ask"):
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def capSaid(self, quote: str, limit: int) -> bool:
        store = self.host.store
        sources = [store.request] + [str(t["function"].get("description") or "") for t in store.tools]
        digits = re.findall(r"\d+", quote)
        if digits and str(limit) not in digits:
            return False
        return any(Interviewer.quoted(quote, text) for text in sources if text)

    def holdFor(self) -> None:
        talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
        self.reserve = min(max(talk), self.msg_limit) if talk and self.host.store.roles.get("ask") else 0

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
        if item.cap is not None:
            head += f", price at most {item.cap:g} {self.host.store.brief.get('budget_unit') or ''} for one".rstrip()
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

    @staticmethod
    def wrapNotes(answer: Any) -> dict[str, list[str]]:
        if not isinstance(answer, dict) or answer.get("error"):
            return {}
        lists = {k: v for k, v in answer.items() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)}
        rest = [v for k, v in answer.items() if k not in lists and isinstance(v, str) and v.strip()]
        return lists if lists and rest else {}

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
        notes = Ledger.wrapNotes(answer)
        if notes:
            meaning = self.hearNote(notes)
            if meaning:
                self.queued.append(f"(seen with a reply) {meaning}")
                words = Text.prose({k: v for k, v in answer.items() if k not in notes})
                host.trace.append({"kind": "ask_note", "note": meaning[:200]})
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
        self.chains: dict[str, dict[str, Any]] = {}
        self.chain_pages: list[str] = []
        self.rejected: set[str] = set()
        self.seller_keys: dict[str, int] = {}
        self.query_cap: int | None = None
        self.begun = False
        self.reported: set[str] = set()
        self.resolved_needs: dict[str, str] = {}

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
        got = host.models.ask_json(prompt, [LIGHT, HEAVY], 500) or {}
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

    def begin(self) -> None:
        self.begun = True
        if self.live() and not self.host.finished:
            self.follow_all()
            text = self.chainText()
            if text:
                self.host.book.queued.append(text)

    def live(self) -> list[tuple[int, str]]:
        now = {need for _, need in self.host.book.needs_now()}
        return [(number, need) for number, need in self.refs if need in now and need not in self.ref_values]

    def unresolved(self) -> list[str]:
        return [need for _, need in self.live()]

    def reserve_calls(self) -> int:
        total = 0
        for _, need in self.live():
            state = self.chains.get(need) or {}
            if state.get("state") in ("lost", "unparsed"):
                continue
            done = len(state.get("pages") or [])
            total += 3 * max(1, CHAIN_GUESS - done) + 4
        return total

    def room(self, extra: int = 0) -> bool:
        host = self.host
        floor = host.book.tail() + 3 * len(host.book.items) + 4 + extra
        return (not host.finished and not host.broke and not host.models.no_credit and host.left() > floor
                and not host.budgets.closing() and host.models.cost < host.budgets.spendHard * 1.4)

    @staticmethod
    def flat(text: Any) -> str:
        low = str(text or "").lower()
        low = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", low)
        low = re.sub(r"[^a-z0-9./]+", " ", low)
        low = re.sub(r"(?<!\d)[./]|[./](?!\d)", " ", low)
        return " ".join(low.split())

    @staticmethod
    def quoted(quote: Any, source: Any) -> bool:
        want = Lookups.flat(quote)
        return bool(want) and f" {want} " in f" {Lookups.flat(source)} "

    @staticmethod
    def alike(one: str, two: str) -> bool:
        one, two = Text.single(one), Text.single(two)
        return one == two or (len(one) >= 4 and len(two) >= 4 and one[:4] == two[:4])

    @staticmethod
    def keyNamed(sentence: str, key: str, other: str) -> bool:
        words = Lookups.flat(sentence).split()
        mine = [w for w in Lookups.flat(key).split() if len(w) >= 3]
        theirs = [w for w in Lookups.flat(other).split() if len(w) >= 3]
        near = bool(mine) and bool(theirs) and all(any(Lookups.alike(a, b) for b in theirs) for a in mine)
        for name, ok in ((mine, True), (theirs, near)):
            if name and ok and all(any(Lookups.alike(a, w) for w in words) for a in name):
                return True
        return False

    @staticmethod
    def sentences(record: Any) -> list[str]:
        found: list[str] = []
        stack: list[tuple[Any, int]] = [(record, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 5:
                continue
            if isinstance(node, str):
                found += [s.strip() for s in re.split(r"(?<=[.!?])\s+", node) if s.strip()]
            elif isinstance(node, dict):
                stack.extend((v, depth + 1) for v in node.values())
            elif isinstance(node, list):
                stack.extend((v, depth + 1) for v in node)
        return found

    def sourceText(self, need: str) -> str:
        said = [str(e.get("a") or "") for e in self.host.trace if isinstance(e, dict) and e.get("kind") == "ask"]
        return "\n".join([self.host.store.request] + said + [need])

    @staticmethod
    def bareForm(value: Any, key: Any) -> str:
        words = Lookups.flat(value).split()
        mine = [w for w in Lookups.flat(key).split() if len(w) >= 3]
        kept = [w for w in words if not any(Lookups.alike(w, k) for k in mine)]
        return " ".join(kept) if mine and kept and len(kept) < len(words) else ""

    @staticmethod
    def readValues(entries: Any, source: str | None) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for entry in Text.items(entries):
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("value") or "").strip()
            key = str(entry.get("key") or "").strip()
            forms = [value] + [str(f).strip() for f in Text.items(entry.get("forms")) if isinstance(f, (str, int, float)) and str(f).strip()]
            if not value:
                continue
            quote = str(entry.get("quote") or "")
            if source is not None and not (Lookups.quoted(quote, source) or any(Lookups.quoted(f, source) for f in forms)):
                continue
            kept: list[str] = []
            for form in forms:
                if Lookups.flat(form) and Lookups.flat(form) not in [Lookups.flat(k) for k in kept]:
                    kept.append(form)
            bare: list[str] = []
            for form in (Lookups.bareForm(f, key) for f in kept):
                if form and Lookups.flat(form) not in [Lookups.flat(k) for k in kept + bare]:
                    bare.append(form)
            where = quote if source is not None and quote and Lookups.quoted(quote, source) else (source or "")
            keyed = not where or bool(bare) or Lookups.keyNear(source or where, key, kept) or Lookups.keyNear(where, key, kept)
            values.append({"key": key, "value": value, "forms": kept[:6], "bare": bare[:3], "keyed": keyed})
        return values[:4]

    def parse(self, need: str) -> dict[str, Any] | None:
        host = self.host
        source = self.sourceText(need)
        prompt = REFER + "\n\nREQUIREMENT: " + need + "\n\nTHE SHOPPER'S WORDS:\n" + source[-6000:]
        best = None
        for models in ([HEAVY, LIGHT], [LIGHT]):
            got = host.models.ask_json(prompt, models, 1500)
            parsed = self.readParse(got, need, source)
            host.trace.append({"kind": "chain_parse", "need": need[:160], "got": Text.clip(got, 120), "ok": parsed is not None})
            if parsed is not None and (best is None or len(parsed["start"]["values"]) > len(best["start"]["values"])):
                best = parsed
            if (best is not None and best["start"]["values"]) or host.models.no_credit or host.finished:
                break
        return best

    def readParse(self, got: Any, need: str, source: str) -> dict[str, Any] | None:
        if not isinstance(got, dict):
            return None
        seller = str(got.get("seller") or "").strip()
        if not seller or not Lookups.quoted(seller, source):
            return None
        relation = str(got.get("relation") or "").strip().lower()
        words = str(got.get("relation_words") or "").strip()
        if relation == "other" and not (words and Lookups.quoted(words, source)):
            return None
        relation = relation if relation in ("pairing", "not", "matching", "other") else "matching"
        shelf = str(got.get("shelf") or "").strip()
        key = need.split(":", 1)[0].strip() if ":" in need else ""
        return {"key": key or str(got.get("key") or "").strip(), "relation": relation, "words": words, "relation_ok": Lookups.quoted(got.get("relation_quote"), source),
                "start": {"seller": seller, "shelf": shelf if Lookups.quoted(shelf, source) else "", "values": Lookups.readValues(got.get("values"), source)}}

    def queryKey(self, finder: str) -> str | None:
        shape = self.host.store.shapes.get(finder) or {}
        if shape.get("query"):
            return shape["query"]
        schema = self.host.store.schemas.get(finder) or {}
        texts = StoreMap.textFields(schema.get("properties") or {})
        required = [k for k in schema.get("required") or [] if k in texts]
        if len(required) == 1:
            return required[0]
        return texts[0] if len(texts) == 1 else None

    def queryRoom(self, finder: str, key: str) -> int:
        spec = ((self.host.store.schemas.get(finder) or {}).get("properties") or {}).get(key) or {}
        cap = spec["maxLength"] if isinstance(spec.get("maxLength"), int) and spec["maxLength"] > 0 else 100
        learned = getattr(self.host.store, "query_limit", None)
        if isinstance(learned, int) and not isinstance(learned, bool) and learned > 0:
            cap = min(cap, learned)
        return min(cap, self.query_cap) if self.query_cap else cap

    @staticmethod
    def compose(parts: list[str], room: int) -> str:
        out = ""
        for part in parts:
            part = " ".join(str(part or "").split())
            if not part:
                continue
            trial = (out + " " + part).strip()
            if len(trial) <= room:
                out = trial
            elif not out:
                out = part[:room].rsplit(" ", 1)[0] if " " in part[:room] else part[:room]
        return out

    def plans(self, desc: dict[str, Any], room: int) -> list[str]:
        seller = desc["seller"]
        stated = [(v.get("bare") or v["forms"] or [v["value"]])[0] for v in desc["values"]]
        rest = [v["forms"][1:] for v in desc["values"]]
        other = [f for k in range(max([len(r) for r in rest] + [0])) for r in rest if k < len(r) for f in [r[k]]]
        out: list[str] = []
        for parts in ([seller] + stated + [desc["shelf"]], [seller] + other, [seller, desc["shelf"]], [seller]):
            words = Lookups.compose(parts, room)
            if words and words not in out:
                out.append(words)
        return out

    def unwrap(self, finder: str, seen: Any) -> Any:
        shape = self.host.store.shapes.get(finder) or {}
        rows = shape.get("rows")
        if isinstance(seen, dict) and rows and rows not in seen:
            inner = next((v for v in seen.values() if isinstance(v, dict) and rows in v), None)
            return inner if inner is not None else seen
        return seen

    def lookFor(self, finder: str, key: str, words: str, token: Any) -> tuple[list[tuple[str, dict[str, Any]]], Any, bool] | None:
        host = self.host
        args: dict[str, Any] = {key: words}
        if token is not None:
            shape = host.store.shapes.get(finder) or {}
            if not shape.get("next_arg"):
                return None
            args[shape["next_arg"]] = token
        call = host.store.shapedArgs(finder, args)
        if call is None:
            return None
        reply = host.act(finder, call)
        if reply is None:
            return None
        seen = reply.get("observation")
        host.book.observe(finder, seen)
        host.read_cache[finder + render(call)] = (host.book.version, host.step, host.show(seen))
        shape = host.store.read_shape(finder, self.unwrap(finder, seen)) or {}
        body = self.unwrap(finder, seen)
        rows = host.shelf.records_in(body)
        fault = not rows and isinstance(body, dict) and not any(isinstance(v, list) for v in body.values())
        nxt = body.get(shape["next"]) if isinstance(body, dict) and shape.get("next") else None
        if isinstance(nxt, bool) or not isinstance(nxt, (str, int)) or nxt == "":
            nxt = None
        return rows, nxt, fault

    def sellerKey(self, rows: list[tuple[str, dict[str, Any]]], seller: str) -> dict[str, list[str]]:
        want = Lookups.flat(seller)
        hits: dict[str, list[str]] = {}
        for ident, row in rows:
            for k, v in row.items():
                if k != self.host.shelf.id_key and isinstance(v, str) and len(v) <= 60 and Lookups.flat(v) == want:
                    hits.setdefault(k, []).append(ident)
        return hits

    def ofSeller(self, rows: list[tuple[str, dict[str, Any]]], seller: str) -> list[str]:
        hits = self.sellerKey(rows, seller)
        words = Lookups.flat(seller).split()
        while not hits and len(words) > 1:
            words = words[:-1]
            hits = self.sellerKey(rows, " ".join(words))
        for k, ids in hits.items():
            self.seller_keys[k] = self.seller_keys.get(k, 0) + len(ids)
        if hits:
            best = max(hits, key=lambda k: (self.seller_keys.get(k, 0), len(hits[k])))
            return list(dict.fromkeys(hits[best]))
        want = Lookups.flat(seller)
        return [ident for ident, row in rows if want and f" {want} " in f" {Lookups.flat(render(row))} "]

    def openPage(self, ident: str, tool: str | None = None) -> bool:
        host = self.host
        paged = self.__dict__.setdefault("paged", set())
        page = host.store.offer_tool() or host.store.facts_tool()
        tool = tool or page
        if tool == page and (ident in paged or ident in host.shelf.opened):
            return True
        target = host.store.single_target(tool)
        if not tool or not target:
            return False
        args = host.store.shapedArgs(tool, {target: ident})
        if args is None:
            return False
        opened = host.shelf.openOne(tool, args, "single")
        if opened is None:
            return False
        seen = opened[1]
        if isinstance(seen, dict) and (seen.get("error") or not seen):
            return False
        if tool == page:
            host.shelf.opened.add(ident)
            paged.add(ident)
        return True

    def openFacts(self, ident: str) -> bool:
        host = self.host
        tool = host.store.facts_tool() or host.store.roles.get("open")
        if not tool or tool == host.store.offer_tool() or Shelf.factBlock(host.shelf.full.get(ident) or {}) or not self.room(2):
            return False
        return self.openPage(ident, tool)

    def judge(self, key: str, desc: dict[str, Any], ident: str) -> dict[str, Any]:
        host = self.host
        record = host.shelf.full.get(ident) or host.shelf.rows.get(ident) or {}
        if desc["values"] and self.grade(desc, record) < 0:
            host.trace.append({"kind": "chain_page", "id": ident, "same": False, "strong": False, "why": "a stated value contradicted (code)", "own": "", "next": "", "doubt": False})
            return {"id": ident, "same": False, "strong": False, "level": -1, "next": None, "own": "", "pairs": [], "doubt": False, "why": "a stated value contradicted"}
        shown = {"seller": desc["seller"], "kind": desc["shelf"], "values": [{"key": v["key"], "value": v["value"], "forms": v["forms"]} for v in desc["values"]]}
        prompt = PAGE + "\n\nKEY: " + key + "\n\nDESCRIBED LISTING: " + render(shown) + "\n\nPAGE:\n" + render(record, 900)[:9000]
        got = host.models.ask_json(prompt, [LIGHT, HEAVY], 1800)
        verdict = self.verdictOf(got, desc, key, ident, record)
        if verdict["doubt"] and verdict["same"]:
            again = host.models.ask_json(prompt, [HEAVY], 1800)
            second = self.verdictOf(again, desc, key, ident, record)
            if second["same"] and (not second["doubt"] or (second["next"] and not verdict["next"])):
                verdict = second
        host.trace.append({"kind": "chain_page", "id": ident, "same": verdict["same"], "strong": verdict["strong"], "why": verdict.get("why", ""),
                           "own": verdict["own"], "next": (verdict["next"] or {}).get("seller", ""), "doubt": verdict["doubt"]})
        return verdict

    def verdictOf(self, got: Any, desc: dict[str, Any], key: str, ident: str, record: Any) -> dict[str, Any]:
        text = render(record)
        out: dict[str, Any] = {"id": ident, "same": False, "strong": False, "level": 0, "next": None, "own": "", "pairs": [], "doubt": False}
        if not isinstance(got, dict):
            out["why"] = "no reading"
            return out
        pointer = str(got.get("next") or "").strip()
        bad = [m for m in Text.items(got.get("mismatch")) if isinstance(m, list) and len(m) >= 2 and Lookups.quoted(m[1], text) and not Lookups.pointsOn(str(m[1]), pointer, desc["seller"])]
        shown = [m for m in Text.items(got.get("shown")) if isinstance(m, list) and len(m) >= 2 and Lookups.quoted(m[1], text)]
        level = self.grade(desc, record)
        found = sum(1 for v in desc["values"] if any(Lookups.quoted(f, text) for f in Lookups.formsOf(v)))
        if got.get("same") is not True and (bad or len(desc["values"]) < 2 or level < 1 or found < len(desc["values"]) or len(shown) < len(desc["values"]) - 1):
            out["why"] = "not the described listing"
            return out
        if bad:
            out["why"] = "mismatch: " + str(bad[0][0])[:80]
            return out
        if not desc["values"]:
            kind = [w for w in Lookups.flat(desc["shelf"]).split() if len(w) >= 3]
            have = Lookups.flat(text).split()
            found = sum(1 for w in kind if any(Lookups.alike(w, h) for h in have))
            if 4 * found < 3 * len(kind) or not kind:
                out["why"] = "values not shown"
                return out
        elif level < 0 or (level == 0 and max(found, len(shown)) < max(1, len(desc["values"]) - 1)):
            out["why"] = "a stated value contradicted" if level < 0 else "values not shown"
            return out
        elif any(not v.get("keyed", True) and not any(Lookups.quoted(f, text) for f in Lookups.formsOf(v)) for v in desc["values"]):
            out["why"] = "values not shown"
            return out
        level = max(level, 0)
        out["same"], out["strong"], out["level"] = True, level == 2, level
        own = str(got.get("own") or "").strip()
        proof = str(got.get("own_quote") or "")
        if own and (Lookups.quoted(own, text) or (Lookups.quoted(proof, text) and Lookups.quoted(own, proof))):
            out["own"] = own
        for pair in Text.items(got.get("pairs")):
            if isinstance(pair, list) and len(pair) >= 2 and all(isinstance(p, str) and Lookups.quoted(p, text) for p in pair[:2]):
                out["pairs"].append((pair[0].strip(), pair[1].strip()))
        sentence = str(got.get("next") or "").strip()
        seller = Lookups.sellerOnly(str(got.get("next_seller") or "").strip(), str(got.get("next_shelf") or ""))
        other = str(got.get("next_key") or "").strip()
        if not sentence and not seller:
            if not out["own"]:
                out["strong"], out["level"] = False, 0
            return out
        if not Lookups.quoted(sentence, text):
            fits = [s for s in Lookups.sentences(record) if seller and Lookups.quoted(seller, s) and Lookups.keyNamed(s, key, other)]
            sentence = fits[0] if len(fits) == 1 else ""
        if sentence and seller and Lookups.quoted(seller, sentence) and Lookups.keyNamed(sentence, key, other):
            shelf = str(got.get("next_shelf") or "").strip()
            out["next"] = {"seller": seller, "shelf": shelf if Lookups.quoted(shelf, sentence) else "", "values": Lookups.readValues(got.get("next_values"), sentence), "quote": sentence}
            out["doubt"] = not out["next"]["values"]
        else:
            out["doubt"] = True
        return out

    @staticmethod
    def pointsOn(quote: str, pointer: str, seller: str) -> bool:
        return bool(pointer) and len(Lookups.flat(pointer).split()) >= 4 and (Lookups.quoted(pointer, quote) or Lookups.quoted(quote, pointer)) and not Lookups.quoted(seller, pointer)

    @staticmethod
    def keyNear(source: str, key: str, forms: list[str]) -> bool:
        words = Lookups.flat(source).split()
        mine = [w for w in Lookups.flat(key).split() if len(w) >= 3]
        if not mine or not words:
            return False
        for form in forms:
            want = Lookups.flat(form).split()
            if not want:
                continue
            for start in range(len(words) - len(want) + 1):
                if words[start:start + len(want)] == want:
                    near = words[max(0, start - 5):start] + words[start + len(want):start + len(want) + 5]
                    if any(Lookups.alike(a, w) for a in mine for w in near):
                        return True
        return False

    @staticmethod
    def formsOf(value: dict[str, Any]) -> list[str]:
        return list(value.get("forms") or []) + list(value.get("bare") or [])

    @staticmethod
    def sellerOnly(seller: str, shelf: str) -> str:
        words = seller.split()
        kind = {Text.single(w) for w in Lookups.flat(shelf).split()}
        while len(words) > 1 and Text.single(Lookups.flat(words[-1])) in kind:
            words = words[:-1]
        return " ".join(words)

    def hopCap(self) -> int:
        host = self.host
        floor = host.book.tail() + 3 * len(host.book.items) + 4
        return HOP_CALLS + (HOP_MORE if host.left() - floor >= 3 * (HOP_CALLS + HOP_MORE) else 0)

    def grade(self, desc: dict[str, Any], record: Any) -> int:
        if not desc["values"]:
            return 0
        block = Shelf.factBlock(record) or {}
        text = render(record)
        level = 2
        for value in desc["values"]:
            mine = [w for w in Lookups.flat(value["key"]).split() if len(w) >= 3] if value.get("keyed", True) else []
            forms = Lookups.formsOf(value)
            held = [str(v) for k, v in block.items() if mine and all(any(Lookups.alike(a, b) for b in Lookups.flat(Text.words(k)).split()) for a in mine)]
            if held and not any(Lookups.quoted(f, h) for f in forms for h in held):
                return -1
            if held and any(Lookups.quoted(f, h) for f in forms for h in held):
                continue
            if not mine and any(Lookups.quoted(f, str(h)) for f in forms for h in block.values()):
                continue
            level = 1 if any(Lookups.quoted(f, text) for f in forms) and level else 0
        return level

    def hop(self, key: str, desc: dict[str, Any], chain: list[str]) -> tuple[dict[str, Any] | None, int, list[str]]:
        host = self.host
        finder = host.store.roles.get("search")
        field = self.queryKey(finder) if finder else None
        if not finder or not field:
            return None, 0, []
        began = host.step
        cap = self.hopCap()
        tried = set(chain)
        rejected: list[str] = []
        weak = None
        words = self.plans(desc, self.queryRoom(finder, field))
        queue: list[tuple[int, Any]] = [(k, None) for k in range(min(2, len(words)))] + [(k, None) for k in range(2, len(words))]
        done: set[tuple[int, str]] = set()
        while queue:
            plan, token = queue.pop(0)
            if (plan, str(token)) in done:
                continue
            done.add((plan, str(token)))
            if True:
                if host.step - began >= cap or not self.room():
                    return weak, host.step - began, rejected
                found = self.lookFor(finder, field, words[plan], token)
                if found is None:
                    continue
                rows, token, fault = found
                if fault and len(words[plan]) > 12 and not self.query_cap:
                    self.query_cap = max(12, len(words[plan]) * 2 // 3)
                    words = self.plans(desc, self.queryRoom(finder, field))
                    queue = [(k, None) for k in range(len(words))]
                    done = set()
                    continue
                mine = self.ofSeller(rows, desc["seller"])
                fresh = [i for i in mine if i not in tried]
                for ident in fresh[:HOP_OPENS]:
                    if host.step - began >= cap or not self.room():
                        break
                    tried.add(ident)
                    if not self.openPage(ident):
                        continue
                    verdict = self.judge(key, desc, ident)
                    if verdict["same"] and verdict["strong"]:
                        return verdict, host.step - began, rejected
                    if verdict["same"] and (weak is None or verdict["level"] > weak["level"]):
                        weak = verdict
                    elif not verdict["same"]:
                        rejected.append(ident)
                        self.rejected.add(ident)
                if weak is not None:
                    return weak, host.step - began, rejected
                if token is not None and rows and len(mine) == len(rows):
                    queue.insert(min(len(queue), 1 if plan == 0 else 0), (plan, token))
        return weak, host.step - began, rejected

    def partner(self, own: str, pairs: list[tuple[str, str]]) -> str:
        mine = Lookups.flat(own)
        for first, second in pairs:
            if Lookups.flat(first) == mine:
                return second
        for first, second in pairs:
            if Lookups.flat(second) == mine:
                return first
        for first, second in pairs:
            if Lookups.quoted(first, own) or Lookups.quoted(own, first):
                return second
        return ""

    def settle(self, number: int, need: str, state: dict[str, Any], verdict: dict[str, Any]) -> bool:
        own = verdict["own"]
        if not own:
            return False
        relation = state["parse"]["relation"]
        value = own
        stated = self.partner(own, verdict.get("pairs") or []) if relation in ("matching", "not") else ""
        if relation == "pairing":
            value = self.partner(own, verdict["pairs"]) or own
            state["paired"] = value != own
        elif relation == "not":
            value = "anything but " + (stated or own)
            state["paired"] = bool(stated)
        elif stated:
            value = stated
            state["paired"] = True
        elif relation == "other":
            value = state["parse"]["words"] + " " + own
        key = state["parse"]["key"] or "value"
        state.update(state="done", own=own, value=value)
        resolved = f"{key}: {value}"
        for item in self.host.book.items:
            item.musts = [resolved if m == need else m for m in item.musts]
            item.told = [resolved if m == need else m for m in item.told]
        self.ref_values[need] = value
        self.resolved_needs[need] = resolved
        self.host.book.version += 1
        return True

    def follow(self, number: int, need: str) -> None:
        host = self.host
        state = self.chains.setdefault(need, {"state": "new", "pages": [], "hops": [], "marks": []})
        if state["state"] in ("done", "lost", "unparsed"):
            return
        if not state.get("parse"):
            parsed = self.parse(need)
            if parsed is None:
                state["state"] = "unparsed"
                return
            state["parse"] = parsed
            state["desc"] = parsed["start"]
        key = state["parse"]["key"]
        while not host.finished:
            desc = state["desc"]
            mark = Lookups.flat(desc["seller"]) + "|" + "|".join(sorted(Lookups.flat(v["value"]) for v in desc["values"]))
            if mark in state["marks"]:
                state["state"] = "lost"
                host.trace.append({"kind": "chain_cycle", "need": need[:120], "pages": state["pages"]})
                return
            if not self.room(2):
                state["state"] = "paused"
                return
            verdict, calls, rejected = self.hop(key, desc, state["pages"])
            state["hops"].append({"calls": calls, "page": verdict["id"] if verdict else "", "rejected": rejected, "strong": bool(verdict and verdict["strong"])})
            host.trace.append({"kind": "chain_hop", "need": need[:120], "hop": len(state["hops"]), "calls": calls, "page": verdict["id"] if verdict else "", "rejected": rejected})
            if verdict is None:
                state["state"] = "lost" if self.room(2) else "paused"
                return
            state["marks"].append(mark)
            state["pages"].append(verdict["id"])
            if verdict["id"] not in self.chain_pages:
                self.chain_pages.append(verdict["id"])
            if verdict["next"] is None and not verdict["own"] and self.openFacts(verdict["id"]):
                verdict = self.judge(key, desc, verdict["id"])
            if verdict["next"] is None:
                ok = self.settle(number, need, state, verdict)
                if not ok:
                    state["state"] = "lost"
                host.trace.append({"kind": "chain_end", "need": need[:120], "pages": state["pages"], "own": verdict["own"], "value": state.get("value", ""), "relation": state["parse"]["relation"], "ok": ok})
                return
            state["desc"] = verdict["next"]

    def follow_all(self) -> None:
        for number, need in self.live():
            if self.host.finished or not self.room(2):
                break
            self.follow(number, need)

    def chainText(self) -> str:
        parts: list[str] = []
        for need, state in self.chains.items():
            if need in self.reported or state.get("state") != "done":
                continue
            self.reported.add(need)
            pages = " -> ".join(state["pages"])
            rule = " and its pairing rule gives " + state["value"] if state.get("paired") else ""
            parts.append(f"'{need[:200]}': Act opened every page of its chain ({pages}); the last page gives {state['own']}{rule}, so the requirement is now '{self.resolved_needs.get(need, '')}'")
        if not parts:
            return ""
        return "[Act followed the page chains itself] " + " | ".join(parts) + ".\n"

    def trail(self, rounds: int = 6) -> tuple[str, bool]:
        if self.live() and not self.host.finished:
            self.follow_all()
        text = self.chainText()
        rest = [need for _, need in self.live() if (self.chains.get(need) or {}).get("state") in ("unparsed", "lost", None)]
        if not rest or self.host.finished:
            return text, False
        more, hold = self.guessTrail(rounds, rest)
        return text + more, hold

    def guessTrail(self, rounds: int = 6, only: list[str] | None = None) -> tuple[str, bool]:
        live = {need for _, need in self.host.book.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values and (only is None or need in only)]
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
        self.route_list: list[dict[str, Any]] | None = None
        self.fetched: dict[str, dict[str, Any]] = {}
        self.page_rows: dict[str, int] = {}
        self.reach_seen: dict[str, int] = {}
        self.cursor_top: dict[str, int] = {}
        self.text_cut: dict[str, int] = {}
        self.order_votes: dict[str, list[int]] = {}
        self.logic = ""
        self.kinds: dict[str, dict[str, Any]] = {}
        self.pools: dict[int, dict[str, Any]] = {}
        self.sections: dict[int, str] = {}
        self.marks: dict[int, str] = {}
        self.pool_marks: set[str] = set()
        self.sweep_calls = 0
        self.swept = False
        self.refreshes = 0

    def routes(self) -> list[dict[str, Any]]:
        if self.route_list is not None:
            return self.route_list
        host = self.host
        store = host.store
        self.route_list = []
        finder = store.roles.get("search")
        closed = {store.roles.get(r) for r in ("cart", "cart_view", "buy", "refuse", "ask", "open", "offer", "compare")} - {None, finder}
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {render(t['function'].get('parameters') or {})}"
                 for t in store.tools if t["function"]["name"] not in closed]
        got = host.models.ask_json(STORE + "\n\nTOOLS:\n" + "\n".join(lines), [LIGHT, HEAVY], 1500) if lines else None
        for entry in Text.items((got or {}).get("routes")):
            route = self.routeOf(entry, closed)
            if route and route not in self.route_list:
                self.route_list.append(route)
        if finder and finder not in closed and not any(r["tool"] == finder and r["words"] for r in self.route_list):
            plain = self.plainRoute(finder)
            if plain:
                self.route_list.append(plain)
        host.trace.append({"kind": "routes", "routes": self.route_list})
        return self.route_list

    def routeOf(self, entry: Any, closed: set[Any]) -> dict[str, Any] | None:
        store = self.host.store
        if not isinstance(entry, dict) or not isinstance(entry.get("tool"), str):
            return None
        tool = entry["tool"]
        if tool not in store.schemas or tool in closed:
            return None
        schema = store.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        route: dict[str, Any] = {"tool": tool, "fixed": {}}
        for role in ("words", "section", "brand", "cap", "page"):
            arg = entry.get(role)
            route[role] = arg if isinstance(arg, str) and isinstance(props.get(arg), dict) else None
        for role, kinds in (("words", {"string"}), ("section", {"string"}), ("brand", {"string"}), ("cap", {"number", "integer"}), ("page", {"string", "integer", "number"})):
            if route[role] and not set(StoreMap.kinds(props[route[role]])) & kinds:
                route[role] = None
        taken = {route[r] for r in ("words", "section", "brand", "cap", "page")} - {None}
        fixed = entry.get("fixed") if isinstance(entry.get("fixed"), dict) else {}
        for arg, value in fixed.items():
            if arg in props and arg not in taken:
                faults: list[str] = []
                shaped = StoreMap.shape(value, props[arg], arg, faults)
                if faults:
                    return None
                route["fixed"][arg] = shaped
        if not (route["words"] or route["section"]):
            return None
        need = [k for k in schema.get("required") or [] if k not in route["fixed"] and k not in (route["words"], route["section"])]
        return None if need else route

    def plainRoute(self, tool: str) -> dict[str, Any] | None:
        schema = self.host.store.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        texts = StoreMap.textFields(props)
        required = [k for k in schema.get("required") or [] if k in props]
        words = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not words or any(k != words and not (isinstance(props[k], dict) and isinstance(props[k].get("enum"), list) and props[k]["enum"]) for k in required):
            return None
        fixed = {k: props[k]["enum"][0] for k in required if k != words}
        pages = [k for k, v in props.items() if k != words and k not in fixed and isinstance(v, dict) and "integer" in StoreMap.kinds(v)]
        return {"tool": tool, "fixed": fixed, "words": words, "section": None, "brand": None, "cap": None, "page": pages[0] if len(pages) == 1 else None}

    def textRoute(self) -> dict[str, Any] | None:
        found = [r for r in self.routes() if r["words"]]
        return next((r for r in found if not r["section"]), None) or (found[0] if found else None)

    def shelfRoute(self) -> dict[str, Any] | None:
        found = [r for r in self.routes() if r["section"]]
        return next((r for r in found if r["cap"]), None) or (found[0] if found else None)

    def argSpec(self, route: dict[str, Any], arg: str | None) -> dict[str, Any]:
        props = (self.host.store.schemas.get(route["tool"]) or {}).get("properties") or {}
        spec = props.get(arg or "")
        return spec if isinstance(spec, dict) else {}

    def pageArg(self, route: dict[str, Any]) -> str | None:
        if route.get("page"):
            return route["page"]
        arg = (self.host.store.shapes.get(route["tool"]) or {}).get("next_arg")
        return arg if arg and self.argSpec(route, arg) and arg not in route["fixed"] else None

    def textLimit(self, route: dict[str, Any]) -> int:
        spec = self.argSpec(route, route.get("words"))
        top = spec["maxLength"] if isinstance(spec.get("maxLength"), int) and spec["maxLength"] > 0 else 200
        return max(1, min(top, self.text_cut.get(route["tool"], top)))

    @staticmethod
    def fitWords(words: list[str], limit: int) -> str:
        kept: list[str] = []
        for word in words:
            if word and word not in kept and len(" ".join(kept + [word])) <= limit:
                kept.append(word)
        return " ".join(kept) if kept else (words[0][:limit] if words else "")

    def reachOf(self, route: dict[str, Any]) -> int:
        tool = route["tool"]
        if tool in self.reach_seen:
            return self.reach_seen[tool]
        size = self.page_rows.get(tool) or 10
        spec = self.argSpec(route, self.pageArg(route))
        top = self.cursor_top.get(tool, spec.get("maximum") if isinstance(spec.get("maximum"), (int, float)) and not isinstance(spec.get("maximum"), bool) else None)
        if top is None:
            return 5 * size
        low = spec.get("minimum") if isinstance(spec.get("minimum"), (int, float)) else 0
        return int(top) * size if low >= 1 else int(top) + size

    def noteOrder(self, tool: str, rows: list[str]) -> None:
        if len(rows) < 3:
            return
        shelf = self.host.shelf
        ups = sum(1 for a, b in zip(rows, rows[1:]) if a < b)
        prices = [shelf.price_of(i) for i in rows]
        cheap = None not in prices and all(prices[k] <= prices[k + 1] for k in range(len(prices) - 1))
        votes = self.order_votes.setdefault(tool, [0, 0, 0, 0])
        votes[0] += 1
        votes[1] += ups
        votes[2] += 1 if cheap else 0
        votes[3] += len(rows) - 1

    def orderOf(self, tool: str) -> str:
        votes = self.order_votes.get(tool)
        if not votes or not votes[0]:
            return ""
        if votes[2] == votes[0]:
            return "price"
        return "id" if 5 * votes[1] >= 4 * votes[3] else "rank"

    def fetch(self, route: dict[str, Any], values: dict[str, Any], cursor: Any = None) -> dict[str, Any] | None:
        host = self.host
        store = host.store
        tool = route["tool"]
        args = dict(route["fixed"])
        for role, value in values.items():
            if route.get(role) and value is not None and value != "":
                args[route[role]] = value
        arg = self.pageArg(route)
        if cursor is not None:
            if not arg:
                return None
            args[arg] = cursor
        call = store.shapedArgs(tool, args)
        if call is None:
            return None
        mark = tool + render(call)
        if mark in self.fetched:
            return self.fetched[mark]
        if host.finished or host.broke or host.left() <= self.reserve() or host.budgets.closing() or self.sweep_calls >= self.allowance():
            return None
        reply = host.act(tool, call)
        if reply is None:
            return None
        self.sweep_calls += 1
        seen = reply.get("observation")
        host.book.observe(tool, seen)
        host.read_cache[mark] = (host.book.version, host.step, host.show(seen))
        rows = [i for i, _ in host.shelf.records_in(seen)]
        shape = (store.read_shape(tool, seen) if isinstance(seen, dict) else None) or store.shapes.get(tool) or {}
        count = seen.get(shape["count"]) if shape.get("count") and isinstance(seen, dict) else None
        if not isinstance(count, int) or isinstance(count, bool):
            count = None
        after = seen.get(shape["next"]) if shape.get("next") and isinstance(seen, dict) else None
        if after is not None and (isinstance(after, bool) or not isinstance(after, (str, int))):
            after = None
        wrong = ""
        if not rows and count is None and seen:
            text = render(seen)
            wrong = next((role for role in ("words", "section", "brand", "cap") if route.get(role) and route[role] in text), "")
            if not wrong and arg and arg in text:
                wrong = "page"
        if rows:
            self.page_rows[tool] = max(self.page_rows.get(tool, 0), len(rows))
        if tool == store.roles.get("search") and isinstance(seen, dict) and rows:
            self.recent_search = ([(call, seen)] + self.recent_search)[:6]
            self.paged_obs = []
            self.note_search(tool, call, seen)
        self.noteOrder(tool, rows)
        if cursor is None and route.get("words"):
            self.noteTally(route, {k: v for k, v in values.items() if v is not None and v != ""}, count)
        got = {"rows": rows, "count": count, "next": after, "has_next": bool(shape.get("next")), "wrong": wrong, "call": call}
        self.fetched[mark] = got
        return got

    def nextCursor(self, route: dict[str, Any], got: dict[str, Any], cursor: Any) -> Any:
        if got["next"] is not None:
            return got["next"]
        if got["has_next"]:
            return None
        arg = self.pageArg(route)
        spec = self.argSpec(route, arg)
        if not arg or "integer" not in StoreMap.kinds(spec):
            return None
        low = spec["minimum"] if isinstance(spec.get("minimum"), int) else 0
        here = cursor if isinstance(cursor, int) and not isinstance(cursor, bool) else low
        after = here + (1 if low >= 1 else max(1, len(got["rows"])))
        top = self.cursor_top.get(route["tool"], spec.get("maximum"))
        if isinstance(top, (int, float)) and not isinstance(top, bool) and after > top:
            return None
        return after

    def walk(self, route: dict[str, Any], values: dict[str, Any], room: int, label: str = "", group: bool = False) -> dict[str, Any]:
        tool = route["tool"]
        out: dict[str, Any] = {"ids": [], "count": None, "ended": False, "full": False, "used": 0, "wrong": "", "top": None}
        cursor: Any = None
        last: str | None = None
        asked: list[str] = []
        while len(asked) < 40 and (out["used"] < room or cursor is None):
            if render(cursor) in asked:
                break
            asked.append(render(cursor))
            before = self.sweep_calls
            got = self.fetch(route, values, cursor)
            out["used"] += self.sweep_calls - before
            if got is None:
                break
            if got["wrong"] == "page":
                if isinstance(cursor, int):
                    self.cursor_top[tool] = cursor - 1
                break
            if got["wrong"]:
                out["wrong"] = got["wrong"]
                break
            if out["count"] is None:
                out["count"] = got["count"]
            rows = got["rows"]
            for ident in rows:
                if group and self.groupEnds(tool, ident, last, label, str(values.get("words") or "")):
                    out["ended"] = True
                    break
                if ident not in out["ids"]:
                    out["ids"].append(ident)
                if not label or self.onShelf(ident, label):
                    last = ident
            if out["ended"]:
                break
            if out["count"] is not None and len(out["ids"]) >= out["count"]:
                out["full"] = True
                break
            size = self.page_rows.get(tool, 0)
            if not rows or len(rows) < size:
                out["full"] = True
                break
            cursor = self.nextCursor(route, got, cursor)
            if cursor is None:
                self.reach_seen[tool] = max(self.reach_seen.get(tool, 0), len(out["ids"]))
                break
            if out["used"] >= room:
                break
        prices = [p for p in (self.host.shelf.price_of(i) for i in out["ids"]) if p is not None]
        if prices and self.orderOf(tool) == "price":
            out["top"] = max(prices)
        return out

    def noteTally(self, route: dict[str, Any], values: dict[str, Any], count: Any) -> None:
        words = values.get("words")
        if self.logic or not isinstance(words, str) or not isinstance(count, int) or len(values) != 1:
            return
        mine = frozenset(re.findall(r"[a-z0-9]+", words.lower()))
        seen = self.__dict__.setdefault("tallies", [])
        for other, number in seen:
            if other < mine and count != number:
                self.logic = "or" if count > number else "and"
            elif mine < other and count != number:
                self.logic = "or" if number > count else "and"
        seen.append((mine, count))

    def onShelf(self, ident: str, label: str) -> bool:
        shelf = self.host.shelf
        return shelf.kind_of(ident, shelf.category_key()) == label.strip().lower()

    def groupEnds(self, tool: str, ident: str, last: str | None, label: str, words: str = "") -> bool:
        if self.orderOf(tool) != "id" or last is None or ident >= last:
            return False
        if label and not self.onShelf(ident, label):
            return False
        want = Text.bag(words)
        return not (want and want <= Text.bag(render(self.host.shelf.rows.get(ident) or {})))

    def shelfWord(self, label: str) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) >= 3]
        if not words:
            return ""
        df, _ = self.vocabDf()
        return min(words, key=lambda w: (df.get(w, 0), -len(w)))

    def labelFor(self, number: int) -> str:
        if number in self.sections:
            return self.sections[number]
        shelf = self.host.shelf
        key = shelf.category_key()
        if not key:
            return ""
        seen = {str((shelf.rows.get(i) or {}).get(key) or "").strip() for i in shelf.seen_order} - {""}
        label = Interviewer(self.host).itemLabel(number)
        return label if label in seen else ""

    @staticmethod
    def covers(label: str, name: str) -> bool:
        have = [w for w in Text.stem(label) if len(w) >= 3]
        want = [w for w in Text.stem(name) if len(w) >= 4]

        def alike(a: str, b: str) -> bool:
            return a == b or (len(a) >= 4 and len(b) >= 4 and abs(len(a) - len(b)) <= 2 and (a.startswith(b) or b.startswith(a)))

        return bool(want) and all(any(alike(w, h) for h in have) for w in want)

    def discover(self, number: int) -> str:
        label = self.labelFor(number)
        route = self.textRoute()
        item = self.host.book.items[number - 1]
        if label and not (Coverage.covers(label, item.name) or (item.shelf and Coverage.covers(label, item.shelf))):
            label = ""
        tried: list[str] = []
        for name in (item.name, item.shelf):
            if label or not route:
                break
            words = self.fitWords(re.findall(r"[a-z0-9]+", str(name or "").lower()), self.textLimit(route))
            if not words or words in tried:
                continue
            tried.append(words)
            got = self.fetch(route, {"words": words})
            if got and got["wrong"] == "words" and len(words) > 3:
                self.text_cut[route["tool"]] = min(self.text_cut.get(route["tool"], len(words)), len(words) - 1)
            label = self.labelFor(number)
            if label and not Coverage.covers(label, str(name or "")) and name == item.name and item.shelf:
                label = ""
        label = label or self.labelFor(number)
        if label:
            self.sections[number] = label
        return label

    def sectionValue(self, route: dict[str, Any], number: int, label: str) -> str:
        options = self.argSpec(route, route["section"]).get("enum")
        if not isinstance(options, list) or not options:
            return label
        item = self.host.book.items[number - 1]
        for want in (label, item.shelf, item.name):
            hit = next((o for o in options if isinstance(o, str) and want and Text.words(o) == Text.words(want)), None)
            if hit:
                return hit
        return ""

    def capValue(self, route: dict[str, Any], cap: float) -> Any:
        if "integer" in StoreMap.kinds(self.argSpec(route, route["cap"])):
            return int(cap)
        return int(cap) if float(cap).is_integer() else round(cap, 2)

    def needsOf(self, number: int) -> list[str]:
        return [need for n, need in self.host.book.needs_now() if n == number]

    def learnKinds(self) -> None:
        host = self.host
        pointing = {need for _, need in host.lookups.refs}
        needs = [need for _, need in host.book.needs_now() if need not in self.kinds and need not in pointing]
        for need in pointing:
            self.kinds.setdefault(need, {"kind": "other", "values": []})
        if not needs:
            return
        got = host.models.ask_json(VALUES + "\n\nREQUIREMENTS:\n" + "\n".join(needs), [LIGHT, HEAVY], 1200) or {}
        for entry in Text.items(got.get("needs")):
            if not isinstance(entry, dict):
                continue
            said = str(entry.get("need") or "").strip()
            match = said if said in needs else next((n for n in needs if Text.words(n) == Text.words(said)), None)
            if not match or match in self.kinds:
                continue
            have = Text.bag(match.split(":", 1)[1] if ":" in match else match)
            values: list[list[str]] = []
            for value in Text.items(entry.get("values")):
                words = re.findall(r"[a-z0-9]+", str(value).lower())
                if words and set(map(Text.single, words)) <= have and words not in values:
                    values.append(words)
            bounds = Coverage.boundsOf(entry.get("bounds"), match)
            kind = entry.get("kind")
            if kind in ("equals", "not") and values:
                self.kinds[match] = {"kind": kind, "values": values, "bounds": []}
            elif kind == "range" and bounds:
                self.kinds[match] = {"kind": "range", "values": [], "bounds": bounds}
            else:
                self.kinds[match] = {"kind": "other", "values": [], "bounds": []}
        for need in needs:
            self.kinds.setdefault(need, {"kind": "other", "values": []})

    @staticmethod
    def boundsOf(given: Any, need: str) -> list[dict[str, Any]]:
        stated = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", need)]
        out: list[dict[str, Any]] = []
        for entry in Text.items(given):
            if not isinstance(entry, dict) or not isinstance(entry.get("unit"), str):
                continue
            low, high = entry.get("low"), entry.get("high")
            low = float(low) if isinstance(low, (int, float)) and not isinstance(low, bool) else None
            high = float(high) if isinstance(high, (int, float)) and not isinstance(high, bool) else None
            unit = " ".join(re.findall(r"[a-z]+", entry["unit"].lower()))
            if low is None and high is None:
                continue
            if not out and not any(abs(x - v) < 1e-6 for x in stated for v in (low, high) if v is not None):
                return []
            out.append({"low": low, "high": high, "unit": unit})
        return out[:8]

    @staticmethod
    def unitMatch(have: str, want: str) -> bool:
        a, b = Text.single(have), Text.single(want)
        return a == b or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)))

    @staticmethod
    def inBounds(key: str, value: Any, bounds: list[dict[str, Any]]) -> bool | None:
        text = str(value).lower()
        found = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]*)", text)
        named = Text.words(key).split()
        verdicts: list[bool] = []
        for number, unit in found:
            for bound in bounds:
                want = (bound["unit"].split() or [""])[0]
                if not want:
                    same = not unit and not any(Coverage.unitMatch(w, b["unit"].split()[0]) for b in bounds if b["unit"] for w in named)
                else:
                    same = (unit and Coverage.unitMatch(unit, want)) or (not unit and any(Coverage.unitMatch(w, want) for w in named))
                if same:
                    x = float(number)
                    verdicts.append((bound["low"] is None or x >= bound["low"] - 1e-9) and (bound["high"] is None or x <= bound["high"] + 1e-9))
                    break
        if not verdicts:
            return None
        return all(verdicts)

    def reserve(self) -> int:
        host = self.host
        book = host.book
        msgs = max(0, book.msgs_free()) if host.store.roles.get("ask") and not host.asked else 0
        live = {need for _, need in book.needs_now()}
        refs = sum(1 for _, need in host.lookups.refs if need in live and need not in host.lookups.ref_values)
        links = sum(len(pairs) for pairs in host.checker.linkedNeeds().values())
        props = (host.store.schemas.get(host.store.roles.get("cart") or "") or {}).get("properties") or {}
        writes = 1 if host.shelf.cartLineShape(props) else len(book.items)
        return msgs + (3 * 2 + 4) * refs + links + writes + 3 + book.tail()

    def ceiling(self, number: int) -> float | None:
        book = self.host.book
        budget = book.budget
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
            return None
        cap = book.items[number - 1].cap if 1 <= number <= len(book.items) else None
        if self.host.store.brief.get("budget_scope") == "each":
            return min(float(budget), cap) if cap is not None else float(budget)
        others = sum((self.lowest(m) or 0.0) * item.qty for m, item in enumerate(book.items, 1) if m != number)
        room = max(0.0, (float(budget) - others) / max(1, book.items[number - 1].qty))
        return min(room, cap) if cap is not None else room

    def lowest(self, number: int) -> float | None:
        label = (self.pools.get(number) or {}).get("label") or self.sections.get(number) or ""
        shelf = self.host.shelf
        key = shelf.category_key()
        if not label or not key:
            return None
        prices = [shelf.price_of(i) for i in shelf.seen_order if shelf.kind_of(i, key) == label.lower() and shelf.buyable(i)]
        found = [p for p in prices if p is not None]
        return min(found) if found else None

    def exactWords(self, number: int) -> list[list[str]]:
        found: list[list[str]] = []
        for need in self.needsOf(number):
            kind = self.kinds.get(need) or {}
            first = (kind.get("bounds") or [None])[0] if kind.get("kind") == "range" else None
            if first and first["low"] is not None and first["low"] == first["high"] and first["unit"]:
                found.append([f"{first['low']:g}"] + first["unit"].split())
        return found

    def valuePool(self, number: int, route: dict[str, Any], label: str, room: int) -> dict[str, Any] | None:
        eq = [self.kinds[need]["values"] for need in self.needsOf(number) if (self.kinds.get(need) or {}).get("kind") == "equals"]
        hints = self.exactWords(number)
        if not eq and hints and room > 0:
            word = self.shelfWord(label)
            got = self.groupPool(route, [w for words in hints for w in words] + ([word] if word else []), hints[0] + ([word] if word else []), word, label, room)
            return {"ids": got["ids"], "complete": False, "used": got["used"], "queries": [got["query"]]}
        if not eq or room <= 0:
            return None
        df, _ = self.vocabDf()
        eq.sort(key=lambda alts: (-len(alts), min(sum(df.get(w, 0) for w in words) for words in alts)))
        word = self.shelfWord(label)
        split = eq[0] if len(eq[0]) > 1 else [eq[0][0]]
        rest = [alts[0] for alts in eq[1:] if len(alts) == 1]
        rest.sort(key=lambda words: sum(df.get(w, 0) for w in words))
        out: dict[str, Any] = {"ids": [], "complete": True, "used": 0, "queries": []}
        for alt in split:
            base = list(alt) + ([word] if word and word not in alt else [])
            words = base + [w for extra in rest for w in extra]
            got = self.groupPool(route, words, base, word, label, max(1, room - out["used"]))
            out["used"] += got["used"]
            out["queries"].append(got["query"])
            out["ids"] += [i for i in got["ids"] if i not in out["ids"]]
            out["complete"] = out["complete"] and got["complete"]
            if out["used"] >= room:
                out["complete"] = out["complete"] and alt == split[-1]
                break
        shelf = self.host.shelf
        key = shelf.category_key()
        if not any(shelf.kind_of(i, key) == label.lower() for i in out["ids"]):
            out["complete"] = False
        return out

    def groupPool(self, route: dict[str, Any], words: list[str], base: list[str], probe: str, label: str, room: int) -> dict[str, Any]:
        tool = route["tool"]
        used = 0
        query = self.fitWords(words, self.textLimit(route))
        if len(query.split()) < len(base) and len(base) > 1:
            query = self.fitWords(base, self.textLimit(route))
        first = None
        for _ in range(3):
            before = self.sweep_calls
            first = self.fetch(route, {"words": query})
            used += self.sweep_calls - before
            if not first or first["wrong"] != "words" or len(query) <= 3:
                break
            self.text_cut[tool] = min(self.text_cut.get(tool, len(query)), len(query) - 1)
            query = self.fitWords(query.split(), self.textLimit(route))
        if not first or first["wrong"]:
            return {"ids": [], "complete": False, "used": used, "query": query}
        if not self.logic and first["count"] is not None and len(query.split()) >= 2:
            lone = probe if probe in query.split() else query.split()[-1]
            before = self.sweep_calls
            alone = self.fetch(route, {"words": lone})
            used += self.sweep_calls - before
            if alone and alone["count"] is not None and alone["count"] != first["count"]:
                self.logic = "or" if first["count"] > alone["count"] else "and"
        walked = self.walk(route, {"words": query}, max(1, room - used), label, True)
        used += walked["used"]
        complete = walked["ended"] or (walked["full"] and (walked["count"] is None or len(walked["ids"]) >= walked["count"]))
        return {"ids": walked["ids"], "complete": complete and not walked["wrong"], "used": used, "query": query}

    def cheapPool(self, number: int, route: dict[str, Any], label: str, room: int) -> dict[str, Any] | None:
        value = self.sectionValue(route, number, label)
        if not value or room <= 0:
            return None
        shelf = self.host.shelf
        ceiling = self.ceiling(number)
        cap = ceiling if route.get("cap") and ceiling is not None else None
        used = 0

        def ask(limit: float | None) -> dict[str, Any] | None:
            nonlocal used
            before = self.sweep_calls
            got = self.fetch(route, {"section": value, **({"cap": self.capValue(route, limit)} if limit is not None else {})})
            used += self.sweep_calls - before
            return got

        got = ask(cap)
        if got and got["wrong"] == "cap":
            cap = None
            got = ask(None)
        if got and not got["wrong"] and not got["rows"] and cap is not None and used < room:
            bare = ask(None)
            if bare and bare["rows"]:
                return {"ids": [], "complete": got["count"] == 0, "below": cap, "used": used, "label": label}
            got = bare
        tried = [value]
        while (not got or got["wrong"] or not got["rows"]) and used < room and len(tried) < 3:
            other = self.otherSection(number, route, tried)
            if not other:
                break
            tried.append(other)
            value = label = other
            got = ask(cap)
        if not got or got["wrong"] or not got["rows"]:
            return None
        if label != tried[0]:
            self.sections[number] = label
        reach = self.reachOf(route)
        aim = max(1.0, 0.8 * reach)
        high: tuple[float, dict[str, Any]] | None = None
        low: tuple[float, dict[str, Any]] | None = None
        if got["count"] is not None and got["count"] > reach and cap is not None:
            high = (cap, got)
        else:
            low = (cap, got) if cap is not None else None
        sample = sorted(p for p in (shelf.price_of(i) for i in got["rows"]) if p is not None)
        probes = 0
        while high is not None and probes < 3 and used < room:
            top, seen = high
            if low is None:
                if not sample:
                    break
                place = int(round(aim / max(1, seen["count"]) * (len(sample) + 1))) - 1
                pick = sample[max(0, min(len(sample) - 1, place))]
            else:
                bottom, under = low
                seen_top = max([p for p in (shelf.price_of(i) for i in seen["rows"]) if p is not None] or [top])
                span = min(top, max(seen_top, bottom + 1.0)) - bottom
                pick = bottom + span * (aim - (under["count"] or 0)) / max(1, seen["count"] - (under["count"] or 0))
            pick = float(int(pick)) if pick >= 2 else pick
            if pick >= top or (low is not None and pick <= low[0]) or pick <= 0:
                break
            probes += 1
            trial = ask(pick)
            if not trial or trial["wrong"] or trial["count"] is None:
                break
            if trial["count"] > reach:
                high = (pick, trial)
                sample = sorted(p for p in (shelf.price_of(i) for i in trial["rows"]) if p is not None)
            else:
                low = (pick, trial)
                if trial["count"] >= reach / 2:
                    break
        if high is not None and low is None:
            return {"ids": list(got["rows"]), "complete": False, "below": cap, "used": used, "label": label}
        if low is not None:
            cap, got = low
        if got["count"] is not None and got["count"] > reach:
            return {"ids": list(got["rows"]), "complete": False, "below": cap, "used": used, "label": label}
        walked = self.walk(route, {"section": value, **({"cap": self.capValue(route, cap)} if cap is not None else {})}, max(1, room - used))
        used += walked["used"]
        full = walked["full"] and (walked["count"] is None or len(walked["ids"]) >= walked["count"])
        return {"ids": walked["ids"], "complete": full, "below": cap, "used": used, "label": label}

    def otherSection(self, number: int, route: dict[str, Any], tried: list[str]) -> str:
        if isinstance(self.argSpec(route, route["section"]).get("enum"), list):
            return ""
        shelf = self.host.shelf
        key = shelf.category_key()
        item = self.host.book.items[number - 1]
        want = {w for w in Text.stem(item.name) | Text.stem(item.shelf or "") | Text.stem(" ".join(tried)) if len(w) >= 4}
        scored: dict[str, int] = {}
        for ident in shelf.seen_order if key else []:
            label = str((shelf.rows.get(ident) or {}).get(key) or "").strip()
            if label and label not in tried:
                hit = len({w for w in Text.stem(label) if len(w) >= 4} & want)
                if hit:
                    scored[label] = max(scored.get(label, 0), hit)
        return max(scored, key=lambda k: (scored[k], -len(k))) if scored else ""

    def textShelfPool(self, route: dict[str, Any], label: str, room: int) -> dict[str, Any] | None:
        word = self.shelfWord(label)
        if not word or self.orderOf(route["tool"]) != "price" or room <= 0:
            return None
        walked = self.walk(route, {"words": word}, room)
        full = walked["full"] and (walked["count"] is None or len(walked["ids"]) >= walked["count"])
        return {"ids": walked["ids"], "complete": full or walked["top"] is not None, "below": None if full else walked["top"], "used": walked["used"]}

    def buildPool(self, number: int, room: int) -> None:
        start = self.sweep_calls
        label = self.discover(number)
        pool: dict[str, Any] = {"ids": [], "group": [], "complete": False, "below": None, "label": label, "value": False, "queries": [], "pass": None, "verdicts": {}, "verified": False}
        old = self.pools.get(number)
        if old:
            pool["ids"] = list(old["ids"])
            pool["group"] = list(old.get("group") or [])
            pool["verified"] = bool(old.get("verified"))
        text, shelfr = self.textRoute(), self.shelfRoute()
        if label and text:
            got = self.valuePool(number, text, label, max(2, int(0.6 * room)) if shelfr else room - (self.sweep_calls - start))
            if got:
                pool["value"] = True
                pool["queries"] = got["queries"]
                pool["group"] = list(got["ids"])
                pool["ids"] += [i for i in got["ids"] if i not in pool["ids"]]
                pool["complete"] = got["complete"]
        if not pool["complete"] and label:
            left = room - (self.sweep_calls - start)
            got = self.cheapPool(number, shelfr, label, left) if shelfr else (self.textShelfPool(text, label, left) if text else None)
            if got:
                pool["label"] = got.get("label") or label
                pool["ids"] += [i for i in got["ids"] if i not in pool["ids"]]
                if got["complete"]:
                    pool["complete"] = True
                    pool["below"] = got["below"]
                    pool["value"] = False
        if old and old.get("complete") and not pool["complete"]:
            pool["complete"], pool["below"], pool["value"] = True, old.get("below"), old.get("value")
        self.pools[number] = pool

    def candidates(self, number: int) -> list[str]:
        host = self.host
        shelf = host.shelf
        pool = self.pools.get(number)
        key = shelf.category_key()
        if not pool or not pool["label"] or not key:
            return []
        label = pool["label"].lower()
        ceiling = self.ceiling(number)
        group = set(pool.get("group") or [])
        inside = [i for i in pool["ids"] if shelf.kind_of(i, key) == label]
        outside = [i for i in shelf.seen_order if i not in inside and shelf.kind_of(i, key) == label]
        found = [i for i in inside + outside if shelf.buyable(i) and (ceiling is None or shelf.price_of(i) is None or (shelf.price_of(i) or 0.0) <= ceiling + 1e-9)]
        rank = {i: k for k, i in enumerate(inside + outside)}
        near = self.fitNear(number, found)
        return sorted(found, key=lambda i: (i not in near, bool(group) and i not in group, i not in inside, shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"), rank[i]))

    def fitNear(self, number: int, ids: list[str]) -> set[str]:
        host = self.host
        pairs = host.checker.linkedNeeds().get(number) or []
        if not pairs or not ids:
            return set(ids)
        given = getattr(host.checker, "fitOf", None)
        near: set[str] = set()
        for other, _ in pairs:
            mates = [i for i in (self.pools.get(other) or {}).get("ids", []) if host.shelf.buyable(i)][:12]
            for ident in ids:
                said = [given(ident, m) for m in mates] if callable(given) else []
                if True in said or (None in said or not said) and any(host.checker.sharedCodes(ident, m) for m in mates):
                    near.add(ident)
        return near or set(ids)

    @staticmethod
    def holds(words: list[str], text: str) -> bool:
        want = [Text.single(w) for w in words]
        have = Text.bag(text, True)
        return bool(want) and any(have[k:k + len(want)] == want for k in range(len(have) - len(want) + 1))

    def judge(self, ident: str, number: int) -> tuple[bool | None, str]:
        shelf = self.host.shelf
        record = shelf.full.get(ident)
        if not record:
            return None, "not opened"
        if not shelf.buyable(ident):
            return False, "cannot be bought now"
        ceiling = self.ceiling(number)
        price = shelf.price_of(ident)
        if ceiling is not None and price is not None and price > ceiling + 1e-9:
            return False, "over the budget"
        block = Shelf.factBlock(record) or {}
        loose = [v for v in record.values() if isinstance(v, str) and len(v) <= 160 and len(v.split()) <= 16] + [str(v) for v in block.values()]
        open_needs: list[str] = []
        for need in self.needsOf(number):
            kind = self.kinds.get(need) or {}
            prop = need.split(":", 1)[0] if ":" in need else ""
            keys = [k for k in block if prop and Text.names(prop, k)]
            if kind.get("kind") == "range":
                said = [Coverage.inBounds(k, block[k], kind["bounds"]) for k in keys]
                if True in said:
                    continue
                if False in said:
                    return False, f"{need} (its {keys[said.index(False)]}: {str(block[keys[said.index(False)]])[:40]})"
                open_needs.append(need)
                continue
            if kind.get("kind") == "not":
                hits = [k for k in keys if any(Coverage.holds(words, str(block[k])) for words in kind["values"])]
                if hits:
                    return False, f"{need} (its {hits[0]}: {str(block[hits[0]])[:40]})"
                if not keys:
                    open_needs.append(need)
                continue
            if kind.get("kind") != "equals":
                open_needs.append(need)
                continue
            texts = [str(block[k]) for k in keys] if keys else loose
            if any(Coverage.holds(words, text) for words in kind["values"] for text in texts):
                continue
            if keys:
                return False, f"{need} (its {keys[0]}: {str(block[keys[0]])[:40]})"
            open_needs.append(need)
        return (None, "; ".join(open_needs)) if open_needs else (True, "")

    def shows(self, ident: str, number: int) -> bool:
        block = Shelf.factBlock(self.host.shelf.full.get(ident) or {}) or {}
        named = [need for need in self.needsOf(number) if (self.kinds.get(need) or {}).get("kind") == "equals"]
        for need in named:
            prop = need.split(":", 1)[0] if ":" in need else ""
            keys = [k for k in block if prop and Text.names(prop, k)]
            if not any(Coverage.holds(words, str(block[k])) for words in self.kinds[need]["values"] for k in keys):
                return False
        return bool(named)

    def openOne(self, ident: str) -> bool:
        host = self.host
        if ident in host.shelf.full:
            return True
        before = self.host.step
        host.shelf.openMore([ident], self.reserve())
        self.sweep_calls += host.step - before
        return ident in host.shelf.full

    def readLine(self, number: int, room: int, most: int) -> list[str]:
        pool = self.pools.get(number)
        if not pool or room <= 0:
            return []
        opened: list[str] = []
        unsure = 0
        for ident in self.candidates(number):
            if self.host.finished or len(opened) >= most or room <= 0 or (ident not in self.host.shelf.full and self.sweep_calls >= self.allowance()):
                break
            if ident not in self.host.shelf.full:
                if not self.openOne(ident):
                    break
                room -= 1
                opened.append(ident)
            verdict = self.judge(ident, number)
            pool["verdicts"][ident] = verdict
            if ident in (pool.get("group") or []) and self.shows(ident, number):
                pool["verified"] = True
            if verdict[0] is True:
                pool["pass"] = ident
                break
            if verdict[0] is None:
                unsure += 1
                if unsure >= 3:
                    break
        pool["early"] = pool["pass"] is None and len(opened) >= most and unsure < 3
        return opened

    def allowance(self) -> int:
        host = self.host
        msgs = max(0, host.book.msgs_free()) if host.store.roles.get("ask") and not host.asked else 0
        room = host.max_steps - 2 * msgs - 4 * len(host.book.items) - 8
        return max(10, min((3 * host.max_steps) // 5, room))

    def most(self) -> int:
        return 12 if self.host.max_steps >= 90 else 5

    def signature(self, number: int) -> str:
        refs = sorted((k, v) for k, v in self.host.lookups.ref_values.items())
        return render([self.needsOf(number), self.host.book.budget, refs])

    def sweep(self) -> str:
        if not SWEEP:
            return ""
        try:
            return self.sweepLines()
        except Exception as exc:
            self.host.trace.append({"kind": "sweep_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def sweepLines(self) -> str:
        host = self.host
        if self.swept or host.finished or host.broke:
            return ""
        self.swept = True
        items = host.book.items
        if not items or not self.routes():
            return ""
        start = host.step
        self.learnKinds()
        room = min(host.left() - self.reserve() - 4, self.allowance())
        if room < 2 * len(items):
            return ""
        per = room // len(items)
        for number in range(1, len(items) + 1):
            self.buildPool(number, max(2, min(per - 2, 12)))
        early = bool(host.store.roles.get("ask")) and host.book.msgs_free() > 0 and not host.asked
        for number in range(1, len(items) + 1):
            left = host.left() - self.reserve() - 4
            self.readLine(number, left // max(1, len(items) - number + 1), 2 if early else self.most())
        for number in range(1, len(items) + 1):
            self.marks[number] = self.signature(number)
        host.trace.append({"kind": "sweep", "calls": host.step - start, "logic": self.logic, "orders": {t: self.orderOf(t) for t in self.order_votes},
                           "pools": {str(n): self.poolBrief(n) for n in self.pools}})
        return self.poolText(list(range(1, len(items) + 1)), host.step - start, False)

    def refresh(self) -> str:
        if not SWEEP:
            return ""
        try:
            return self.refreshLines()
        except Exception as exc:
            self.host.trace.append({"kind": "sweep_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def refreshLines(self) -> str:
        host = self.host
        if not self.swept or not self.pools or host.finished or host.broke or self.refreshes >= 4 or host.budgets.closing():
            return ""
        changed = [n for n in sorted(self.pools) if n <= len(host.book.items) and (self.marks.get(n) != self.signature(n) or (host.asked and self.pools[n].get("early")))]
        if not changed:
            return ""
        for number in changed:
            self.marks[number] = self.signature(number)
        if host.left() - self.reserve() < 3:
            return ""
        self.refreshes += 1
        start = host.step
        self.learnKinds()
        for number in changed:
            room = (host.left() - self.reserve() - 2) // max(1, len(changed))
            pool = self.pools[number]
            pool["pass"] = None
            pool["early"] = False
            pool["verdicts"] = {}
            queried = " ".join(pool.get("queries") or []).split()
            fresh = any(not all(w in queried for w in kind["values"][0]) for kind in (self.kinds.get(n) or {} for n in self.needsOf(number)) if kind.get("kind") == "equals" and kind.get("values"))
            if room > 2 and (not pool["complete"] or (pool["below"] is not None and fresh)):
                self.buildPool(number, max(2, room // 2))
            self.readLine(number, (host.left() - self.reserve() - 2) // max(1, len(changed)), self.most())
        host.trace.append({"kind": "sweep_again", "items": changed, "calls": host.step - start, "pools": {str(n): self.poolBrief(n) for n in changed}})
        return self.poolText(changed, host.step - start, True)

    def poolBrief(self, number: int) -> dict[str, Any]:
        pool = self.pools.get(number) or {}
        shelf = self.host.shelf
        return {"label": pool.get("label"), "n": len(pool.get("ids") or []), "complete": pool.get("complete"), "below": pool.get("below"), "value": pool.get("value"), "verified": pool.get("verified"),
                "queries": pool.get("queries"), "pass": pool.get("pass"), "read": [i for i in pool.get("ids") or [] if i in shelf.full][:12],
                "whole": self.covered(number), "ids": list(pool.get("ids") or [])[:150]}

    def covered(self, number: int, price: float | None = None) -> bool:
        pool = self.pools.get(number) or {}
        if not pool.get("complete"):
            return False
        below = pool.get("below")
        worded = bool(pool.get("value")) and below is None
        if price is not None:
            return (not worded or bool(pool.get("verified"))) and (below is None or price <= below + 1e-9)
        if worded:
            return False
        ceiling = self.ceiling(number)
        return below is None or (ceiling is not None and below >= ceiling - 1e-9)

    def pool(self, number: int) -> dict[str, Any]:
        pool = self.pools.get(number) or {}
        shelf = self.host.shelf
        ids = sorted(pool.get("ids") or [], key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))
        whole = self.covered(number)
        return {"ids": ids, "complete": whole, "below": None if whole or not pool.get("complete") else pool.get("below"), "label": pool.get("label") or "", "value": bool(pool.get("value")),
                "candidates": self.candidates(number) if pool else [], "read": [i for i in ids if i in shelf.full], "pass": pool.get("pass"),
                "verdicts": {i: v[0] for i, v in (pool.get("verdicts") or {}).items()}}

    def poolText(self, numbers: list[int], calls: int, again: bool) -> str:
        host = self.host
        shelf = host.shelf
        many = len(host.book.items) > 1
        head = (f"[Act read the pools again for the changed requirements ({calls} steps)]" if again
                else f"[Act searched the store itself before your first move ({calls} steps). Result rows are summaries; the attributes below come from the listings' own pages]")
        lines = [head]
        shown = 0
        for number in numbers:
            pool = self.pools.get(number)
            if not pool:
                continue
            if not pool["label"]:
                lines.append(f"item {number}: no category found for it yet; search for it yourself.")
                continue
            if pool["complete"] and pool["below"] is None and pool["value"]:
                scope = "the search for the stated values ended inside this pool" + (f" ({'; '.join(pool['queries'])})" if pool["queries"] else "") + ", but listings that state them in other words may be missing"
            elif pool["complete"] and pool["below"] is None:
                scope = "every listing of this category is in this pool" + (f" (searched: {'; '.join(pool['queries'])})" if pool["queries"] else "")
            elif pool["complete"]:
                scope = f"every listing of this category priced up to {pool['below']:g} is in this pool"
            else:
                scope = "this pool may miss listings: search further when none of it fits"
            cands = self.candidates(number)
            listed = ", ".join(f"{i} ({shelf.price_of(i):g})" if shelf.price_of(i) is not None else i for i in cands[:10])
            lines.append((f"item {number}" if many else "item") + f" (category \"{pool['label']}\"): {scope}. Buyable within the budget, cheapest first: " + (listed or "none") + ".")
            for ident, verdict in pool["verdicts"].items():
                said = ("meets every stated value by its attributes" if verdict[0] is True else f"fails {verdict[1]}" if verdict[0] is False
                        else f"to judge from its record: {verdict[1]}")
                lines.append(f"  {ident}: {said}")
                if ident in shelf.full and shown < 8:
                    shown += 1
                    lines.append("    record: " + render(shelf.full[ident], 160)[:1100])
        lines.append("Use these pools: judge the opened listings, open the next ones in price order when they fail, and search again only for what a pool may miss.")
        return "\n".join(lines)

    def poolCheaper(self) -> list[str]:
        host = self.host
        shelf = host.shelf
        if host.store.brief.get("pick") != "cheapest" or not self.pools or host.finished:
            return []
        mark = shelf.cart_mark() + f"#{host.book.version}"
        if mark in self.pool_marks:
            return []
        self.pool_marks.add(mark)
        key = shelf.category_key()
        texts: list[str] = []
        for line in shelf.cart_ids():
            price = shelf.price_of(line)
            kind = shelf.kind_of(line, key)
            number = next((n for n, p in sorted(self.pools.items()) if p.get("label") and p["label"].lower() == kind), 0)
            if not number or price is None or not self.covered(number, price):
                continue
            cheaper = [i for i in self.candidates(number) if i != line and i not in shelf.full and (shelf.price_of(i) if shelf.price_of(i) is not None else float("inf")) < price]
            opened: list[str] = []
            unsure = 0
            for ident in cheaper[:6]:
                if host.left() - host.book.tail() - len(host.book.items) - 3 <= 0 or host.finished or unsure >= 3:
                    break
                if not self.openOne(ident):
                    break
                opened.append(ident)
                verdict = self.judge(ident, number)
                self.pools[number]["verdicts"][ident] = verdict
                if verdict[0] is True:
                    break
                unsure += 1 if verdict[0] is None else 0
            if opened:
                said = [f"{i} ({shelf.price_of(i):g}): " + ("meets every stated value by its attributes" if self.pools[number]["verdicts"][i][0] is True
                        else "fails " + self.pools[number]["verdicts"][i][1] if self.pools[number]["verdicts"][i][0] is False else "to judge: " + self.pools[number]["verdicts"][i][1]) for i in opened]
                texts.append(f"cheaper listings of item {number}'s complete pool that were unread, opened now: " + "; ".join(said))
        if texts:
            host.trace.append({"kind": "pool_cheaper", "texts": [t[:300] for t in texts]})
        return texts

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

    def staleQuery(self, entry: dict[str, Any]) -> bool:
        said = set(re.findall(r"[a-z0-9]+", " ".join(str(v) for v in (entry.get("args") or {}).values() if isinstance(v, str)).lower()))
        for item in self.host.book.items:
            for need in item.dropped:
                value = need.split(":", 1)[1] if ":" in need else need
                words = {w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 2}
                if words and words <= said:
                    return True
        return False

    def best_found(self, kind: str, key: str | None) -> dict[str, Any] | None:
        best = None
        for entry in reversed(self.found):
            if self.staleQuery(entry):
                continue
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
        return self.poolCheaper() + self.coverFound()

    def coverFound(self) -> list[str]:
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
        self.codes: dict[str, dict[str, Any]] = {}
        self.fit_tried: set[str] = set()

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
            cap = self.host.book.items[line["item"] - 1].cap if line["item"] else None
            if cap is not None and line["price"] is not None and line["price"] > cap:
                problems.append(f"{line['id']}: its price {line['price']:g} is over the {cap:g} the shopper allows for one {self.host.book.items[line['item'] - 1].name}")
            if shelf and line["label"] and shelf.strip().lower() != line["label"].lower():
                problems.append(f"{line['id']}: its label '{line['label']}' is not the item's store label '{shelf}'")
        if not lead:
            covered = {line["item"] for line in lines}
            for number, item in enumerate(self.host.book.items, 1):
                if number not in covered:
                    problems.append(f"no cart line for item {number} ({item.name})")
            per: dict[int, list[str]] = {}
            for line in lines:
                if line["item"]:
                    per.setdefault(line["item"], []).append(line["id"])
            for line in lines:
                if not line["item"]:
                    problems.append(f"{line['id']}: it matches no requested item; each requested item is ONE listing bought in its quantity, so remove this line, or raise the quantity on the item's own line instead of adding a second listing")
            for number, ids in per.items():
                if len(ids) > 1:
                    item = self.host.book.items[number - 1]
                    problems.append(f"item {number} ({item.name}) has {len(ids)} different listings in the cart ({', '.join(ids)}); it must be ONE listing with quantity {item.qty}")
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

    @staticmethod
    def asLines(got: Any) -> dict[str, Any] | None:
        if isinstance(got, dict) and isinstance(got.get("lines"), list) and got["lines"]:
            return got
        if isinstance(got, dict) and str(got.get("id") or "").strip() and isinstance(got.get("grades"), list):
            return {"lines": [got]}
        return None

    @staticmethod
    def linesRead(got: dict[str, Any] | None) -> int:
        return len([e for e in Text.items(got.get("lines")) if isinstance(e, dict)]) if isinstance(got, dict) else 0

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
        got = Checker.asLines(self.host.models.ask_json(prompt, [LIGHT], 9000, "medium"))
        if Checker.linesRead(got) < len(lines):
            again = Checker.asLines(self.host.models.ask_json(prompt, [HEAVY, TOP], 9000, "medium"))
            if Checker.linesRead(again) > Checker.linesRead(got):
                got = again
        blind = got is None
        self.placeLines(lines, got, lead)
        coded = self.codeProblems(lines, lead)
        said, better, disputed, hard = self.gradeVerdict(got, lines, names, cheaper)
        if blind and not lead:
            said.append("the review of this cart returned no reading, so no line is shown to meet its requirements yet; re-open the cart's listings, compare each with every requirement, then check out")
        second = None
        if not coded and said and len(said) == len(disputed):
            asked = ", ".join(f"{tag} on {ident}" for ident, tag in disputed)
            again = base + "\n\nA first reading graded " + asked + " as absent, contradicted or not shown as one unbroken phrase. Re-read those records in full before grading, and quote the field that states each value when one does."
            second = self.host.models.ask_json(again, [LIGHT], 9000, "medium")
            if not (second and all(self.resolved(second, ident, tag, names.get(tag, "")) for ident, tag in disputed)):
                second = self.host.models.ask_json(again, [HEAVY, LIGHT], 3500, "medium")
            if second and all(self.resolved(second, ident, tag, names.get(tag, "")) for ident, tag in disputed):
                said = self.labelTurned(got, second, lines)
                hard = False
        problems = coded + said
        if not lead and not problems:
            cheaper = self.planProblem(lines)
            problems += cheaper
            better += [i for i in self.__dict__.get("plan_better", []) if cheaper and i not in better]
        flags = self.lineFlags(lines, said, got) if problems and not blind else {}
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
        coded = self.codeFits(lines)
        text = ""
        if out:
            text = ("\n\nCODES FOUND IN BOTH RECORDS (a string match made by code, shown for the requirements that name another item: such a requirement is met only when a record states the fit,"
                    " the model or the condition with that code and no exception it names excludes the other line):\n" + "\n".join(out[:8]))
        if coded:
            text += ("\n\nFIT BY THE CODES EACH RECORD STATES (read from the model code of one line and the fit list, family pattern, exceptions and part pages of the other):\n" + "\n".join(coded[:8]))
        return text

    @staticmethod
    def squash(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    @staticmethod
    def famMatch(pattern: str, code: str) -> bool:
        stem = pattern.rstrip("x")
        return 2 <= len(stem) < len(pattern) and len(code) == len(pattern) and code.startswith(stem)

    def codeRecord(self, ident: str) -> str:
        record = self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident) or {}
        return render({k: v for k, v in record.items() if v != ident} if isinstance(record, dict) else record, 300)[:900]

    @staticmethod
    def keepCodes(values: Any, text: str) -> list[str]:
        flat = Checker.squash(text)
        out: list[str] = []
        for value in (values if isinstance(values, list) else [values]):
            code = Checker.squash(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
            if len(code) >= 3 and any(c.isdigit() for c in code) and code in flat and code not in out:
                out.append(code)
        return out

    def readCodes(self, ids: list[str]) -> None:
        host = self.host
        codes = self.__dict__.setdefault("codes", {})
        todo = [i for i in dict.fromkeys(ids) if i not in codes and (host.shelf.full.get(i) or host.shelf.rows.get(i))]
        for start in range(0, len(todo), 25):
            if host.models.no_credit or host.budgets.closing() or host.models.cost > host.budgets.spendHard * 1.4:
                return
            chunk = todo[start:start + 25]
            texts = {i: self.codeRecord(i) for i in chunk}
            got = host.models.ask_json(FITS + "\n\nRECORDS:\n" + "\n".join(f"{i}: {t}" for i, t in texts.items()), [LIGHT, HEAVY], 5000)
            said = {str(e.get("id") or "").strip(): e for e in Text.items((got or {}).get("codes")) if isinstance(e, dict)}
            for i in chunk:
                entry, text = said.get(i) or {}, texts[i]
                also = []
                for pair in Text.items(entry.get("also")):
                    if isinstance(pair, list) and len(pair) >= 2:
                        part, fits = Checker.keepCodes(pair[0], text), Checker.keepCodes(pair[1], text)
                        if part and fits:
                            also.append((part[0], fits))
                codes[i] = {"model": (Checker.keepCodes(entry.get("model"), text) or [""])[0], "fits": Checker.keepCodes(entry.get("fits"), text),
                            "families": Checker.keepCodes(entry.get("families"), text), "except": Checker.keepCodes(entry.get("except"), text),
                            "part": (Checker.keepCodes(entry.get("part"), text) or [""])[0], "see": Checker.keepCodes(entry.get("see"), text),
                            "see_raw": [str(s).strip() for s in Text.items(entry.get("see")) if isinstance(s, str) and Checker.squash(s) in Checker.squash(text)][:3], "also": also}
            host.trace.append({"kind": "fit_codes", "read": len(chunk), "with_codes": sum(1 for i in chunk if any(codes[i][k] for k in ("model", "fits", "families", "part")))})

    def claimsFor(self, part: str) -> set[str]:
        codes = self.__dict__.setdefault("codes", {})
        return {c for entry in codes.values() for owner, fits in entry["also"] if owner == part for c in fits}

    def fitOf(self, acc: str, anchor: str) -> bool | None:
        codes = self.__dict__.setdefault("codes", {})
        mine, theirs = codes.get(acc), codes.get(anchor)
        if not mine or not theirs or not theirs["model"]:
            return None
        model = theirs["model"]
        if model in mine["except"]:
            return False
        if model in mine["fits"] or (mine["part"] and model in self.claimsFor(mine["part"])):
            return True
        if any(Checker.famMatch(p, model) for p in mine["families"]):
            return True
        if mine["see"] and not any(e["part"] in mine["see"] for e in codes.values()):
            return None
        return False if (mine["fits"] or mine["families"] or mine["see"]) else None

    def fitPools(self) -> tuple[dict[int, list[int]], dict[int, list[str]]]:
        host = self.host
        shelf = host.shelf
        links = {n: sorted({o for o, _ in pairs}) for n, pairs in self.linkedNeeds().items()}
        catKey = shelf.category_key()
        if not links or not catKey:
            return {}, {}
        talk = Interviewer(host)
        group = sorted(set(links) | {o for others in links.values() for o in others})
        pools: dict[int, list[str]] = {}
        for number in group:
            label = Text.words(talk.itemLabel(number))
            pool = [i for i in shelf.seen_order if Text.words(str((shelf.rows.get(i) or {}).get(catKey) or "")) == label and shelf.buyable(i) and shelf.price_of(i) is not None]
            pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
            pools[number] = pool[:50]
        return links, pools

    def fitPlans(self, limit: int = 12) -> list[dict[str, Any]]:
        host = self.host
        shelf = host.shelf
        links, pools = self.fitPools()
        if not links or any(not pool for pool in pools.values()):
            return []
        self.readCodes([i for pool in pools.values() for i in pool])
        codes = self.__dict__.setdefault("codes", {})
        anchors = [n for n in pools if n not in links]
        if not anchors or not all(any((codes.get(i) or {}).get("model") for i in pools[n]) for n in anchors):
            return []
        qty = {n: host.book.items[n - 1].qty for n in pools}
        choices: list[dict[int, str]] = [{}]
        for number in anchors:
            choices = [{**c, number: i} for c in choices for i in pools[number] if (codes.get(i) or {}).get("model")][:400]
        plans: list[dict[str, Any]] = []
        for choice in choices:
            plan = dict(choice)
            alts: dict[int, list[str]] = {}
            unknown = 0
            maybe: list[str] = []
            for number, others in links.items():
                fitting = [i for i in pools[number] if all(self.fitOf(i, plan[o]) in (True, None) for o in others if o in plan)
                           and all(self.fitOf(i, plan[o]) is True or (codes.get(i) or {}).get("see") for o in others if o in plan)]
                unknown += sum(1 for i in pools[number] if any(self.fitOf(i, plan[o]) is None for o in others if o in plan))
                if not fitting:
                    break
                plan[number] = fitting[0]
                alts[number] = fitting[1:4]
                if any(self.fitOf(fitting[0], plan[o]) is None for o in others if o in plan):
                    maybe.append(fitting[0])
            else:
                if len(set(plan.values())) < len(plan):
                    continue
                total = sum((shelf.price_of(i) or 0.0) * qty[n] for n, i in plan.items())
                plans.append({"items": plan, "lines": {i: qty[n] for n, i in plan.items()}, "total": total, "alts": alts, "unknown": unknown, "maybe": maybe,
                              "model": {n: codes[plan[n]]["model"] for n in anchors}})
        plans.sort(key=lambda p: (p["total"], sorted(p["lines"])))
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for plan in plans:
            mark = "|".join(sorted(plan["lines"]))
            if mark not in seen:
                seen.add(mark)
                out.append(plan)
        return out[:limit]

    def fit_options(self) -> list[dict[str, Any]]:
        out = []
        for plan in self.fitPlans():
            key = self.check_key(list(plan["lines"]))
            out.append({**plan, "checked": key in self.checked, "ok": self.verdicts.get(key)})
        return out

    def fit_untried(self) -> list[dict[str, Any]]:
        return [p for p in self.fit_options() if not p["checked"] and self.inBudget(p["lines"])]

    def fit_scarce(self) -> list[int]:
        links, pools = self.fitPools()
        if not links:
            return []
        self.readCodes([i for pool in pools.values() for i in pool])
        anchors = [n for n in pools if n not in links]
        counts: dict[int, int] = {}
        for number, others in links.items():
            counts[number] = sum(1 for i in pools[number] for o in others if o in anchors for a in pools[o] if self.fitOf(i, a) is True)
        return sorted(counts, key=lambda n: (counts[n], n))

    def seePart(self, acc: str) -> bool:
        host = self.host
        codes = self.__dict__.setdefault("codes", {})
        entry = codes.get(acc) or {}
        opened = False
        for raw, part in zip(entry.get("see_raw") or [], entry.get("see") or []):
            if host.finished or host.left() <= host.book.tail() + len(host.book.items) + 3:
                break
            owners = [i for i, e in codes.items() if e["part"] == part and i != acc]
            if not owners:
                finder = host.store.roles.get("search")
                field = host.lookups.queryKey(finder) if finder else None
                found = host.lookups.lookFor(finder, field, raw, None) if field else None
                if not found:
                    continue
                rows = [i for i, _ in found[0] if part in Checker.squash(self.codeRecord(i))]
                self.readCodes(rows[:6])
                owners = [i for i in rows if (codes.get(i) or {}).get("part") == part]
            for owner in owners[:1]:
                if owner not in host.shelf.opened and host.left() > host.book.tail() + len(host.book.items) + 2:
                    opened = host.lookups.openPage(owner) or opened
                    host.trace.append({"kind": "see_part", "acc": acc, "part": part, "page": owner})
        return opened

    def codeFits(self, lines: list[dict[str, Any]]) -> list[str]:
        links = {n: sorted({o for o, _ in pairs}) for n, pairs in self.linkedNeeds().items()}
        held = {line["item"]: line["id"] for line in lines if line["item"]}
        if not links:
            return []
        self.readCodes(list(held.values()))
        out: list[str] = []
        for number, others in links.items():
            acc = held.get(number)
            if not acc:
                continue
            self.seePart(acc)
            for other in others:
                anchor = held.get(other)
                verdict = self.fitOf(acc, anchor) if anchor else None
                if verdict is None:
                    continue
                model = (self.__dict__.setdefault("codes", {}).get(anchor) or {}).get("model", "")
                out.append(f"{acc} (item {number}) and {anchor} (item {other}, model code {model}): " + ("the codes say it fits" if verdict else "the codes say it does not fit"))
        return out

    def codePlan(self, lines: dict[str, int], held: dict[int, str], group: list[int]) -> dict[str, int] | None:
        host = self.host
        mine = {held[n] for n in group if n in held}
        tried = self.__dict__.setdefault("fit_tried", set())
        looked: set[str] = set()
        checks = 0
        for _ in range(8):
            plan = self.nextPlan(lines, mine, tried, looked)
            if plan is None or checks >= 3 or host.finished:
                return None
            trial, mark, maybe = plan
            if maybe:
                looked.update(maybe)
                for ident in maybe:
                    self.seePart(ident)
                continue
            tried.add(mark)
            checks += 1
            if host.left() <= len(trial) + 5:
                return None
            host.shelf.openMore([i for i in trial if i not in host.shelf.full], len(trial) + 4)
            self.check_cart(list(trial), host.budgets.spendHard * 1.3, False)
            verdict = self.verdicts.get(self.check_key(list(trial)))
            host.trace.append({"kind": "fit_plan", "plan": trial, "ok": verdict})
            if verdict:
                return trial
        return None

    def nextPlan(self, lines: dict[str, int], mine: set[str], tried: set[str], looked: set[str]) -> tuple[dict[str, int], str, list[str]] | None:
        for plan in self.fitPlans(8):
            trial = {**{i: q for i, q in lines.items() if i not in mine}, **plan["lines"]}
            mark = self.check_key(list(trial))
            maybe = [i for i in plan.get("maybe") or [] if i not in looked]
            if trial == lines or mark in tried or (plan.get("maybe") and not maybe) or not self.inBudget(trial) or not self.keepsKinds(lines, trial):
                continue
            return trial, mark, maybe
        return None

    def planProblem(self, lines: list[dict[str, Any]]) -> list[str]:
        host = self.host
        self.plan_better = []
        if host.store.brief.get("pick") != "cheapest" or host.budgets.closing() or not lines or any(line["price"] is None for line in lines) or not self.linkedNeeds():
            return []
        held = {line["item"]: line["id"] for line in lines if line["item"]}
        cart = {line["id"]: line["have"] for line in lines}
        total = sum(line["price"] * line["have"] for line in lines)
        tried = self.__dict__.setdefault("fit_tried", set())
        checks = 0
        for plan in self.fitPlans(6):
            mine = {held[n] for n in plan["items"] if n in held}
            trial = {**{i: q for i, q in cart.items() if i not in mine}, **plan["lines"]}
            cost = sum((host.shelf.price_of(i) or 0.0) * q for i, q in trial.items())
            mark = self.check_key(list(trial))
            if trial == cart or cost >= total - 1e-9 or plan.get("maybe") or mark in tried or not self.inBudget(trial) or not self.keepsKinds(cart, trial):
                continue
            if mark not in self.checked:
                if checks >= 2 or host.left() <= len(trial) + 6 or host.models.cost > host.budgets.spendHard * 1.3:
                    return []
                checks += 1
                saved = (self.problems, self.better, dict(self.altFor))
                host.shelf.openMore([i for i in trial if i not in host.shelf.full], len(trial) + 4)
                self.check_cart(list(trial), host.budgets.spendHard * 1.3, False)
                self.problems, self.better, self.altFor = saved
            if not self.verdicts.get(mark):
                tried.add(mark)
                continue
            self.plan_better = [i for i in plan["lines"] if i not in cart]
            host.trace.append({"kind": "fit_plan", "plan": trial, "ok": True, "cheaper_than": total})
            return [f"{', '.join(sorted(trial))} also fit together by the fit codes their records state, pass this same check and cost {cost:g} in all against {total:g}: put that set in the cart before buying"]
        return []

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
        coded = self.codePlan(lines, held, group)
        if coded is not None:
            return coded
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
    rule: dict[str, Any] | None = None
    graded: dict[str, tuple[int, int, list[str]]] = {}
    route_rounds = 0
    search_held = False
    clean_tried: set[str] = set()
    short_failed: set[str] = set()
    named: list[str] = []
    EACH = 8
    ALL = 24
    BATCH = 12
    ENDS = """Read what the shopper's request says about ending without a purchase when nothing fits. Return ONE JSON object and nothing else: {"name": true when the request asks to offer, name, suggest or put forward some listing in that case (such as the closest one, or the one that comes up short on the fewest counts), false when it says to suggest or name nothing, null when it says nothing about it, "count": how many listings it asks to put forward (1 unless it states another number), "band": the percent above the cheapest such listing that the request still accepts, as a number (for example 15 for "up to 15% more"), or null when it states none, "quote": the request's own words that state this, copied exactly, or ""}"""
    SHORT = """A shopping agent may end without buying; it must then put forward the listing that comes up short on the fewest requirements. Grade each listing below against each numbered requirement of the item it stands under, from that listing's own record only, with these reading rules: """ + READING + """
A requirement is "met" only when the record's own words state it: give the record's words that state it, copied exactly as the record shows them (the value, or the key and its value). Every other case is "short": the record states another value, or says nothing that decides it. Price, budget and stock are counted elsewhere: do not grade them.
Return ONE JSON object and nothing else: {"rows": [[listing id, requirement tag, "met" or "short", the record's words for a met requirement, else ""]]} with one row for each listing and each requirement of its item."""

    def __init__(self, host: Any):
        self.host = host
        self.rule: dict[str, Any] | None = None
        self.graded: dict[str, tuple[int, int, list[str]]] = {}
        self.route_rounds = 0
        self.search_held = False
        self.clean_tried: set[str] = set()
        self.short_failed: set[str] = set()
        self.named: list[str] = []
        self.audit_runs = 0
        self.audit_seen = -1
        self.audit_version = -1
        self.decline_proven = False
        self.near_holds = 0
        self.force_buy = False
        self.decline_reads = 0
        self.audited = False
        self.rowsHeld = False
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.completed_at = -1

    def freshen(self) -> None:
        now = self.host.book.version
        if self.completed and self.completed_at != now:
            self.completed = False
            self.host.trace.append({"kind": "ending_reopened", "version": now})
        if self.decline_proven and self.audit_version != now:
            self.decline_proven = False

    def audit(self, near: bool = False, late: bool = False) -> str:
        if self.audit_runs >= 2 or (self.audit_runs and len(self.host.shelf.full) == self.audit_seen and self.host.book.version == self.audit_version) or self.host.models.cost > self.room(near or late):
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
            if not (short or unread) or self.host.left() <= len(items) + 3 or self.host.models.cost > self.room(near or late):
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

    def room(self, wide: bool) -> float:
        hard = self.host.budgets.spendHard
        if self.openWork():
            return hard * (1.3 if wide else 1.0)
        return Budgets.auditRoom(hard, wide, len(self.host.book.items), Budgets.held(self.host))

    def chainsOpen(self) -> list[str]:
        host = self.host
        live = {need for _, need in host.book.needs_now()}
        return [need for _, need in host.lookups.refs if need in live and need not in host.lookups.ref_values]

    def numberOf(self, ident: str) -> int:
        entry = self.graded.get(ident)
        if entry:
            return entry[1]
        number = next((Text.count(n) for n, ids in self.lead_items.items() if ident in ids and Text.count(n)), None) or self.host.shelf.item_of(ident)
        return number or (1 if len(self.host.book.items) == 1 else 0)

    def countOf(self, ident: str) -> int | None:
        host = self.host
        entry = self.graded.get(ident)
        if not entry or entry[0] != host.book.version or not 1 <= entry[1] <= len(host.book.items):
            return None
        shelf = host.shelf
        count = len(entry[2]) + (0 if shelf.buyable(ident) else 1)
        price = shelf.price_of(ident)
        budget = host.book.budget
        if price is not None and isinstance(budget, (int, float)) and not isinstance(budget, bool) and price * host.book.items[entry[1] - 1].qty > budget:
            count += 1
        return count

    def fairShare(self) -> int:
        host = self.host
        return max(host.book.tail() + 2 * len(host.book.items) + 6, host.max_steps // 4)

    def workLeft(self) -> str:
        host = self.host
        pending = self.chainsOpen()
        why = ""
        if pending and host.left() > host.book.tail() + 3 * len(pending) + 4 and host.lookups.room(2):
            why = "chain"
        elif host.left() > self.fairShare():
            why = "steps"
        if why:
            host.trace.append({"kind": "closeout_held", "why": why, "left": host.left()})
        return why

    def provenNone(self) -> bool:
        if self.decline_proven:
            return True
        host = self.host
        items = host.book.items
        table = self.shortTable()
        failing = [n for n in range(1, len(items) + 1) if not (table.get(n) and table[n][0][0] == 0)]
        if not failing:
            return False
        for number in failing:
            pool = self.poolOf(number)
            if pool is None or not pool[1] or any(i not in host.shelf.full and host.shelf.buyable(i) for i in pool[0]):
                return False
        return True

    def orderBest(self) -> str:
        host = self.host
        items = host.book.items
        self.freshen()
        if self.completed or host.finished or not host.store.roles.get("cart") or not host.store.roles.get("buy") or host.left() < len(items) + 2:
            return ""
        table = self.shortTable()
        groups: dict[int, list[str]] = {}
        for number in range(1, len(items) + 1):
            rows = [r for r in table.get(number, []) if host.shelf.buyable(r[2])]
            if not rows:
                return ""
            groups[number] = [i for c, _, i in rows if c == rows[0][0]]
        saved = (dict(self.lead_items), list(self.leads), self.confirmed, self.force_buy, self.loose)
        for lines in self.plans(groups, 1):
            key = host.checker.check_key(list(lines))
            if key in self.clean_tried or host.finished or host.left() < len(items) + 2 or host.models.cost > host.budgets.spendHard * 1.5:
                continue
            self.clean_tried.add(key)
            self.lead_items = {str(self.numberOf(i)): [i] for i in lines}
            self.leads = list(lines)
            self.confirmed = next(iter(lines))
            self.force_buy = False
            self.loose = frozenset()
            host.trace.append({"kind": "order_best", "lines": lines})
            buys = host.buys
            done = self.complete_cart()
            if done or host.buys != buys or host.finished:
                return done
            self.lead_items, self.leads, self.confirmed, self.force_buy, self.loose = saved
            self.completed = False
        return ""

    def closeOrder(self) -> str:
        host = self.host
        for _ in range(2):
            if host.finished or host.left() <= len(host.book.items) + host.book.tail() + 6 or not self.proof():
                break
        done = self.orderClean()
        if done or host.finished:
            return done
        if self.provenNone():
            host.trace.append({"kind": "closeout_decline", "proven": True})
            return ""
        return self.orderBest()

    def lateOrder(self) -> str:
        if self.host.finished or self.provenNone():
            return ""
        return self.orderBest()

    def openWork(self) -> bool:
        host = self.host
        items = host.book.items
        pending = self.chainsOpen()
        if pending and host.left() > 3 * len(pending) + 4:
            return True
        if host.left() <= len(items) + 4 or not items:
            return False
        groups: dict[int, list[str]] = {}
        for number, ids in self.lead_items.items():
            if Text.count(number) and number not in self.loose:
                groups.setdefault(Text.count(number), []).extend(i for i in ids if not host.shelf.sold_out(i))
        for ident in self.graded:
            if self.countOf(ident) == 0 and ident not in groups.get(self.graded[ident][1], []):
                groups.setdefault(self.graded[ident][1], []).append(ident)
        if not all(groups.get(n) for n in range(1, len(items) + 1)):
            return False
        return any(host.checker.check_key(list(lines)) not in host.checker.checked for lines in self.plans(groups, 6))

    def plans(self, groups: dict[int, list[str]], limit: int) -> list[dict[str, int]]:
        host = self.host
        shelf = host.shelf
        items = host.book.items
        combos: list[dict[int, str]] = [{}]
        for number in range(1, len(items) + 1):
            pool = sorted(dict.fromkeys(groups.get(number, [])), key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))[:4]
            combos = [{**plan, number: i} for plan in combos for i in pool if i not in plan.values()][:256]
        ranked: list[tuple[float, int, dict[str, int]]] = []
        for place, plan in enumerate(combos):
            lines = {i: items[n - 1].qty for n, i in plan.items()}
            if len(lines) == len(items) and all(shelf.price_of(i) is not None for i in lines) and host.checker.inBudget(lines):
                ranked.append((sum((shelf.price_of(i) or 0.0) * q for i, q in lines.items()), place, lines))
        ranked.sort(key=lambda r: r[:2])
        return [lines for _, _, lines in ranked[:limit]]

    def poolOf(self, number: int) -> tuple[list[str], bool] | None:
        finder = getattr(self.host.coverage, "pool", None)
        if not callable(finder) or not number:
            return None
        try:
            got = finder(number)
        except Exception:
            return None
        if isinstance(got, dict):
            ids, complete = (got["ids"] if "ids" in got else got.get("pool")), got.get("complete")
        elif isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[1], bool):
            ids, complete = got
        else:
            ids, complete = getattr(got, "ids", None), getattr(got, "complete", None)
        if not isinstance(ids, (list, tuple)):
            return None
        rows = self.host.shelf.rows
        known = [str(i) for i in ids if str(i) in rows]
        if not known and complete is not True:
            return None
        return known, complete is True

    def lineIds(self, number: int) -> list[str]:
        shelf = self.host.shelf
        opened = [i for i in dict.fromkeys(shelf.seen_order + list(shelf.full)) if i in shelf.full and i in shelf.rows]
        mine = [i for i in opened if shelf.item_of(i) == number or i in self.lead_items.get(str(number), [])]
        if not mine and len(self.host.book.items) == 1:
            mine = opened
        return sorted(mine, key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))

    def lineNeeds(self) -> dict[int, list[tuple[str, str]]]:
        refs = self.host.lookups.ref_values
        out: dict[int, list[tuple[str, str]]] = {}
        for number, item in enumerate(self.host.book.items, 1):
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            out[number] = [(f"R{number}.{k}", need + (f" (the page it points at gives: {refs[need]})" if need in refs else "")) for k, need in enumerate(needs, 1)]
        return out

    def quoted(self, ident: str, quote: str) -> bool:
        if self.host.checker.quoteIn(ident, quote):
            return True
        want = set(re.findall(r"[a-z0-9]+", quote.lower()))
        have = set(re.findall(r"[a-z0-9]+", render(self.host.shelf.full.get(ident) or self.host.shelf.rows.get(ident) or {}).lower()))
        return bool(want) and want <= have

    def shortTable(self) -> dict[int, list[tuple[int, float, str]]]:
        host = self.host
        shelf = host.shelf
        version = host.book.version
        needs = self.lineNeeds()
        todo: list[tuple[int, str]] = []
        for number in range(1, len(host.book.items) + 1):
            ids = self.lineIds(number)
            keep = ids[: self.EACH] + [i for i in self.lead_items.get(str(number), []) if i in ids[self.EACH:]]
            todo += [(number, i) for i in keep if (self.graded.get(i) or (-1, 0, []))[0] != version and (number, i) not in todo]
        for number, ident in todo:
            if not needs.get(number):
                self.graded[ident] = (version, number, [])
        asked = [(n, i) for n, i in todo if needs.get(n)][: self.ALL]
        for start in range(0, len(asked), self.BATCH):
            chunk = asked[start:start + self.BATCH]
            mark = f"{version}:" + ",".join(i for _, i in chunk)
            if mark in self.short_failed:
                continue
            numbers = sorted({n for n, _ in chunk})
            prompt = self.SHORT + "\n\nREQUEST:\n" + host.store.request + "\n\nREQUIREMENTS:\n"
            prompt += "\n".join(f"item {n} ({host.book.items[n - 1].name}): " + "; ".join(f"{tag}: {need}" for tag, need in needs[n]) for n in numbers)
            prompt += "\n\nLISTINGS:\n" + "\n".join(f"ITEM {n} {i}: " + render(shelf.full.get(i), 160)[:1100] for n, i in chunk)
            got = host.models.ask_json(prompt, [LIGHT, HEAVY], 8000)
            if not isinstance(got, dict):
                self.short_failed.add(mark)
                break
            met: dict[str, set[str]] = {}
            seen: set[str] = set()
            for row in Text.items(got.get("rows")):
                if isinstance(row, dict):
                    row = [row.get("id"), row.get("tag") or row.get("requirement"), row.get("grade") or row.get("verdict"), row.get("quote") or row.get("words")]
                if not isinstance(row, list) or len(row) < 3:
                    continue
                ident, tag, verdict = (str(x if x is not None else "").strip() for x in row[:3])
                quote = str(row[3] if len(row) > 3 and row[3] is not None else "").strip()
                seen.add(ident)
                if verdict.lower().startswith("met") and self.quoted(ident, quote):
                    met.setdefault(ident, set()).add(tag)
            for n, i in chunk:
                if i in seen:
                    self.graded[i] = (version, n, [tag for tag, _ in needs[n] if tag not in met.get(i, set())])
        table: dict[int, list[tuple[int, float, str]]] = {}
        for ident, entry in self.graded.items():
            count = self.countOf(ident)
            price = shelf.price_of(ident)
            if count is not None and price is not None:
                table.setdefault(entry[1], []).append((count, price, ident))
        for rows in table.values():
            rows.sort()
        if todo:
            host.trace.append({"kind": "short_table", "table": {str(n): rows[:6] for n, rows in table.items()}})
        return table

    def endRule(self) -> dict[str, Any]:
        if self.rule is not None:
            return self.rule
        host = self.host
        request = host.store.request
        rule: dict[str, Any] = {"name": host.store.brief.get("quiet_decline") is False, "band": 0.0, "count": 1, "quote": "", "read": False}
        got = host.models.ask_json(self.ENDS + "\n\nREQUEST:\n" + request, [LIGHT, HEAVY], 700)
        quote = re.sub(r"\s+", " ", str((got or {}).get("quote") or "")).strip()
        if isinstance(got, dict) and len(quote) >= 12 and quote.lower() in re.sub(r"\s+", " ", request).lower():
            rule["quote"], rule["read"] = quote, True
            if isinstance(got.get("name"), bool) or got.get("name") is None:
                rule["name"] = got.get("name") is True
            band = got.get("band")
            if isinstance(band, (int, float)) and not isinstance(band, bool):
                for value in (float(band), float(band) * 100.0):
                    if 0 < value < 100 and re.search(r"(?<![\d.])" + re.escape(f"{value:g}") + r"(?![\d])", quote):
                        rule["band"] = value / 100.0
                        break
            count = Text.count(got.get("count"))
            if count and 1 <= count <= 5:
                rule["count"] = count
        self.rule = rule
        host.trace.append({"kind": "end_rule", "rule": rule})
        return rule

    def nearIds(self) -> list[str]:
        rule = self.endRule()
        if not rule.get("name"):
            return []
        rows = sorted((c, p, i) for got in self.shortTable().values() for c, p, i in got)
        if not rows:
            shelf = self.host.shelf
            guess = [i for ids in self.lead_items.values() for i in ids] + [i for n in range(1, len(self.host.book.items) + 1) for i in self.lineIds(n)]
            guess = sorted((i for i in dict.fromkeys(guess) if shelf.price_of(i) is not None and not shelf.sold_out(i)), key=lambda i: shelf.price_of(i) or 0.0)
            return guess[:1]
        low, cheap = rows[0][0], rows[0][1]
        picks = [i for c, p, i in rows if c == low and p <= cheap * (1.0 + rule["band"]) + 1e-9]
        return picks[: rule["count"]]

    def declineText(self, limit: int) -> str:
        ids = self.nearIds()
        self.named = ids
        self.host.trace.append({"kind": "decline_text", "ids": ids})
        if not ids:
            return "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[:limit]
        head = ("Closest listing: " if len(ids) == 1 else "Closest listings: ") + ", ".join(ids) + "."
        text = head + " Nothing in the store meets every requirement, so I am not placing an order; " + ("it comes" if len(ids) == 1 else "they come") + " up short on the fewest counts."
        return text if len(text) <= limit else (head if len(head) <= limit else ", ".join(ids[:1])[:limit])

    def muteNotice(self) -> None:
        book = self.host.book
        ends = [p for p in book.process if p["when"] == "end"]
        if ends and all(p["messages"] > 0 for p in ends):
            book.gate_ok.add(("end", self.host.shelf.cart_mark() + f"#{book.version}"))

    def cleanBasket(self) -> dict[str, int] | None:
        host = self.host
        items = host.book.items
        if host.store.brief.get("pick") in ("most", "least") or host.finished or not items:
            return None
        table = self.shortTable()
        groups = {n: [i for c, _, i in rows if c == 0] for n, rows in table.items()}
        if not all(groups.get(n) for n in range(1, len(items) + 1)):
            return None
        tries = 0
        leads = list(self.leads)
        found = None
        for lines in self.plans(groups, 6):
            key = host.checker.check_key(list(lines))
            if host.checker.verdicts.get(key):
                found = lines
                break
            if key in self.clean_tried or key in host.checker.checked:
                continue
            if tries >= 3 or host.models.cost > host.budgets.spendHard * 1.5:
                break
            self.clean_tried.add(key)
            tries += 1
            host.checker.check_cart(list(lines), host.budgets.spendHard * 1.5, False)
            if host.checker.verdicts.get(key):
                found = lines
                break
        self.leads = leads
        return found

    def openIds(self, wanted: list[str], room: int) -> list[str]:
        shelf = self.host.shelf
        wanted = [i for i in dict.fromkeys(wanted) if i in shelf.rows and i not in shelf.full and i not in shelf.opened and not shelf.sold_out(i)]
        wanted.sort(key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))
        wanted = wanted[: max(0, room)]
        if not wanted or self.host.finished:
            return []
        before = set(shelf.full)
        shelf.openRows(wanted)
        return [i for i in wanted if i in shelf.full and i not in before]

    def cheaperReads(self, lines: dict[str, int]) -> list[str]:
        host = self.host
        shelf = host.shelf
        if host.store.brief.get("pick") != "cheapest" or host.finished:
            return []
        wanted: list[str] = []
        for ident in lines:
            price = shelf.price_of(ident)
            pool = self.poolOf(self.numberOf(ident))
            if price is None or not pool or not pool[1]:
                continue
            wanted += [i for i in pool[0] if i != ident and shelf.price_of(i) is not None and (shelf.price_of(i) or 0.0) < price and shelf.buyable(i)]
        room = min(10, host.left() - len(lines) - host.book.tail() - 3)
        done = self.openIds(wanted, room)
        if done:
            host.trace.append({"kind": "cheaper_reads", "opened": done})
        return done

    def orderClean(self) -> str:
        host = self.host
        self.freshen()
        if self.completed or host.finished or not host.store.roles.get("cart") or not host.store.roles.get("buy") or host.left() < len(host.book.items) + 2:
            return ""
        lines = self.cleanBasket()
        if lines and self.cheaperReads(lines):
            lines = self.cleanBasket() or lines
        if not lines or host.finished:
            return ""
        saved = (dict(self.lead_items), list(self.leads), self.confirmed, self.force_buy)
        self.lead_items = {str(self.numberOf(i)): [i] for i in lines}
        self.leads = list(lines)
        self.confirmed = next(iter(lines))
        self.force_buy = False
        host.trace.append({"kind": "order_clean", "lines": lines})
        buys = host.buys
        done = self.complete_cart()
        if not done and host.buys == buys and not host.finished:
            self.lead_items, self.leads, self.confirmed, self.force_buy = saved
            self.completed = False
        return done

    def proof(self) -> str:
        host = self.host
        shelf = host.shelf
        items = host.book.items
        if host.finished or self.route_rounds >= 3 or host.left() <= len(items) + 6:
            return ""
        table = self.shortTable()
        lows = {n: (table[n][0][0] if table.get(n) else None) for n in range(1, len(items) + 1)}
        failing = [n for n, low in lows.items() if low != 0] or list(range(1, len(items) + 1))
        rows = sorted((c, p, i, n) for n, got in table.items() for c, p, i in got)
        wanted: list[str] = []
        loose: list[int] = []
        for number in failing:
            pool = self.poolOf(number)
            if pool is None:
                continue
            wanted += [i for i in pool[0] if shelf.buyable(i)]
            if not pool[1]:
                loose.append(number)
        if rows:
            _, price, _, number = rows[0]
            pool = self.poolOf(number)
            seen = (pool[0] if pool is not None else []) + [i for i in shelf.seen_order if shelf.item_of(i) == number or (len(items) == 1 and not shelf.item_of(i))]
            wanted += [i for i in seen if shelf.price_of(i) is not None and (shelf.price_of(i) or 0.0) < price and shelf.buyable(i)]
        room = min(8, host.left() - len(items) - host.book.tail() - 4)
        done = self.openIds(wanted, room)
        if done:
            self.route_rounds += 1
            host.trace.append({"kind": "decline_proof", "opened": done, "failing": failing, "loose": loose})
            return ("Before ending without a purchase, Act read the remaining candidates of item(s) " + ", ".join(map(str, sorted({self.numberOf(i) or failing[0] for i in done})))
                    + ": " + ", ".join(done) + ". Judge them against every requirement; buy what fits. End without a purchase only when no listing of some item meets every requirement.")
        if loose and not self.search_held and host.left() > max(10, host.max_steps // 4):
            self.search_held = True
            host.trace.append({"kind": "decline_proof", "opened": [], "failing": failing, "loose": loose})
            return (f"The search has not yet covered every listing of item(s) {', '.join(map(str, loose))}, and {host.left()} steps are left. Page on through the results or search the category with other words,"
                    " open the listings that might fit, and end without a purchase only when every listing of some item fails a requirement by its own record.")
        return ""

    def lastLook(self) -> str:
        host = self.host
        if host.finished:
            return ""
        if host.left() > len(host.book.items) + 6 and not host.budgets.closing():
            self.proof()
        return self.orderClean()

    def openCart(self) -> None:
        host = self.host
        shelf = host.shelf
        closed = [i for i in shelf.cart_ids() if i in shelf.rows and i not in shelf.full and i not in shelf.opened and not shelf.sold_out(i)]
        if closed and host.left() > len(closed) + 1 and not host.finished:
            shelf.openMore(closed, 1)
            host.trace.append({"kind": "open_cart", "ids": closed})

    def decline_hold(self) -> str:
        if self.decline_reads < 2 and self.host.left() > 4:
            self.decline_reads += 1
            opened = self.read_candidates(3, min(6, self.host.left() - (8 if self.host.left() > 10 else 4)))
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
            keys = ["fit", "best", "same"] + (["near"] if near else [])
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
        firm = {str(i) for entry in Text.items((got or {}).get("items")) if isinstance(entry, dict) for key in ("fit", "best", "same") for i in Text.items(entry.get(key))}
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
        self.freshen()
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
        if self.force_buy and loose and self.near_holds < 2 and self.host.left() > len(self.host.book.items) + 8:
            self.near_holds += 1
            self.host.trace.append({"kind": "lead_hold", "why": "near", "items": loose})
            return ""
        self.completed = True
        self.completed_at = self.host.book.version
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
        self.order: list[int] = []
        self.fits: list[int] = []
        self.single = ASK_KEYS <= 1
        self.pairs: list[tuple[int, str]] = []
        self.counts: dict[int, set[str]] = {}
        self.cap = 2
        self.keysOf: dict[int, list[str]] = {}
        self.weightOf: dict[tuple[int, str], float] = {}
        self.itemOdds: dict[int, float] = {}

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

    @staticmethod
    def junk(per: list[str]) -> bool:
        if sum(len(v) for v in per) / len(per) > 40:
            return True
        if sum(1 for v in per if re.search(r"[a-z)]\s*/\s*[a-z0-9(]", v)) * 2 > len(per):
            return True
        listy = sum(1 for v in per if len([p for p in re.split(r"[,;/|]", v) if p.strip()]) >= 3)
        if listy * 2 > len(per):
            return True
        codes = [v for v in per if re.fullmatch(r"[a-z]{1,6}[\s\-_]?\d[\w\-./]*", v) and len(re.findall(r"\d", v)) >= 2]
        return len(per) >= 3 and len(codes) * 2 > len(per) and len(set(codes)) * 4 >= len(codes) * 3

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
                slot = stats.setdefault(name, {"n": 0, "values": [], "per": [], "ids": set()})
                slot["n"] += 1
                slot["ids"].add(listings)
                text = str(value).strip().lower()
                slot["per"].append(text)
                if text and text not in slot["values"]:
                    slot["values"].append(text)
        Interviewer.mergeVariants(stats, label)
        self.counts[number] = {name for name, slot in stats.items() if Interviewer.countLike(name, slot["per"])}
        ranked = []
        for name, slot in stats.items():
            values = slot["values"]
            per = slot["per"]
            if not values or not per or Interviewer.junk(per) or re.search(r"\d", name):
                continue
            ranked.append((name, len(slot["ids"]), len(values), [v[:40] for v in values[:4]]))
        merged: list[tuple[str, int, int, list[str]]] = []
        values = {name: slot["values"] for name, slot in stats.items()}
        for row in sorted(ranked, key=lambda r: (-r[1], r[0])):
            if not Interviewer.alias(row, merged, values):
                merged.append(row)
        return merged[:18]

    @staticmethod
    def mergeVariants(stats: dict[str, dict[str, Any]], label: str) -> None:
        units = {w for slot in stats.values() for v in slot["values"] for w in re.findall(r"\d\s*([a-z]+)", v)}
        for name in sorted(stats, key=lambda k: (-len(k.split()), k)):
            if name not in stats:
                continue
            words = name.split()
            for other in sorted(stats, key=lambda k: (len(k.split()), k)):
                ow = other.split()
                if other == name or other not in stats or len(ow) >= len(words):
                    continue
                if words[:len(ow)] == ow:
                    extra = words[len(ow):]
                elif words[-len(ow):] == ow:
                    extra = words[:len(words) - len(ow)]
                else:
                    continue
                mine, theirs = set(stats[name]["values"]), set(stats[other]["values"])
                shared = len(mine & theirs)
                if shared < 2 and shared * 10 < 3 * max(1, len(mine)):
                    continue
                plain = all(w in units or any(c.isdigit() for c in w) for w in extra)
                keep, drop = (other, name) if plain or len(stats[other]["ids"]) >= len(stats[name]["ids"]) else (name, other)
                stats[keep]["ids"] |= stats[drop]["ids"]
                stats[keep]["per"] += stats[drop]["per"]
                stats[keep]["values"] += [v for v in stats[drop]["values"] if v not in stats[keep]["values"]]
                del stats[drop]
                break

    @staticmethod
    def countLike(name: str, per: list[str]) -> bool:
        own = {Text.single(w) for w in name.split()}
        hits = 0
        for value in per:
            found = re.fullmatch(r"(\d)(?:\s+([a-z]+))?", value.strip())
            if found and (not found.group(2) or Text.single(found.group(2)) in own):
                hits += 1
        last = name.split()[-1] if name.split() else ""
        return bool(per) and Text.single(last) != last and hits * 5 >= len(per) * 3

    @staticmethod
    def quoted(quote: Any, text: str) -> bool:
        want = Text.bag(quote, True) if isinstance(quote, str) else []
        have = Text.bag(text, True)
        if len(want) < 3 or not have:
            return False
        best = 0
        last = [0] * (len(want) + 1)
        for word in have:
            row = [0] * (len(want) + 1)
            for j, other in enumerate(want, 1):
                if word == other:
                    row[j] = last[j - 1] + 1
                    best = max(best, row[j])
            last = row
        return best >= len(want) or best >= max(4, int(0.7 * len(want) + 0.999))

    @staticmethod
    def settled(item: Item) -> list[tuple[frozenset[str], str]]:
        return [(Text.stem(m.split(":", 1)[0]), m.split(":", 1)[1].lower()) for m in item.musts + item.told if ":" in m and m not in item.dropped]

    @staticmethod
    def stated(row: tuple[str, int, int, list[str]], settled: list[tuple[frozenset[str], str]], names: list[str]) -> bool:
        name = Text.stem(row[0])
        words = [v for v in row[3] if len(v) >= 3 and not re.fullmatch(r"[\d\s.,]+", v)]
        for prop, value in settled:
            if not prop:
                continue
            if prop == name:
                return True
            shared = any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", value) for v in words)
            if shared and (prop < name or not any(Text.stem(k) == prop for k in names)):
                return True
        return False

    @staticmethod
    def alias(row: tuple[str, int, int, list[str]], kept: list[tuple[str, int, int, list[str]]], values: dict[str, list[str]]) -> bool:
        mine = {v for v in values.get(row[0], []) if not re.fullmatch(r"[\d\s.,]+", v)}
        for other in kept:
            theirs = set(values.get(other[0], []))
            if Text.names(row[0], other[0]) and set(row[3]) & set(other[3]):
                return True
            if len(mine) >= 2 and len(mine & theirs) >= 2 and len(mine & theirs) * 5 >= len(mine) * 3:
                return True
        return False

    def tableKey(self, key: Any, table: list[tuple[str, int, int, list[str]]]) -> str | None:
        name = Text.words(re.split(r"[/,;(]", str(key or ""))[0]).strip().rstrip("?.")
        if not name or len(name.split()) > 5:
            return None
        if not table:
            return name
        for row in table:
            if row[0] == name or Text.stem(row[0]) == Text.stem(name):
                return row[0]
        want = Text.stem(name)
        near = [row for row in table if Text.names(name, row[0]) or Text.names(row[0], name)]
        shared = [len(want & Text.stem(row[0])) for row in near]
        best = [row for row, n in zip(near, shared) if n == max(shared)] if near else []
        return best[0][0] if len(best) == 1 else None

    def queueFor(self, number: int, table: list[tuple[str, int, int, list[str]]], front: list[str], skip: set[str]) -> list[str]:
        item = self.host.book.items[number - 1]
        settled = Interviewer.settled(item)
        listings = max((row[1] for row in table), default=0)
        queue: list[str] = []
        few: list[str] = []
        late: list[str] = []
        names = [row[0] for row in table]
        for name in front:
            if name not in queue and not any(prop == Text.stem(name) for prop, _ in settled):
                queue.append(name)
        for row in table:
            if row[0] in queue or Interviewer.stated(row, settled, names):
                continue
            if row[0] in skip or row[0] in self.counts.get(number, set()):
                late.append(row[0])
            elif listings >= 4 and row[1] * 4 < listings:
                few.append(row[0])
            else:
                queue.append(row[0])
        lead = [name for name in dict.fromkeys(front) if name in queue]
        if self.cap >= 3 and len(lead) > 1:
            plain = [name for name in queue if name not in lead]
            mixed = lead[:1]
            for k in range(max(len(plain), len(lead) - 1)):
                mixed += plain[k:k + 1] + lead[k + 1:k + 2]
            queue = mixed
        return queue + [name for name in few + late if name not in queue]

    def plan(self, count: int) -> dict[int, list[str]]:
        host = self.host
        items = host.book.items
        request = host.store.request
        blocks = []
        tables: dict[int, list[tuple[str, int, int, list[str]]]] = {}
        for number, item in enumerate(items, 1):
            tables[number] = self.keyStats(number)
            settled = "; ".join(m for m in item.musts + item.told if m not in item.dropped) or "nothing"
            lines = [f"item {number}: {item.name} (store category: {self.itemLabel(number)}) - settled by the request: {settled}"]
            if tables[number]:
                for name, carried, distinct, values in tables[number][:PLAN_ROWS]:
                    lines.append(f"  - {name}: on {carried} listings, {distinct} values: " + " | ".join(values[:3]))
            else:
                lines.append("  (no opened listing of this item yet)")
            blocks.append("\n".join(lines))
        self.cap = count
        prompt = ASK_PLAN.replace("PICKS", str(max(3, count + 1))) + "\n\nREQUEST:\n" + request + "\n\nITEMS AND THEIR KEYS:\n" + "\n".join(blocks)
        first = [str(x) for x in Text.items(host.store.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        if ASK_PAIR and len(items) >= 2:
            prompt += "\n\n" + PAIR_ASK
        got = host.models.ask_json(prompt, [TOP, HEAVY, LIGHT] if len(items) >= 2 else [HEAVY, LIGHT], 1800, "low") or {}
        return self.arrange(got, tables)

    def arrange(self, got: dict[str, Any], tables: dict[int, list[tuple[str, int, int, list[str]]]]) -> dict[int, list[str]]:
        host = self.host
        request = host.store.request
        total = len(host.book.items)
        front: dict[int, list[str]] = {}
        pointed: list[int] = []
        for field in ("ask_me", "story"):
            for entry in Text.items(got.get(field)):
                if not isinstance(entry, dict):
                    continue
                number = Text.count(entry.get("item"))
                if not number or not 1 <= number <= total or not Interviewer.quoted(entry.get("quote"), request):
                    continue
                name = self.tableKey(entry.get("key"), tables[number])
                if name is None:
                    continue
                front.setdefault(number, [])
                if name not in front[number]:
                    front[number].append(name)
                if number not in pointed:
                    pointed.append(number)
        fits: list[int] = []
        for entry in Text.items(got.get("fits")):
            number = Text.count(entry.get("item")) if isinstance(entry, dict) else None
            if number and 1 <= number <= total and number not in fits and Interviewer.quoted(entry.get("quote"), request):
                fits.append(number)
        skip: dict[int, set[str]] = {}
        for entry in Text.items(got.get("skip")):
            number = Text.count(entry.get("item")) if isinstance(entry, dict) else None
            if number and 1 <= number <= total:
                skip.setdefault(number, set()).update(Text.words(k) for k in Text.items(entry.get("keys")) if isinstance(k, str))
        chosen: dict[int, list[str]] = {}
        for field in ("rank", "plain"):
            for entry in Text.items(got.get(field)):
                number = Text.count(entry.get("item")) if isinstance(entry, dict) else None
                if not number or not 1 <= number <= total or (field == "plain" and len(tables[number]) >= SMALL_TABLE) or (field == "rank" and not tables[number]):
                    continue
                for key in Text.items(entry.get("keys")):
                    name = self.tableKey(key, tables[number]) or (self.tableKey(key, []) if field == "plain" else None)
                    if name and name not in chosen.setdefault(number, []):
                        chosen[number].append(name)
        self.pairs = []
        if total >= 2 and ASK_PAIR:
            for entry in Text.items(got.get("pair")):
                number = Text.count(entry.get("item")) if isinstance(entry, dict) else None
                quote = str(entry.get("quote") or "").strip() if isinstance(entry, dict) else ""
                if not number or not 1 <= number <= total or (quote and not Interviewer.quoted(quote, request)):
                    continue
                name = self.tableKey(entry.get("key"), tables[number])
                if not name or name in skip.get(number, set()) or (number, name) in self.pairs:
                    continue
                if tables[number] and any(row[0] == name and Interviewer.stated(row, Interviewer.settled(host.book.items[number - 1]), [r[0] for r in tables[number]]) for row in tables[number]):
                    continue
                self.pairs.append((number, name))
            self.pairs = self.pairs[:2] if len(self.pairs) >= 2 and self.pairs[0][0] == self.pairs[1][0] else []
        rest = [n for n in range(1, total + 1) if n not in pointed]
        self.order = [n for n in pointed if n not in fits] + [n for n in rest if n not in fits] + [n for n in pointed if n in fits] + [n for n in rest if n in fits]
        self.fits = fits
        ranked = {}
        for n in self.order:
            place = {k: i for i, k in enumerate(chosen.get(n, []))}
            table = sorted(tables[n], key=lambda r: (-r[1], place.get(r[0], len(place)), r[0]))
            plain = [k for k in chosen.get(n, []) if not tables[n] and k not in front.get(n, [])]
            ranked[n] = self.queueFor(n, table, front.get(n, []) + plain, skip.get(n, set()))
            if tables[n] and len(tables[n]) < SMALL_TABLE:
                ranked[n] += [k for k in chosen.get(n, []) if k not in ranked[n] and not any(prop == Text.stem(k) for prop, _ in Interviewer.settled(host.book.items[n - 1]))]
        host.trace.append({"kind": "ask_plan", "order": list(self.order), "fits": fits, "story": {n: list(v) for n, v in front.items()}, "pair": [list(p) for p in self.pairs],
                           "ranked": {n: list(v[:8]) for n, v in ranked.items()}, "tables": {n: len(t) for n, t in tables.items()}})
        return ranked

    def nextPick(self, ranked: dict[int, list[str]]) -> tuple[int, str] | None:
        for number, name in self.pairs:
            if name not in self.tried.setdefault(number, set()):
                if name in ranked.get(number, []):
                    ranked[number].remove(name)
                return number, name
        place = {n: i for i, n in enumerate(self.order)}
        order = sorted(ranked, key=lambda n: (self.revealed.get(n, 0) + len(self.tried.get(n, set())) + (2 if n in self.fits else 0), place.get(n, len(place) + n)))
        for number in order:
            queue = ranked[number]
            while queue:
                name = queue.pop(0)
                if name not in self.tried.setdefault(number, set()):
                    return number, name
        return None

    def picks(self, ranked: dict[int, list[str]]) -> list[tuple[int, str]]:
        first = self.nextPick(ranked)
        if first is None:
            return []
        chosen = [first]
        self.tried.setdefault(first[0], set()).add(first[1])
        while not self.single and len(chosen) < ASK_KEYS:
            if ASK_ACROSS:
                more = self.nextPick(ranked)
            else:
                queue = ranked.get(first[0]) or []
                more = (first[0], queue.pop(0)) if queue else None
            if more is None:
                break
            chosen.append(more)
            self.tried.setdefault(more[0], set()).add(more[1])
        return chosen

    def oddsSeed(self, ranked: dict[int, list[str]]) -> None:
        lines = len(self.host.book.items)
        for number, queue in ranked.items():
            if number in self.keysOf:
                continue
            keys = [k for k in queue if k and k not in self.tried.get(number, set())]
            self.keysOf[number] = keys
            self.itemOdds[number] = ASK_ITEM_FEW if lines <= 2 else ASK_ITEM_MANY
            for rank, name in enumerate(keys):
                self.weightOf[(number, name)] = ASK_RANK_TOP * ASK_RANK_DECAY ** rank

    def oddsShares(self, number: int) -> list[tuple[float, str]]:
        keys = self.keysOf.get(number) or []
        mass = sum(self.weightOf.get((number, k), 0.0) for k in keys)
        outside = max(ASK_OUTSIDE, 1.0 - sum(ASK_RANK_TOP * ASK_RANK_DECAY ** r for r in range(len(keys))))
        total = mass + outside
        return sorted(((self.weightOf.get((number, k), 0.0) / total, k) for k in keys if self.weightOf.get((number, k), 0.0) > 0), key=lambda e: -e[0])

    @staticmethod
    def twins(a: str, b: str) -> bool:
        x, y = Text.stem(a), Text.stem(b)
        return bool(x and y) and len(x & y) / len(x | y) > 0.5

    def oddsPlan(self, number: int) -> tuple[float, list[str]]:
        shares = []
        for share, name in self.oddsShares(number):
            if not any(Interviewer.twins(name, kept) for _, kept in shares):
                shares.append((share, name))
        chance = self.itemOdds.get(number, 0.0)
        best: tuple[float, list[str]] = (0.0, [])
        top = 1 if self.single else len(ASK_RATES)
        for size in range(1, min(top, len(shares)) + 1):
            rates = (1.0,) if size == 1 else ASK_RATES
            value = chance * sum(rates[j] * shares[j][0] for j in range(size))
            if value > best[0] + 1e-9:
                best = (value, [k for _, k in shares[:size]])
        return best

    def oddsPicks(self, ranked: dict[int, list[str]]) -> list[tuple[int, str]]:
        self.oddsSeed(ranked)
        place = {n: i for i, n in enumerate(self.order)}
        options = []
        for number in self.keysOf:
            value, keys = self.oddsPlan(number)
            if keys:
                options.append((value, -place.get(number, len(place) + number), number, keys))
        if not options:
            return []
        value, _, number, keys = max(options)
        self.host.trace.append({"kind": "ask_odds", "item": number, "keys": list(keys), "value": round(value, 3),
                                "options": {str(n): round(v, 3) for v, _, n, _ in options}})
        for name in keys:
            self.tried.setdefault(number, set()).add(name)
        return [(number, name) for name in keys]

    @staticmethod
    def sameProp(a: str, b: str, loose: bool = False) -> bool:
        x, y = Text.stem(a), Text.stem(b)
        if not (x and y):
            return False
        if x <= y or y <= x or Interviewer.twins(a, b):
            return True
        return loose and any(u.startswith(v) or v.startswith(u) for u in x if len(u) >= 4 for v in y if len(v) >= 4)

    def oddsHit(self, number: int, names: list[str], got: dict[str, Any]) -> None:
        book = self.host.book
        props = [str(e.get("need") or "").split(":", 1)[0].strip() for e in Text.items(got.get("needs"))
                 if isinstance(e, dict) and book.itemIndex(e.get("item")) == number and str(e.get("need") or "").strip()]
        spot = next((j for j, name in enumerate(names) if any(Interviewer.sameProp(name, p) for p in props if p)), None)
        if spot is None:
            spot = next((j for j, name in enumerate(names) if any(Interviewer.sameProp(name, p, True) for p in props if p)), None)
        for name in names[:spot + 1] if spot is not None else []:
            for other in self.keysOf.get(number) or []:
                if other == name or Interviewer.twins(name, other):
                    self.weightOf[(number, other)] = 0.0
        self.itemOdds[number] = ASK_ITEM_AGAIN

    def oddsHeard(self, picks: list[tuple[int, str]], hit: set[int], got: dict[str, Any] | None = None) -> None:
        if not picks:
            return
        number = picks[0][0]
        names = [name for _, name in picks]
        if number in hit:
            self.oddsHit(number, names, got or {})
            for other in hit:
                if other != number and other in self.itemOdds:
                    self.itemOdds[other] = ASK_ITEM_AGAIN
            return
        rates = (1.0,) if len(names) == 1 else ASK_RATES
        shares = {k: v for v, k in self.oddsShares(number)}
        chance = sum(rates[min(j, len(rates) - 1)] * shares.get(name, 0.0) for j, name in enumerate(names))
        for j, name in enumerate(names):
            miss = 1.0 - rates[min(j, len(rates) - 1)]
            for other in self.keysOf.get(number) or []:
                if other == name or Interviewer.twins(name, other):
                    self.weightOf[(number, other)] = self.weightOf.get((number, other), 0.0) * miss
        prior = self.itemOdds.get(number, 0.0)
        if number not in hit:
            self.itemOdds[number] = prior * (1.0 - chance) / max(1e-9, 1.0 - prior * chance)
        for other in hit:
            if other in self.itemOdds:
                self.itemOdds[other] = ASK_ITEM_AGAIN

    def question(self, picks: list[tuple[int, str]]) -> str:
        groups: list[tuple[int, list[str]]] = []
        for number, name in picks:
            if groups and groups[-1][0] == number:
                groups[-1][1].append(name)
            else:
                groups.append((number, [name]))
        parts = []
        for number, names in groups:
            keys = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
            parts.append(f"{keys} would you like for the {Text.words(self.itemLabel(number))}")
        return "What " + ", and what ".join(parts) + "?"

    def readCap(self) -> None:
        host = self.host
        book = host.book
        if book.cap_from not in ("brief", "assumed"):
            return
        lines = [f"- {t['function']['name']}: {str(t['function'].get('description') or '')[:400]}" for t in host.store.tools]
        got = host.models.ask_json(CAP + "\n\nREQUEST:\n" + host.store.request + "\n\nTOOLS:\n" + "\n".join(lines), [LIGHT, HEAVY], 400)
        if not isinstance(got, dict) or not got:
            return
        meter = got.get("meter")
        book.meter = meter if isinstance(meter, str) and meter in host.store.schemas else None
        limit = Text.count(got.get("messages"))
        quote = str(got.get("quote") or "")
        if limit is not None and limit >= 0 and book.capSaid(quote, limit):
            book.msg_limit, book.cap_from = max(limit, book.msg_used), "request"
        else:
            book.msg_limit, book.cap_from = max(min(book.msg_limit, 1), book.msg_used), "assumed"
        book.holdFor()
        host.trace.append({"kind": "cap", "messages": limit, "quote": quote[:120], "meter": book.meter, "limit": book.msg_limit, "from": book.cap_from})

    def readMeter(self) -> None:
        host = self.host
        book = host.book
        tool = book.meter
        busy = {host.store.roles.get(r) for r in ("ask", "cart", "buy", "refuse")} - {None}
        if book.cap_from != "assumed" or not tool or tool not in host.store.schemas or tool in busy or host.left() <= 10 or book.meter_read:
            return
        book.meter_read = True
        faults: list[str] = []
        args = StoreMap.shape({}, host.store.schemas.get(tool), "args", faults)
        if faults:
            return
        reply = host.act(tool, args)
        seen = reply.get("observation") if isinstance(reply, dict) else None
        if seen is None:
            return
        numbers = {int(v) for v in re.findall(r"(?<![\w.])\d+(?![\w.])", render(seen))}
        got = host.models.ask_json(METER + "\n\nTOOL: " + tool + "\nRESULT:\n" + render(seen, 400), [LIGHT, HEAVY], 300) or {}
        left, used = Text.count(got.get("left")), Text.count(got.get("used"))
        host.trace.append({"kind": "meter", "tool": tool, "seen": Text.clip(seen, 200), "left": left, "used": used})
        if left is not None and left in numbers:
            book.msg_limit = book.msg_used + left
            book.cap_from = "meter"
        elif Text.count(got.get("total")) is not None and Text.count(got.get("total")) in numbers:
            book.msg_limit = Text.count(got.get("total"))
            book.cap_from = "meter"

    def handOff(self, heard: dict[str, Any], picks: list[tuple[int, str]], text: str, words: str) -> None:
        book = self.host.book
        for entry in Text.items(heard.get("needs")):
            if not isinstance(entry, dict) or entry.get("points") is not True or not str(entry.get("need") or "").strip():
                continue
            quote = str(entry.get("quote") or "").strip()
            book.pending_refs.append({"item": book.itemIndex(entry.get("item")), "need": str(entry["need"]).strip(), "quote": quote if Interviewer.quoted(quote, words) else words,
                                      "question": text, "step": self.host.step, "state": "pending"})

    def run(self) -> str:
        host = self.host
        book = host.book
        tool = host.store.roles.get("ask")
        if not tool:
            return ""
        self.readCap()
        self.readMeter()
        book.holdFor()
        count = book.msgs_free()
        if count <= 0:
            return ""
        ranked = self.plan(count)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        fresh: list[tuple[int, str]] = []
        shape: set[str] | None = None
        while book.msgs_free() > 0 and not host.finished and host.left() > 6:
            picks = self.oddsPicks(ranked)
            if not picks:
                break
            text = self.question(picks)
            heard = book.sendOne(tool, text)
            if heard is None:
                break
            words, answer = heard
            batch = [(picks[0][0], text, words)]
            got = {} if isinstance(answer, dict) and answer.get("error") else book.hear(batch, (LIGHT, HEAVY))
            named = [e for e in Text.items(got.get("needs")) + Text.items(got.get("free")) if isinstance(e, dict)]
            odd = shape is not None and (set(answer) if isinstance(answer, dict) else set()) != shape
            if (got.get("refused") == "limit" and not named and odd) or (isinstance(answer, dict) and answer.get("error")):
                book.msg_used -= 1
                book.msg_limit = book.msg_used
                host.trace.append({"kind": "ask_refused", "reply": words[:200], "why": got.get("refused") or "error"})
                break
            sent += batch
            if shape is None:
                shape = set(answer) if isinstance(answer, dict) else set()
            before = len(told)
            book.take_heard(got, told)
            for entry in Text.items(got.get("needs")):
                if isinstance(entry, dict) and entry.get("points") is not True and str(entry.get("need") or "").strip():
                    fresh.append((book.itemIndex(entry.get("item")), str(entry["need"]).strip()))
            hit = {book.itemIndex(e.get("item")) for e in Text.items(got.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()} if len(told) > before else set()
            for number in hit:
                self.revealed[number] = self.revealed.get(number, 0) + 1
            crowded = [Text.count(c) for c in Text.items(got.get("crowded"))]
            if len(picks) > 1 and 1 in crowded and not hit:
                self.single = True
                for number, name in reversed(picks):
                    self.tried.get(number, set()).discard(name)
                    ranked.setdefault(number, []).insert(0, name)
            else:
                self.oddsHeard(picks, hit, got)
            self.handOff(got, picks, text, words)
        if told:
            book.version += 1
        found = self.reSearch(fresh) if told else []
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "The interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in host.lookups.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        live = {need for _, need in book.needs_now()}
        if any(need in live and need not in host.lookups.ref_values for _, need in host.lookups.refs) and host.left() > 12 and not host.finished:
            chain, _ = host.lookups.trail(2)
            if chain:
                summary += "\n" + chain
        if found:
            summary += "\nAct already searched the new value(s) itself: " + "; ".join(found) + ". Open the cheapest of these that may fit before anything else."
        if book.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(book.queued)
            book.queued = []
        return summary

    def reSearch(self, fresh: list[tuple[int, str]]) -> list[str]:
        host = self.host
        cover = host.coverage
        shelf = host.shelf
        finder = host.store.roles.get("search")
        if not finder or not fresh:
            return []
        df, total = cover.vocabDf()
        out: list[str] = []
        done: set[str] = set()
        for number, need in fresh:
            if len(out) >= 2 or host.finished or host.left() <= len(host.book.items) + 10:
                break
            words = cover.valueWords(need, df, total)
            if not words:
                value = need.split(":", 1)[1] if ":" in need else need
                words = [w for m in re.findall(r"\d+(?:\.\d+)?\s*[a-z]{0,4}\b", value.lower()) for w in m.split()][:3]
            if not words:
                continue
            label = [w for w in re.findall(r"[a-z0-9]+", self.itemLabel(number).lower()) if w not in words]
            call = cover.search_args(finder, " ".join(words + label))
            if call is None:
                continue
            key = finder + render(call)
            if key in host.read_cache or key in done:
                continue
            done.add(key)
            reply = host.act(finder, call)
            if reply is None:
                break
            seen = reply.get("observation")
            host.book.observe(finder, seen)
            host.read_cache[key] = (host.book.version, host.step, host.show(seen))
            cover.note_search(finder, call, seen)
            rows = [i for i, _ in shelf.records_in(seen) if shelf.buyable(i)]
            rows.sort(key=lambda i: shelf.price_of(i) if shelf.price_of(i) is not None else float("inf"))
            priced = [f"{i} at {shelf.price_of(i):g}" if shelf.price_of(i) is not None else i for i in rows[:6]]
            out.append(f"'{' '.join(words + label)}' for item {number} gave {len(rows)} buyable row(s)" + (": " + ", ".join(priced) if priced else ""))
            host.trace.append({"kind": "re_search", "need": need, "item": number, "rows": rows[:8]})
        return out

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
        self.units_fixed = 0

    def unitsHold(self) -> None:
        host = self.host
        shelf = host.shelf
        items = host.book.items
        if self.units_fixed >= 2 or not host.store.roles.get("cart") or host.left() < 3 or not items:
            return
        seen = shelf.unitsIn(shelf.cart_view) if shelf.cartFresh() else {}
        if not seen and shelf.cart_said:
            try:
                seen = shelf.unitsIn(json.loads(shelf.cart_said))
            except ValueError:
                seen = {}
        lines = [i for i in shelf.cart_ids() if not shelf.sold_out(i)]
        have = {i: seen.get(i, shelf.cart_lines.get(i)) for i in lines}
        owner = {i: shelf.item_of(i) or (1 if len(items) == 1 else 0) for i in lines}
        want: dict[str, int] = {}
        for ident, number in owner.items():
            if number and 1 <= number <= len(items) and list(owner.values()).count(number) == 1:
                want[ident] = items[number - 1].qty
        fix = {i: (have[i], want[i]) for i in want if isinstance(have.get(i), int) and have[i] != want[i]}
        if not fix:
            return
        self.units_fixed += 1
        calls = shelf.unitCalls(fix, {i: want.get(i, have.get(i) or 1) for i in lines})
        if not calls or len(calls) + 2 > host.left():
            host.trace.append({"kind": "units_fix", "fix": {i: list(v) for i, v in fix.items()}, "calls": None})
            return
        done = shelf.setLines(calls)
        host.trace.append({"kind": "units_fix", "fix": {i: list(v) for i, v in fix.items()}, "calls": done})

    def hold(self, job: Pending) -> str | None:
        host = self.host
        if job.ending and host.store.is_refuse(job.name, job.args) and Budgets.closeOut(host.models.cost, len(host.book.items), Budgets.held(host)) and not host.decline.openWork() and not host.decline.workLeft():
            host.trace.append({"kind": "refuse_closeout", "cost": round(host.models.cost, 4), "left": host.left()})
            done = host.decline.closeOrder()
            if done:
                return done + (" [episode finished]" if host.finished else "")
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
        self.unitsHold()
        swap = shelf.swap_hint()
        held = [i for i in shelf.cart_ids() if i in shelf.gone] if self.held < 2 else []
        if swap or held:
            self.held += 0 if swap else 1
            return "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + shelf.cart_text()
        chain, hold = host.lookups.trail()
        if hold:
            return "Not run yet (no step used). " + chain + "\nCart now: " + shelf.cart_text()
        read = shelf.openMore(shelf.cart_ids(), host.book.tail() + 3) + host.coverage.cover_pages() + host.coverage.cheaper_rows()
        read += [f"{i}: " + shelf.record_of(i) for i in host.decline.cheaperReads({i: 1 for i in shelf.cart_ids()})]
        verdict = host.checker.check_cart()
        if verdict and (host.checker.better or host.checker.linkedNeeds()) and host.left() > len(shelf.cart_ids()) + 2:
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
        if host.left() <= 3:
            return None
        hold = decline.decline_hold()
        if host.left() <= 5:
            return ("Not run yet (no step used). " + hold) if hold else None
        if hold:
            return "Not run yet (no step used). " + hold
        review = decline.audit()
        if review:
            return "Not run yet (no step used). " + review
        done = decline.orderClean()
        if done:
            return done + (" [episode finished]" if host.finished else "")
        if not decline.leads or host.left() <= 4:
            route = decline.proof()
            return ("Not run yet (no step used). " + route) if route else None
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
        route = decline.proof()
        return ("Not run yet (no step used). " + route) if route else None

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
        done = decline.complete_cart() or decline.orderClean() or decline.lateOrder()
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
        if host.left() < 2 or (name == store.roles.get("ask") and (book.msg_used >= book.msg_limit or stage != "order")):
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
        if "observation" in data:
            data["observation"] = TextRows.observation(data["observation"])
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
        self.decline.freshen()
        if buy and not holding and not self.decline.decline_proven and not self.decline.completed and self.left() >= len(self.book.items) + 2:
            stale = self.decline.audit_runs == 0 or self.book.version != self.decline.audit_version
            if not self.budgets.closing() and (stale or (not self.decline.lead_items and self.shelf.cart_ids())):
                self.decline.audit(near=not self.decline.lead_items and bool(self.shelf.cart_ids()), late=True)
            done = self.decline.buy_leads() or self.decline.orderClean()
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
                self.decline.openCart()
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
            if (buy and self.decline.lastLook()) or self.finished or self.left() <= 0:
                return ""
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
        self.lookups.begin()
        opening = self.store.view.get("opening")
        if opening is not None:
            self.shelf.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.store.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        swept = self.coverage.sweep()
        intro += ("\n\n" + swept + "\n\n") if swept else "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n"
        intro += self.status()
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
            fresh = self.coverage.refresh()
            if fresh and self.chat and self.chat[-1].get("role") in ("user", "tool") and isinstance(self.chat[-1].get("content"), str):
                self.chat[-1]["content"] += "\n" + fresh
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
