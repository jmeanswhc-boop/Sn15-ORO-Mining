
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.045
SPEND_HARD = 0.066
REVIEW_SPEND = 0.06         # reviews stay on up to here, past the soft line: an unreviewed ending is the costly kind
CHASE_NEEDS = 2             # heard requirements Kyo searches for itself, right after the interview
CHASE_FLOOR = 10            # steps that must remain for one such search and its reads
FREE_HOLDS = 2              # checkouts the no-model record check may hold back once reviews are out of money
# pauses before a failed model call is sent again; the whole episode has 300 s, so waiting may take 45 s at most
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
ROW_CHARS = 260             # text shown per field of a long result list; the full row stays in the shelf
OBS_LIMIT = 15000
HISTORY_LIMIT = 100000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo fetches itself under a cheapest, most or least pick
COMPARE_START = 4           # side-by-side size tried first; the store's refusals teach the real limit
CLOCK_STOP = 250            # seconds into the 300 s problem: no new model turn starts; Kyo finishes with what it holds
CLOCK_END = 292             # a model call never runs past this: its timeout is cut to the time left
CLOCK_GATE = 262            # past this, the process gate's model rounds are skipped and no failed model call is retried
CHEAPER_OPENS = 3           # cheaper same-kind rows whose attributes Kyo reads before a cheapest checkout

# a single attempt per request (think() does the resending): a stuck call then costs 50 s of the 300 s, not minutes
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

STAGES = ("choose", "order", "end")
DECLINE = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
CONFIRM = "Please place this order."
SKIP = "Not executed, no step spent: "
PAGE_ARG = re.compile(r"token|cursor|page|next|offset|after|continue", re.I)   # an argument or key that moves to another page
HOLD = "Held back, no step spent. "
DONE = " [the episode has ended]"
_NUMBER = r"\s*\d+(\.\d+)?\s*"
# words a tool reply uses to report that the call did not go through
_FAILED = ("error", "failed", "failure", "fail", "rejected", "refused", "denied", "invalid")
_INF = float("inf")


GUIDE = """Role: you are Kyo, buying on behalf of a single shopper in an online shop. Nothing happens except through the shop's tools.

Score: all or nothing. A 1 needs an exactly right finish, one of two kinds:
(a) a checkout whose cart carries every item asked for, in the quantity asked for, where each item satisfies every requirement still in force at checkout time and the order's total stays inside the budget;
(b) when the shop genuinely has nothing that satisfies everything, finishing the way the request describes: with the no-purchase tool, or, when the request asks for the listing that comes closest in that case, with a checkout of that listing.

== Matching a listing to a requirement ==
1. Kind. A listing's category label settles what kind of product it is. Only a label for the requested item as a whole qualifies; a neighbouring label never does.
2. Attribute values. A property requirement holds when the listing's attribute for that property carries the value. Ignore how the key is spelled (spaces, underscores, camelCase, prefixes); one entry of a comma- or slash-separated list suffices. Compare values word by word: case, singular/plural and plain synonyms are allowed, and a value that contains the required value's words side by side, in their order, is a match ("deep teal" satisfies teal, "extra roomy fit" satisfies roomy). Other words are no match, nor are the wanted words split by another word ("soft grip handle" is no "soft handle"): a code or abbreviation is not the word behind it, "teal green" is not "sea green", and a value that only implies the wanted one by going further ("none" for "low", "not needed" for "quick") is no match either. Numbers are compared after unit conversion; a limit includes its edge value; a range has to sit entirely within a limit. When other listings write a wanted span ("a to b unit") as a value of their own, that label decides: only a listing whose entry restates the span meets it, and a figure converted into the span does not.
3. Which key decides. When a key names the requirement's property (under that name or a plain synonym), that key decides, and a different value there fails the listing no matter what other keys or the title say. A key about some other property, even a related one (the handle's colour says nothing about the body's colour), settles nothing. Kyo appends [evidence] lines to opened listings showing the keys whose names match a requirement.
4. No such key. A requirement the ledger marks as stated by no opened listing neither passes nor fails a listing, so it never decides a pick or an ending. Otherwise the wanted value counts when another attribute states it (say, a list of styles or features containing it) or the title or description does. A number measuring something else does not count (a weight is no capacity), nor does a bare figure whose text leaves unclear what it measures.
5. Labels with values. A category label that itself states a requirement's value (a season or an audience in the label) satisfies that requirement.
6. Values found on another listing. Some requirements point to another listing rather than naming a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle these first. Track that listing down (search its category, brand, shop or words) and open its full page with the single-listing or side-by-side tool, because a search row is not enough; if that page refers on to a further listing for the property, find and open that one in the same way. The value comes from the last page, and an order only counts when every page of the chain has been opened. When the requirement wants a value that works or goes together with that page's value (compatible with it, its counterpart, used alongside it), the answer is not the page's value but its partner under the store's pairing rule: a note, on that page or on other listings of its category, saying which value goes together with which. Read the rule in its stated direction (A going with B says nothing about what goes with B); a value the rule pairs with itself is the same value. Until such a rule is found, look on a few more listings of that category rather than copying the page's value. If the wanted value is anything except that other listing's value, the rule rules out only listings with exactly that value, and on its own it is never grounds for ending without a purchase.
7. A listing you are about to buy that refers you to another page about itself: read that page before buying.
8. Items that must go together (an accessory for a device): pick them as a pair, using the listings' own words (a list of what it fits, a model or series code carried by both pages). A device's model number and the names it is also sold as are one device. A line about some other part ("part X also fits Y", "more on part X") speaks for part X only, never for the listing it appears on. When several items must fit the same device, start from those items: open listings meeting each one's own requirements, note the device codes they name, and search those codes for a device that meets its own requirements and is named by a fitting listing of every such item. If no accessory meeting its own requirements fits the chosen device, move to another device that meets its requirements and that such an accessory fits; while that switch is untried, never end without a purchase and never check out with an item's line missing.
9. A requirement that holds only under a condition on another listing ("x, only if the device turns out to be v") lapses only when that listing's page states a value other than v; while the page leaves the condition open, meet the requirement.
10. Exclusions and closed lists are binding. Wishes ("ideally", "would be nice") only break ties.
11. Sold-out listings are never bought. Price times quantity, added up over all lines, must not go over the budget (landing exactly on it is allowed).

== Using the shop ==
- Every tool call uses one of a limited number of steps; reasoning costs nothing. Tool names, argument names and result layouts belong to this particular shop, so read each tool's description and schema and pass the exact argument names and permitted values.
- Searching may be restricted: a category filter, a keyword mode that perhaps reads only the opening words of the query, pages holding a few rows, a token or offset that fetches the next page. If the answer hinges on seeing every match, set the reported total against the rows you have and keep paging with the precise token or offset the shop hands back. Keyword queries do best kept short with the value first (the wanted value, then the product noun). An empty result from a filtered search usually signals a wrong filter value or filter parameter: try again with the label spelled exactly as result rows show it, or with a different filter parameter.
- A search row only summarises a listing. Shops divide a listing's record in different ways: one view may hold its attributes (properties), another only its offer and description, and a side-by-side view may hold either. As soon as Kyo learns which tool shows attributes, the ledger says so. Every property requirement is judged on those attributes, so read them for any listing you might buy; one call covering several listings saves steps when it shows what you need. Do not open a listing again unless something changed since you read it.
- All questions to the shopper go through Kyo's interviewer; you never message the shopper. The answers show up in the conversation and in the ledger.
- The request may also demand steps along the way: an approval, a message, something to check or read first. The ledger lists them. The ones due immediately before the order or the ending are carried out by Kyo when you call the checkout or ending tool, so do not wait for them and do not end without a purchase on their account; do any others yourself with the shop's tools (product questions still go through the interviewer).

== Choosing ==
- Cheapest pick (lowest total, smallest price): the order has to contain the lowest-priced listing out of ALL listings that fit. Establish the whole fitting set (page through it: rows rarely come in price order, and Kyo itself fetches the further pages of a short result), check candidates from the lowest price up, and take the first that fits completely. If a search accepts a maximum price, use it to list the category below your current best pick's price, so only cheaper candidates remain to check.
- Most or least of some measure: compare every fitting listing on that measure and take the extreme one; a listing silent on the measure cannot win.
- Any other request: any listing that meets everything is correct, so do not burn steps looking for a better one.
- Put each item in the cart as soon as you have opened a listing meeting all its requirements (with a cheapest, most or least pick, once the fitting set has been compared), then go on to the next. Carted lines often sell out shortly after being added, and carting early leaves steps for a replacement.
- Several different items mean one cart line per item, each at its own requested quantity.

== Changes ==
- Shopper or shop may change the task: a new budget, a requirement added or removed, a price change, a listing selling out. Such changes arrive inside tool results, frequently as a refused checkout; a note counts whichever tool's result carries it, and the ledger always lists the requirements in force. After a refusal the episode continues: apply every change, repair the cart, and check out again.
- Shortly before checking out, re-read each chosen listing's record: price and stock can change without any notice.

== Finishing ==
- Leave exactly the chosen lines in the cart (take out everything else), view the cart once, then check out with a call that stands alone.
- Ending without a purchase is right only after a genuine search finds no listing meeting everything. Being low on steps shows no such thing; in that case check out the best cart you have. Word that ending as the request asks, and if a remark is required keep it brief and name no listing.
- Each reply must include at least one tool call. Keep your text brief: what you know, then the next move."""


BRIEF = """Job: a shopping agent needs a structured reading of the shopper's request below, made with the store tools listed alongside it. Produce exactly one JSON object, written compactly on one line with no extra text, containing the fields described here.

1) The products
"items" -> a list giving each separate product to buy its own entry {"name", "shelf", "quantity", "musts"}:
 - name: what kind of product it is, worded as the shopper words it;
 - shelf: when some tool enumerates category labels, the matching label character for character; otherwise "";
 - quantity: integer count of units of that single listing; 1 by default; a set or pair sold as one listing is 1;
 - musts: list of hard constraints on this product, each written "property: constraint" and keeping the precise value, limit, exclusion or closed list.
Guidance for musts:
 - Call the property by the word the shopper used ("meant for a lakeside occasion" -> "occasion: lakeside"; "a 12-inch blade" -> "blade length: 12 inch"), never by a nearby property.
 - A constraint that gets its value from some other listing is written as a single must carrying the full description of that listing, e.g. "rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid". Whatever identifies that other listing (its seller, its kind, a trailing "with <property> <value> and <property> <value>") belongs inside that one must and is never a separate must of the product.
 - A constraint that holds only under a condition ("should the X turn out to be v, ...") stays one must with its condition written in, e.g. "colour: black, only if the first item is the large size"; never drop the condition.
 - A must states a value a listing can carry. A property whose value the request leaves for the shopper to give when asked is no must: write it under ask_first. How the shopper uses the product (how much at a time, a daily routine) is context and no must.
 - Money, budget, stock and quantity never go into musts.
 - A line saying only that the shopper cares about some property (is particular or has views about it) without giving its value is no must: that property is left to ask the shopper about.
 - Background in the request (the person it is for, where it will be placed, a past problem, the occasion) is context and not a must, except where it spells out a property's value or limit. Words forming part of the product's own name (a season, an audience or a purpose mentioned with it) stay in name; when such words give the value of a property a listing can carry (a purpose, an occasion, a season), also write that as a must ("trays for garden parties" -> name "trays for garden parties", must "occasion: garden party").
 - A phrase that joins products with "and" under one "I want ..." and one quantity (often the title of a single store department) is a single item. Split into several items only when each product carries its own quantity or its own constraints.

2) Price and choice
"budget" -> the shopper's maximum total spend as a number in the unit the request uses, or null.
"budget_unit" -> the name of that unit, e.g. "cents", or null.
"budget_scope" -> "total" for a cap on the full basket, "each" for a cap per item, null without a cap.
"in_stock_only" -> true, except when the request clearly allows unavailable items.
"pick" -> "cheapest" if the shopper asks for the lowest price or smallest total; "most" or "least" if they want the maximum or minimum of some measure; "any" in every other case (saying money is tight does not make a pick rule).
"pick_measure" -> for "most"/"least", the measure in question; otherwise "".

3) Working rules
"messages" -> integer number of messages the agent may send the shopper; null when the request is silent.
"steps" -> integer number of steps, moves or tool calls allowed; null when unstated.
"lookups" -> those musts, copied verbatim, whose value must first be read off another listing the request refers to; [] when none. A must tying this product to another product bought in this same order (fits it, works with it) is no lookup.
"rules" -> each instruction about how to proceed or how to finish, 12 words at most apiece.
"wishes" -> things the shopper would only like to have.
"ask_first" -> things the request explicitly wants confirmed with the shopper before choosing, and properties whose value the shopper will give once asked; [] when none.
"process" -> extra actions the request demands besides locating and buying the right products (obtaining the shopper's approval, telling or showing them something, confirming something with them, reading something first, using a specific tool). Each is {"step": the action in 15 words at most, "when": "choose" for before the products are chosen, "order" for immediately before checkout, "end" for just before ending without a purchase, "messages": number of messages to the shopper it needs, 0 when none}. Use [] when there are none. Constraints on the items, the budget, stock, the message and step limits, and the way to end when nothing fits (saying that nothing fits, putting forward or buying the closest listing) are not process steps.
"quiet_decline" -> true if the request says alternatives must not be suggested or named when nothing fits; false if it asks for them; null if it does not say.
"closest_if_none" -> true if the request says that, should nothing meet every requirement, the listing coming closest (for instance the one missing the fewest requirements) is to be put forward, taken or bought; false otherwise.

4) Tools
"roles" -> object mapping each job to the exact tool name, null where no tool does that job. Jobs: "search"; "open" = shows a single listing's attributes or properties (when two tools show a single listing, the one described as giving attributes, properties or specifications); "offer" = shows a single listing's current offer such as price and stock, when this is another tool than "open"; "compare" = several listings in full; "ask" = sends a message to the shopper; "cart" = modifies the cart; "cart_view" = displays the cart; "buy" = checks out; "refuse" = ends without a purchase.
"buy_args" -> fixed arguments under which the "buy" tool checks out (for a tool with a mode argument, that mode); {} otherwise.
"refuse_args" -> fixed arguments under which the "refuse" tool ends without a purchase; {} otherwise.
"""


MODES = """The store tool <<TOOL>> does two jobs: some arguments make it check out, others make it end without a purchase. From its schema below, name only the arguments that choose the job. Reply with ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase}

SCHEMA:
"""


PLAN = """Task: decide which questions a shopping agent sends to its shopper, and in what order.

Background. Every shopper keeps a handful of firm preferences out of the written request; each concerns one property of one product being bought. Only a question naming that very property brings the preference out, and the shopper replies solely about the first property a question mentions, so one question is worth one property.

Material provided: the request; the items together with what the request has already fixed for each; a numbered list of the attribute keys found on opened listings in each item's category, with the number of listings showing each key and a few sample values. A hidden preference nearly always concerns one of these keys; one that no key names is rare and worth a question only on a clear story hint, or for an item with no keys listed.

Priority when choosing properties:
first - anything the request itself tells you to confirm with the shopper (given under ASK FIRST when present), each expressed as the property it is about;
second - properties the story hints at. Requests often allude to a hidden preference without stating its value: a line about something the shopper cannot assess or leaves up to you, a person whose taste is decisive, how the product looks or the surroundings it must suit, who is going to use or wear it, or what will go inside it. For every such line pick the single property it alludes to and the item concerned, keeping the story's order. Name it by that item's listed key the line is about, written as the key reads (an allusion to appearance goes to whichever listed key describes the look, such as colour, finish, pattern or style but not material, and is not fixed by the request); only when none of the item's listed keys fits, give it its everyday name (the wearer means gender, appearance means colour, or finish when the request fixes colour, or colour when it fixes finish);
third - the listed keys, strongest first: properties a buyer of that product type settles before anything else (frequently size, capacity or dimensions, type, material or colour, whichever drives that purchase) and keys present on most of that item's listings; keys found on just one or two listings are weak. Every property picked without a story line behind it must be one of its item's listed keys.

Never pick a property the request already fixes for that item, a synonym of a property already picked, a key that enumerates purchasable variants (mixed sizes, colours, pack counts), identifiers, warranty or other logistics. For several items, put all items' hinted properties first and then alternate from item to item.

The "label" field holds the property's everyday name. For a listed key write every word of the key in normal spelling ("lid colour" from lidColour, "rim finish" from rim_finish; not reduced to "colour" or "finish"). Give the name alone: no sample values, no parentheses, never two properties.

Output, best first and at most <<COUNT>> entries, as a single JSON object with nothing around it: {"ask": [{"item": the item's number, "label": "...", "hint": "the story's words it rests on, 8 at most, or an empty string"}]}"""


HEAR = """Below are questions a shopping agent sent to a shopper, with the shopper's answers. Turn the answers into a single JSON object (no other text) of this form:
{"needs": [{"item": number of the item, "need": "property: constraint", "points": true if the value has to be taken from another listing, otherwise false}], "drop": [{"item": number of the item, "property": "a requirement the shopper no longer insists on"}], "free": [{"item": number of the item, "property": "..."}], "budget": a new overall limit as a number in the unit of the limit in force, or null}

How to fill it in:
1. needs: each firm preference an answer expresses, keeping its precise value or limit together with the unit ("at least 30 cm", "oval", "not wicker"). Use the property the answer names, even when it is not the one the question asked about; answers about the same property combine into a single need. Reproduce the value exactly in the answer's wording, with its nouns and units ("at least 3 shelves", not "at least 3").
2. A value expressed by reference to another listing (alike, unlike, close to or going together with the value on a product of some shop, that product identified by a few of its properties) forms ONE need, on the property the answer names, with "points": true. Its value is the answer's text reproduced verbatim from the comparing words through to the end of the description, including anything it says about a listing that page may mention; never trimmed or rephrased. The properties identifying the other product are not needs of this item.
3. free: for each question the shopper says they don't care about, its property.
4. Homely or indirect phrasing that still names a value counts as stating that value."""


NOTE = """During a shopping agent's run the store displayed the message below; its author may be the shopper or the store. Describe its effect in one JSON object and nothing else:
{"kind": "change" when it alters the task (budget, requirements, a price, availability, or a refusal of the order); "error" when it merely reports a malformed or rejected call; "other" otherwise,
 "budget": a new overall limit as a number in the request's unit, else null,
 "add": [{"item": item number, "need": "property: constraint"}],
 "drop": [{"item": item number, "need": "the requirement that is no longer wanted"}],
 "gone": [listing ids the message describes as unavailable],
 "sold_out": true when it says a chosen listing has gone, sold out or become unavailable, even with no listing named,
 "price_moved": true when it reports a price change,
 "summary": "a single short sentence"}
List every change in the message. When a requirement is replaced, record a drop for the old one and an add for the new one."""


CHECK = """You are the final gate before a shopping agent pays. Using only the records given, judge whether the cart is exactly correct.

Interpreting records:
* A requirement holds when the attribute for its property states the value. Key spelling is irrelevant, a single member of a list suffices, case, plural forms and ordinary synonyms are accepted, and a value that includes the wanted value's words side by side passes ("deep teal" passes teal; "soft grip handle" fails "soft handle"), while one that only implies it by going further ("none" for "low") does not. Convert units; limits are inclusive; a range has to fall completely within the limit.
* When some attribute key names the requirement's property, that attribute rules, and a different value there means failure. Keys about another property, related ones included (a handle colour tells nothing about the body colour), rule on nothing.
* Without an attribute for the property, the wanted value written under another attribute, in the title or in the description is accepted; not, however, a number measuring something else (weight is not capacity), nor a lone figure whose text does not say what it measures.
* A requirement the category label itself expresses is satisfied by that category. The label must correspond to the requested item as a whole.
* A rule of differing from another listing's value is satisfied unless the records show that very value.
* A rule of going together with another listing's value is met by the partner value the ledger names for it, never by that listing's own value.
* A rule to fit another item of the order holds when one of the two pages names the other (its model number or a name it is also sold as) as fitting; a line about some other part ("part X also fits Y") speaks for that part only.
* A requirement under a condition on another listing is in force unless that listing's record shows the condition is not met.
* A record flagged by Kyo for not restating a wanted span that other listings write as a label does not meet that requirement.
* A requirement the ledger marks as stated by no opened listing neither passes nor fails a listing.
* When a listing has several records, the latest one counts.

The cart is correct only if:
* each requested item appears as its own line at the requested quantity, and the cart holds nothing more;
* no line is shown as out of stock;
* the sum of price times quantity over all lines does not exceed the budget;
* under a cheapest, most or least pick rule, none of the other listings shown clearly meets every requirement while beating the cart's choice on that rule.

Answer ok=false only for a problem the records plainly show, or for a requirement no record covers (unless the ledger marks it as stated by no opened listing). Output a single JSON object and nothing else: {"ok": true or false, "problems": ["a brief concrete problem", ...], "better": [ids of listings that should take a cart line's place, best first]}"""


READING = """
* When an attribute key names a requirement's property, that attribute rules (key spelling is irrelevant, one member of a list suffices, synonyms and converted units are accepted, a value including the wanted value's words passes, as "deep teal" passes teal, and limits are inclusive).
* One property can sit under two spellings of a key (spaced and underscored) with different values: the listing carries the value when either entry does, and only both together can rule it out.
* Keys about another property, related ones included (a handle colour tells nothing about the body colour), rule on nothing.
* Without an attribute naming the property, the wanted value written under another attribute, in the title or in the description is accepted; a lone figure that does not say what it measures proves nothing.
* A requirement the category label itself expresses is satisfied by that category, and the label must correspond to the item.
* A rule of differing from another listing's value is satisfied unless the records show that very value.
* A rule of going together with another listing's value is met by the partner value the ledger names for it, never by that listing's own value.
* A rule to fit another item of the order holds when one of the two pages names the other (its model number or a name it is also sold as) as fitting; a line about some other part ("part X also fits Y") speaks for that part only.
* A requirement under a condition on another listing is in force unless that listing's record shows the condition is not met.
* A requirement the ledger marks as stated by no opened listing neither passes nor fails a listing.
* Out-of-stock listings, and listings over the budget, never fit."""

AUDIT = """The shopping agent intends to stop without buying anything, believing no listing fits. Go through the listing records given and test them against the requirements, interpreting them as follows:""" + READING + """
* Records tagged ROW come from search result rows only; their page with its attributes has not been read, so a property missing from the row can neither pass nor fail them.
Output a single JSON object and nothing else: {"fits": {"item number": [ids of listings that seem to satisfy every requirement, lowest price first, 4 at most]}, "open": [ids of ROW listings of a requested kind and within budget whose page might prove they fit, most promising first, 6 at most], "reason": "a single sentence"}"""

UNMIX = """Context: a product's hard requirements were extracted from a shopper's request. Some of them borrow their value from a different listing the request refers to, and the words singling out that different listing (its kind, its seller, properties attached to it with "with ...") sometimes end up wrongly extracted as requirements of the product itself. Read the request and decide which entries under OTHER REQUIREMENTS do nothing but describe that different listing.
Output a single JSON object and nothing else: {"describe_other": [such requirements, verbatim; leave it empty when all of them truly apply to the product]}"""

TRAIL = """A shopping agent will soon pay. Certain requirements do not state their value but borrow it from another listing, and that listing's page can in turn refer onward to yet another listing for the property. Payment only counts when the agent has fully opened every listing on each such chain.
Trace every chain starting from the requirement's own wording, moving one link at a time by what the pages themselves say, with the help of the records given: OPENED means the agent has opened that page, ROW means it is known only as a search result row with its page unread. Give the links in order, and end a chain at the first link that is not OPENED or cannot be identified, because what its page says is not yet known.
Output a single JSON object and nothing else: {"chains": [{"need": the requirement, verbatim, "links": [{"id": the listing id when one of the records is that link, otherwise "", "about": the way the requirement or the preceding page describes that listing, "seller": the shop or brand name that description gives for it, or ""}], "done": true when the final link is an OPENED page stating the property's value itself, otherwise false, "value": the property's value as written on that final page, or "", "relation": "pair" when the requirement wants a value that works or goes together with that value (compatible with it, its counterpart, used alongside it), "unlike" when it wants a different one, "limit" when it sets a bound from it, "same" otherwise, "partner": for "pair", the value the store's own pairing rule (a note on any OPENED record saying which value of this property goes together with which) gives for the final page's value, read in the rule's stated direction, or "" when no record states such a rule}]}"""

GATE = """The shopping agent is about to <<STAGE>>. On the way there, the shopper's request requires the steps listed below. From the numbered RECORD of the agent's actions, determine for each step that falls due before it can <<STAGE>> whether it has been completed for the cart in its present state.
Rules:
(1) Only a RECORD line showing the step being carried out completes it. For a step concerning the cart or the order, that line has to be later than the cart's last change, and the reply on it must give consent. Asking about a product's properties completes no step.
(2) For a step still outstanding, name the one store call that should come next: a tool from TOOLS plus its arguments. Messages to the shopper go through the messaging tool as one short plain sentence or question about that step only, mentioning no product properties or values the step does not need; a total or sum they state is the exact figure CART NOW gives.
(3) Name no call when all due steps are complete, or when the remaining messages or steps cannot achieve the outstanding one.
Output a single JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [RECORD line numbers showing it completed]}], "call": {"tool": "...", "args": {...}} or null, "hold": anything a reply asks to have changed before continuing, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """One listing in the shopping agent's cart has just become unavailable. Look through the other listing records given for listings that could take over its cart line, interpreting them as follows:""" + READING + """
Output a single JSON object and nothing else: {"fits": [{"id": the listing id, "price": its price as a number, taken from the record} for each listing that seems to satisfy all requirements of the item whose line sold out, 6 at most], "reason": "a single sentence"}"""

SHAPE = """Here is a store tool's output. Point out its components by the precise key names that appear in the output and in the tool's parameters. Output a single JSON object and nothing else:
{"rows": key under which the listings are listed, "count": key with the total number of matching listings (the full count, not the number displayed) or null, "next": key whose value requests the next page or null, "next_arg": name of the tool parameter that accepts that value or null, "price": key inside a row giving the listing's price or null, "stock": key inside a row telling whether the listing is purchasable now or null}"""


_KIND_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def cut(value: Any, limit: int, depth: int = 0) -> Any:
    """Shorten every string inside a value to `limit` characters (and anything nested past eight levels)."""
    if depth > 7:
        return "..."
    if isinstance(value, dict):
        return {key: cut(inner, limit, depth + 1) for key, inner in value.items()}
    if isinstance(value, list):
        return [cut(inner, limit, depth + 1) for inner in value]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def slim(observation: Any) -> Any:
    """A long result list with its fields shortened: such rows are scanned, and a side-by-side view stays whole."""
    if isinstance(observation, dict):
        return {k: slim(v) for k, v in observation.items()}
    if isinstance(observation, list) and len(observation) >= 5 and all(isinstance(r, dict) for r in observation):
        return cut(observation, ROW_CHARS)
    return observation


def render(observation: Any) -> str:
    text = to_json(cut(slim(observation), SHOW_CHARS))
    return text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + "...(cut)"


def parse_object(text: Any) -> dict[str, Any] | None:
    """The JSON object in a model reply: the whole text, a fenced block, or the span between the outer braces."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        got = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if not 0 <= start < end:
            return None
        try:
            got = json.loads(body[start:end + 1])
        except ValueError:
            return None
    return got if isinstance(got, dict) else None


def plain(key: Any) -> str:
    """A key in plain lower-case words: camel case split, separators turned into spaces."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", spaced).strip().lower()


def keep_comparing(need: str, said: str) -> str:
    """A need taken from another listing whose value lost the comparing words that open it in the answer ("other than",
    "unlike", ...) gets them back verbatim: without them a wanted difference reads as a wanted match."""
    prop, _, value = need.partition(":")
    value = value.strip()
    at = said.lower().find(value.lower()[:60]) if len(value) >= 12 else -1
    if at < 0:
        return need
    found = re.search(r"\b(?:(?:anything|something|nothing)\s+)?(?:other\s+than|except(?:\s+for)?|apart\s+from|unlike|different\s+(?:from|to|than)|anything\s+but|not)\s+$",
                      said[max(0, at - 40):at], re.I)
    return f"{prop}: {found.group(0).strip()} {value}" if found else need


def stems(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in plain(text).split())


def key_names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Whether an attribute key (as stems) names a requirement's property (as stems)."""
    return bool(prop and key) and (prop <= key or (len(key) >= 2 and key <= prop))


_NOT_PLAIN = frozenset("not no non without except other than unlike same least most only if fit fits match matches as any anything".split())


def word_values(need: str) -> list[str]:
    """The wanted values of a plain requirement ("property: a or b"), word values only: none for a number or limit, an
    exclusion, a condition or a value taken from another listing."""
    value = need.rsplit(":", 1)[1].lower() if ":" in need else ""
    if re.search(r"\d", value) or _NOT_PLAIN & set(re.findall(r"[a-z]+", value)):
        return []
    parts = [p.strip(" .'") for p in re.split(r",|/|;|\bor\b|\beither\b", value)]
    return [p for p in parts if re.fullmatch(r"[a-z][a-z' &-]*", p) and len(p.split()) <= 3]


def says_phrase(wanted: str, value: Any) -> bool:
    """Whether one entry of a value carries the wanted words side by side and in order ("deep teal" says teal; a word
    between them, as in "soft grip handle" for "soft handle", breaks it)."""
    want = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", wanted.lower())]
    for part in re.split(r"[,;/|]", str(value).lower()):
        have = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", part)]
        if want and any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1)):
            return True
    return False


_EXCLUDING = r"(?:(?:anything|something)\s+(?:else\s+)?(?:but|other\s+than|except)|other\s+than|except(?:\s+for)?|apart\s+from|unlike|not|never)"


def keep_excluded(need: str, request: str) -> str:
    """A plain word must that the request only ever names after excluding words ("anything else but <property> <value>")
    gets "not" back: the brief sometimes drops it, and the wanted value then reads as the one value ruled out."""
    prop, _, value = need.partition(":")
    value = value.strip()
    if not word_values(need) or not prop.strip():
        return need
    text = re.sub(r"\s+", " ", request.lower())
    said = [m.start() for m in re.finditer(r"\b" + re.escape(value.lower()) + r"\b", text)]
    lead = re.compile(r"\b" + _EXCLUDING + r"\s+(?:the\s+)?(?:" + re.escape(prop.strip().lower()) + r"\s+(?:is\s+|of\s+)?)?$")
    if said and all(lead.search(text[max(0, at - 60):at]) for at in said):
        return f"{prop.strip()}: not {value}"
    return need


_SPAN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*([a-z]*)")


def spans_of(text: Any) -> set[tuple[float, float, str]]:
    """The spans ("a to b unit") written in a text, unit reduced to its first two letters ("inches" -> "in")."""
    return {(float(a), float(b), unit[:2]) for a, b, unit in _SPAN.findall(str(text).lower())}


def restates_span(wanted: set[tuple[float, float, str]], text: Any) -> bool:
    """Whether a text writes one of the wanted spans itself (same figures; the same unit when both name one)."""
    return any(a == x and b == y and (not u or not v or u == v) for a, b, u in spans_of(text) for x, y, v in wanted)


def kinds_of(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    return [kind] if isinstance(kind, str) else [k for k in kind or [] if isinstance(k, str)]


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; whatever cannot be bent is described in `faults`."""
    if not isinstance(schema, dict):
        return value
    kinds = kinds_of(schema)
    if kinds and not any(k not in _KIND_TESTS or _KIND_TESTS[k](value) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
            elif kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
            elif kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            elif kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            elif kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
            else:
                continue
            break
        if not any(k not in _KIND_TESTS or _KIND_TESTS[k](value) for k in kinds):
            faults.append(f"{where} must be of type {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            same = [o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()]
            if same:
                value = same[0]
        if value not in options:
            faults.append(f"{where} must be one of {options}")
    if isinstance(value, str):
        longest, shortest = schema.get("maxLength"), schema.get("minLength")
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
        if isinstance(shortest, int) and len(value) < shortest:
            faults.append(f"{where} is shorter than allowed")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            faults.append(f"{where} is under the minimum {low}")
        if isinstance(high, (int, float)) and value > high:
            faults.append(f"{where} is over the maximum {high}")
    if isinstance(value, list):
        most, fewest = schema.get("maxItems"), schema.get("minItems")
        if isinstance(most, int) and len(value) > most:
            faults.append(f"{where} takes at most {most} entries")
        if isinstance(fewest, int) and len(value) < fewest:
            faults.append(f"{where} takes at least {fewest} entries")
        if isinstance(schema.get("items"), dict):
            value = [conform(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            faults += [f"{where}.{name} is not an accepted argument" for name in value if name not in props]
        faults += [f"{where}.{name} is required but missing" for name in schema.get("required") or [] if name not in value]
        value = {k: conform(v, props[k], f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def as_int(value: Any) -> int | None:
    """A whole number from an int, a whole float or a string of digits; never from a bool."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def as_budget(value: Any) -> float | None:
    """A positive amount from a number or a string of one; never from a bool."""
    if isinstance(value, str) and re.fullmatch(_NUMBER, value):
        value = float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return value
    return None


def id_like(value: Any, named: bool = False) -> bool:
    """A listing id: a code of 3-64 characters without spaces carrying a digit (any code under an id-like key), or a
    whole number under an id-like key."""
    if isinstance(value, int) and not isinstance(value, bool):
        return named and value >= 0
    if not isinstance(value, str) or " " in value.strip() or not 3 <= len(value.strip()) <= 64:
        return False
    return named or bool(re.search(r"\d", value))


def id_named(key: Any) -> bool:
    return bool(re.search(r"id|sku|code", str(key).lower()))


def id_word(key: Any) -> bool:
    """A key whose last word is id or sku (product_id, productId, SKU), not a word ending in those letters (paid)."""
    return bool(re.search(r"(?:^|[^A-Za-z])(?:id|sku)$", str(key).lower()) or re.search(r"[a-z](?:Id|ID|Sku|SKU)$", str(key)))


def pick_id_key(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: id-like strings unique per row; a key named like an id wins, then the longest values."""
    best = None
    for key in rows[0]:
        named = id_named(key)
        values = [row.get(key) for row in rows]
        if any(isinstance(v, int) for v in values) and not id_word(key):
            # a whole number is an id only under a key that names one: a count or size never is
            continue
        if not all(id_like(v, named) for v in values):
            continue
        values = [str(v).strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (named, sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def holds_id(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether some list in a result holds a record carrying this id (a cart reply still listing the line)."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(holds_id(v, ident, depth + 1) for v in node.values())
    if not isinstance(node, list):
        return False
    if any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, (str, int))] for v in node):
        return True
    return any(holds_id(v, ident, depth + 1) for v in node)


def failed(observation: Any) -> bool:
    """Whether a reply reports a failed call: an error field, a false ok/success flag, a failing status, or a bare
    error text."""
    if isinstance(observation, str):
        return bool(re.match(r"\s*(error|failed|failure|invalid)\b", observation, re.I))
    if not isinstance(observation, dict):
        return False
    status = observation.get("status")
    return (bool(observation.get("error") or observation.get("errors")) or observation.get("ok") is False
            or observation.get("success") is False or (isinstance(status, str) and status.strip().lower() in _FAILED))


def store_note(roles: dict[str, Any], name: str, observation: Any) -> Any:
    """A result as notes are read from it: a bare text reply counts only from a tool with a store job (a message
    tool's text is the shopper's answer)."""
    if isinstance(observation, str) and name in {t for job, t in roles.items() if job != "ask"}:
        return [observation]
    return observation


def scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def named_pairs(value: Any) -> dict[str, Any] | None:
    """Attributes given as a list of small records, each a name field and its value (a unit after it kept):
    {name: value}, when two or more."""
    if not isinstance(value, list) or len(value) < 2:
        return None
    out: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, dict) or not 2 <= len(entry) <= 3 or any(str(k).lower().endswith("id") for k in entry):
            return None
        name = next((k for k in entry if re.search(r"name|key|label|attr|prop|field", str(k), re.I)), None)
        rest = [v for k, v in entry.items() if k != name and scalar(v)]
        if name is None or not isinstance(entry[name], str) or not rest:
            return None
        out[entry[name]] = rest[0] if len(rest) == 1 else " ".join(str(v) for v in rest)
    return out if len(out) >= 2 else None


def attribute_block(node: Any) -> dict[str, Any] | None:
    """The richest nested dict of plain values in a record (its attributes), or list of name/value records, when it
    has two or more."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        scalars = named_pairs(value)
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if scalar(v)}
        if scalars and len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
            best = scalars
    return best


def money(value: Any) -> float | None:
    """A price from a number, a short text carrying one (currency marks, thousands or decimal commas) or a dict
    holding one."""
    if isinstance(value, dict):
        return next((m for m in map(money, value.values()) if m is not None), None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    found = re.search(r"\d[\d.,]*", value) if isinstance(value, str) and len(value) <= 24 else None
    if not found:
        return None
    text = found.group().rstrip(".,")
    if "," in text and "." not in text and re.fullmatch(r"\d+,\d{1,2}", text):
        text = text.replace(",", ".")
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def answers(label: str, reply: str) -> bool:
    """A reply states the asked property's value: it repeats every word of the property's name."""
    words = [w for w in plain(label).split() if len(w) >= 3]
    text = " " + plain(reply) + " "
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", text) for w in words)


_TEXT_MARK = re.compile(r"(?:^|\s\|\s)([a-z][a-z0-9_]{1,30}):\s")


def text_value(raw: str) -> Any:
    """One field of a text record as the value a structured store would send."""
    raw = raw.strip()
    if raw in ("None", "null", ""):
        return None
    if raw in ("True", "False"):
        return raw == "True"
    if re.fullmatch(r"-?\d+(\.\d+)?", raw):
        return float(raw) if "." in raw else int(raw)
    pairs = [part.partition("=") for part in raw.split(";")]
    if all(sep for _, sep, _ in pairs):
        return {k.strip(): v.strip() for k, _, v in pairs if k.strip() and v.strip()}
    return raw


def text_record(text: Any) -> dict[str, Any] | None:
    """A listing written as one line of `key: value | key: value`, as a dict. Only a mark of a lower-case key opens a field,
    since titles hold bars of their own; a line that opens with no key, or carries no id, is ordinary text."""
    if not isinstance(text, str) or " | " not in text:
        return None
    marks = list(_TEXT_MARK.finditer(text))
    if len(marks) < 3 or marks[0].start() != 0:
        return None
    record: dict[str, Any] = {}
    for mark, after in zip(marks, marks[1:] + [None]):
        record.setdefault(mark.group(1), text_value(text[mark.end(): after.start() if after else len(text)]))
    return record if any(isinstance(v, str) and id_like(v) for v in list(record.values())[:2]) else None


def unfold(node: Any, depth: int = 0) -> Any:
    """A store reply with its text records turned into dicts, and a lone record lifted out of its wrapper."""
    if depth > 6:
        return node
    if isinstance(node, str):
        record = text_record(node)
        return node if record is None else record
    if isinstance(node, list):
        return [unfold(n, depth + 1) for n in node]
    if isinstance(node, dict):
        out = {k: unfold(v, depth + 1) for k, v in node.items()}
        only = next(iter(out.values())) if len(out) == 1 else None
        lifted = isinstance(only, dict) and isinstance(next(iter(node.values())), str)
        return only if lifted else out
    return node


def walk_keys(node: Any, depth: int = 0) -> list[str]:
    """Every key of a record, nested ones included."""
    if depth > 4:
        return []
    if isinstance(node, dict):
        return [str(k) for k in node] + [k for v in node.values() for k in walk_keys(v, depth + 1)]
    if isinstance(node, list):
        return [k for v in node for k in walk_keys(v, depth + 1)]
    return []


def words_of(observation: Any) -> str:
    """What a reply says: its strings of two words or more, else the reply itself."""
    if isinstance(observation, str):
        return observation
    found: list[str] = []
    stack = [(observation, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 4:
            continue
        if isinstance(node, str):
            if len(node.split()) >= 2:
                found.append(node)
        elif isinstance(node, (dict, list)):
            # children pushed in reverse so they are read in their own order
            stack += [(inner, depth + 1) for inner in reversed(list(node.values()) if isinstance(node, dict) else node)]
    return " ".join(found)[:800] if found else to_json(observation)[:400]


class Want:
    """One requested product and what is known of the shopper's wishes for it."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

    def live(self) -> list[str]:
        return [need for need in self.musts + self.told if need not in self.dropped]


class Episode:
    """State every stage shares, and the two wires: model inference and store calls."""

    def __init__(self, problem: dict[str, Any]):
        self.began = time.monotonic()
        world = problem["environment"]
        self.binding = dict(world["binding"])
        self.view = world["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or []
                      if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            params = tool["function"].get("parameters")
            self.schemas[tool["function"]["name"]] = params if isinstance(params, dict) else {"type": "object", "properties": {}}
        given = self.view.get("max_steps")
        cap = as_int(given)
        if cap is None and isinstance(given, float):
            cap = int(given)
        self.max_steps = cap if cap and cap > 0 else 30
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.cost = 0.0
        self.calls = 0
        self.trace: list[dict[str, Any]] = []
        # the task as read from the request
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Want] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        # what has changed along the way
        self.version = 0
        self.notes: list[str] = []
        self.queued: list[str] = []
        self.recent: list[str] = []
        self.leads: list[str] = []
        self.opened: set[str] = set()
        # the cart as Kyo has seen it change
        self.cart_lines: dict[str, int] = {}
        self.cart_view: Any = None
        self.cart_seen_at = -1
        self.cart_at = -1
        self.cart_touched = False
        self.buys = 0
        self.refused_at = -1

    # ----- derived facts -----
    def left(self) -> int:
        return self.max_steps - self.step

    def late(self, mark: float = CLOCK_STOP) -> bool:
        """Past `mark` seconds of the problem's wall time (any world's store or model may be slow)."""
        return time.monotonic() - self.began > mark

    def msgs_free(self) -> int:
        """Messages the interviewer may still send: those the order and ending steps need are kept back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return to_json(sorted(self.cart_lines.items()))

    def gate_due(self, stage: str) -> bool:
        # the request has steps at this stage, and the gate has not passed them for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, f"{self.cart_mark()}#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, plus one while an order step still needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.live()]

    def item_no(self, value: Any) -> int:
        return min(max(1, as_int(value) or 1), len(self.items))

    def ends(self, job: str, name: str, args: dict[str, Any]) -> bool:
        """Whether a call is the checkout (job "buy") or the ending without a purchase (job "refuse")."""
        if name != self.roles.get(job):
            return False
        fixed = self.brief.get(f"{job}_args")
        return all(args.get(k) == v for k, v in (fixed if isinstance(fixed, dict) else {}).items())

    def offer_tool(self) -> str | None:
        """The view of a listing's page and live offer: the offer tool when the store has one apart from the attributes
        view (an attributes list alone is not the page), else the single-listing tool."""
        return self.roles.get("offer") or self.roles.get("open")

    def fresh_cart(self) -> bool:
        # a cart view older than the last cart change still shows lines that are gone
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def fresh_budget(self, value: Any, line: str = "") -> float | None:
        """A new budget from a reply or a note, in the unit of the one in force. With the line's text, the reader's figure
        counts only when the line carries it; otherwise the line's one figure within tenfold of the budget in force does.
        Without it, a figure about a hundredth of the budget is that amount in the hundredfold unit (a reader's unit slip).
        One more than tenfold off is no change."""
        budget = as_budget(value)
        if budget is None or not self.budget:
            return budget
        figures = [float(f.replace(",", "")) for f in re.findall(r"\d[\d,]*(?:\.\d+)?", line)]
        if figures and not any(abs(f - budget) < 0.01 for f in figures):
            near = {f for f in figures if 0.1 <= f / self.budget <= 10}
            if len(near) != 1:
                return None
            budget = near.pop()
            budget = int(budget) if float(budget).is_integer() else budget
        if budget * 20 < self.budget and 0.1 <= budget * 100 / self.budget <= 10:
            budget = round(budget * 100, 2)
            budget = int(budget) if float(budget).is_integer() else budget
        return budget if 0.1 <= budget / self.budget <= 10 else None

    def fit_args(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        faults: list[str] = []
        shaped = conform(args, self.schemas.get(tool), "args", faults)
        return None if faults else shaped

    def buy_args(self, tool: str) -> dict[str, Any] | None:
        """The checkout's arguments: the brief's fixed ones, and a short confirmation in a required free text."""
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        args = dict(self.brief.get("buy_args") or {})
        for key in schema.get("required") or []:
            spec = props.get(key)
            if key not in args and isinstance(spec, dict) and "string" in kinds_of(spec) and "enum" not in spec:
                args[key] = CONFIRM[: spec.get("maxLength", 200)]
        return self.fit_args(tool, args)

    # ----- model wire -----
    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # rate limits, timeouts and server errors are worth another try after a pause; a spent key is not, and
                # the seconds are better used finishing with what the episode holds
                transient = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout has already used the client's 50 s, so it is resent once at most
                limit = 1 if status in (None, 408) else len(WAITS)
                if not transient or tries >= limit or self.waited + WAITS[tries] > WAIT_TOTAL or self.late(CLOCK_GATE):
                    break
                self.waited += WAITS[tries]
                time.sleep(WAITS[tries] * (0.8 + 0.05 * (tries % 9)))
                tries += 1
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
                   tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference request: (message, 200), or (None, the failure's status; 0 for a reply with no content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in (0, len(sent) - 1):
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        # the client may send a timed-out request twice: each try gets half of what is left, never more than its usual 50 s
        left = CLOCK_END - (time.monotonic() - self.began)
        if left < 10:
            self.trace.append({"kind": "llm_skip", "model": model, "left": round(left)})
            return None, 400
        _llm.timeout = int(min(50, left / 2))
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
            # the provider's own error inside a 200 reply
            status = as_int(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = parse_object((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    # ----- store wire -----
    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                if "observation" in data:
                    data["observation"] = unfold(data["observation"])
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                pages = {t for t in (self.offer_tool(), self.roles.get("compare")) if t}
                if name in pages and not failed(seen):
                    # these listings now count as opened in full (a search row never does)
                    for value in args.values():
                        self.opened.update(str(v).strip() for v in (value if isinstance(value, list) else [value])
                                           if isinstance(v, (str, int)) and not isinstance(v, bool))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if any(word in low for word in ("finish", "quarantin", "finaliz")):
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in low or "different action" in low:
                    self.step += 1
                else:
                    return None
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                self.broke = self.broke or status == 402
                return None
        return None


class Shelf:
    """Store memory: every listing record seen, the id key, which view shows what, each tool's result shape, and
    Kyo's record of the cart."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.id_key: str | None = None
        self.id_args: set[tuple[str, str]] = set()
        self.facts_from: dict[str, bool] = {}
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.row_keys: set[str] = set()
        self.text_search: tuple[str, str, dict[str, Any]] | None = None   # a search that took free words: tool, word argument, its fixed choices
        self.queried: set[str] = set()     # the words (as stems) of every free-text search so far
        self.nested = False
        self.offer_seen: dict[str, int] = {}
        self.compare_ok = 0
        self.compare_bad = 0
        self.cart_said = ""
        self.built: list[str] = []
        self.dropped: set[str] = set()

    # ----- records -----
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything that is not a listing."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and id_like(node.get(self.id_key), True):
            return str(node[self.id_key]).strip()
        if not (in_list or depth == 0 or any(isinstance(v, dict) for v in node.values())):
            return None
        key = pick_id_key([node])
        if key and (id_named(key) or str(node[key]).strip() in self.rows):
            return str(node[key]).strip()
        return None

    def facts_of(self, node: Any) -> dict[str, Any] | None:
        """A record's attributes: its nested block, else (in a store that nests none on a single listing) the plain
        fields it carries beyond those of the first result rows, when three or more."""
        block = attribute_block(node)
        if block or self.nested or not self.row_keys or not isinstance(node, dict):
            return block
        extra = {k: v for k, v in node.items() if k not in self.row_keys and scalar(v)}
        return extra if len(extra) >= 3 else None

    def absorb(self, observation: Any) -> None:
        """Store every listing record in a result; a list of rows teaches the id key on the way."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, list):
                rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
                listing = bool(rows) and len(rows) >= len(node) / 2
                if listing and not (self.id_key and all(id_like(r.get(self.id_key), True) for r in rows)):
                    key = pick_id_key(rows)
                    self.id_key = key or self.id_key
                    listing = bool(key)
                if listing and not self.row_keys:
                    # the fields of plain result rows: what a single-listing view adds to them are its attributes
                    self.row_keys = {k for r in rows for k in r}
                for inner in node:
                    walk(inner, depth + 1, listing)
                return
            if not isinstance(node, dict):
                return
            ident = self.ident(node, in_list, depth)
            if not ident:
                for inner in node.values():
                    walk(inner, depth + 1)
                return
            if ident not in self.rows:
                self.seen_order.append(ident)
                self.rows[ident] = {}
            record = self.rows[ident]
            # a flag of the listing's own record going from true to false means it can no longer be bought (and back)
            if any(record.get(k) is True and v is False for k, v in node.items()):
                self.unavailable.add(ident)
            elif any(record.get(k) is False and v is True for k, v in node.items()):
                self.unavailable.discard(ident)
            record.update(node)
            self.nested = self.nested or (not in_list and bool(attribute_block(node)))
            if self.facts_of(node):
                self.full[ident] = dict(record)
            elif ident in self.full:
                self.full[ident].update(node)

        walk(observation, 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
        groups: list[list[Any]] = [observation] if isinstance(observation, list) else []
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            for value in observation.values():
                if isinstance(value, list):
                    groups.append(value)
                elif isinstance(value, dict):
                    groups += [inner for inner in value.values() if isinstance(inner, list)]
        return [(i, row) for rows in groups for row in rows if isinstance(row, dict) for i in [self.ident(row, True)] if i]

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return to_json(cut(record, 400))[:3000] if record else "(no record seen)"

    def price_of(self, ident: str, missing: float | None = None) -> float | None:
        """The listing's price in its newest record, under the price key named for result rows."""
        value = money((self.rows.get(ident) or {}).get(self.price_key)) if self.price_key else None
        return missing if value is None else value

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record, and its stock field (the key named for result rows) does not say no."""
        if self.sold_out(ident):
            return False
        stock = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(stock, str):
            # a stock text: a count, or words saying it is gone
            stock = as_int(stock) if as_int(stock) is not None else not re.search(r"\b(out|sold|unavailable|not|no|none)\b", re.sub(r"[_-]", " ", stock.lower()))
        return not (stock is False or (isinstance(stock, (int, float)) and not isinstance(stock, bool) and stock <= 0))

    def kin_first(self) -> list[str]:
        """Every listing seen, newest first, but those of a requested item's category before the rest and, within each,
        opened pages before search rows: listings read only along a page chain or met in stray searches stay last."""
        cat_key = self.category_key()
        labels = {str(self.rows[i].get(cat_key) or "") for i in self.seen_order} if cat_key else set()
        kin = {label for label in labels if label and self.for_items(label)}
        order = list(reversed(self.seen_order))
        if kin:
            order.sort(key=lambda i: (str(self.rows[i].get(cat_key) or "") not in kin, i not in self.full))
        else:
            order.sort(key=lambda i: i not in self.full)
        return order

    def others(self, skip: list[str]) -> str:
        """Up to 15 other buyable listings, newest first; under a cheapest pick up to 25, only those priced below the
        dearest line (no other can beat a line), the read ones first, each part by price."""
        order = list(reversed(self.seen_order))
        cap = 15
        if self.ep.brief.get("pick") == "cheapest":
            # cheap listings of no use must not push out the read one just below the line's price
            top = max((self.price_of(i, _INF) for i in skip), default=_INF)
            order = [i for i in order if self.price_of(i, 0.0) < top]
            order.sort(key=lambda i: (i not in self.full, self.price_of(i, _INF)))
            cap = 25
        picked = [i for i in order if i not in skip and not self.sold_out(i)][:cap]
        return "\n".join(to_json(cut(self.full.get(i) or self.rows.get(i), 160))[:900] for i in picked)

    # ----- categories and attribute keys -----
    def category_key(self) -> str | None:
        """The row key holding the category label: on nearly every row, few short values, best when tools list them."""
        listed: set[str] = set()

        def collect(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            listed.update(o.strip().lower() for o in schema.get("enum") or [] if isinstance(o, str))
            for inner in schema.values():
                if isinstance(inner, dict):
                    collect(inner, depth + 1)

        for schema in self.ep.schemas.values():
            collect(schema, 0)
        values: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    values.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        scored = []
        for key, seen in values.items():
            if present[key] < 0.8 * total or len(seen) > 40:
                continue
            share = sum(1 for v in seen if v.strip().lower() in listed) / len(seen)
            scored.append(((-share, len(seen), -present[key], -sum(len(v) for v in seen) / len(seen)), key))
        return min(scored, key=lambda entry: entry[0])[1] if scored else None

    def for_items(self, label: str) -> bool:
        """Whether a category label names a requested item: the whole label, or a shared word of four letters or more."""
        whole = plain(label)
        long = {w for w in stems(label) if len(w) >= 4}
        return any(whole in (plain(item.shelf or ""), plain(item.name))
                   or long & {w for w in stems(item.shelf or "") | stems(item.name) if len(w) >= 4} for item in self.ep.items)

    def item_of(self, ident: str) -> int | None:
        """The requested item a listing is of: the only one, or the only one whose category its label names."""
        if len(self.ep.items) == 1:
            return 1
        cat_key = self.category_key()
        label = plain(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key else ""
        if not label:
            return None
        hits = [n for n, item in enumerate(self.ep.items, 1) if label in (plain(item.shelf or ""), plain(item.name))]
        return hits[0] if len(hits) == 1 else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values) over the opened listings, most carried first."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            named: set[str] = set()
            for key, value in (self.facts_of(record) or {}).items():
                name = plain(key)
                if not name or name in named:
                    continue
                named.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        groups = {label: g for label, g in groups.items() if self.for_items(label)} or groups
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                # long texts, variant lists and unique free-text values are no property to ask about
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                if any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept):
                    continue
                kept.append((name, slot))
            table += [(label, name, slot["n"], [v[:40] for v in slot["values"][:4]]) for name, slot in kept[:14]]
        return table

    def item_unseen(self) -> bool:
        """Whether every opened listing is of a category no requested item names while unopened listings remain."""
        cat_key = self.category_key()
        if not cat_key or not self.full:
            return False
        labels = {str((self.full.get(i) or {}).get(cat_key) or "") for i in self.full}
        return not any(self.for_items(label) for label in labels if label) and any(i not in self.full for i in self.seen_order)

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories of requested items (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            groups.setdefault(str(self.rows[ident].get(cat_key) or "") if cat_key else "", []).append(ident)
        pool = [ids for label, ids in groups.items() if self.for_items(label)] or list(groups.values())
        return sorted(pool, key=len, reverse=True)[: max(1, len(self.ep.items))]

    # ----- what each view shows -----
    def single_target(self, tool: str | None) -> str | None:
        """The one free-text argument of a tool that takes a single listing (and requires nothing else)."""
        schema = self.ep.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in kinds_of(v) and "enum" not in v]
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return None
        return target

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        shown = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if shown:
            return shown
        tool = self.ep.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or (untried) the store has a separate
        attributes view, one that showed them or one apart from the offer view."""
        seen = self.facts_from.get(self.ep.roles.get("compare") or "")
        if seen is not None:
            return seen is False
        return self.facts_from.get(self.facts_tool() or "") is True or bool(self.ep.roles.get("offer"))

    def compare_target(self) -> str | None:
        schema = self.ep.schemas.get(self.ep.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in kinds_of(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """Listings one side-by-side call may carry: what the store has shown to work, never above the schema's limit;
        a schema limit can overstate the real one, so a refused size always wins over it; a refused pair means one at a time."""
        props = (self.ep.schemas.get(self.ep.roles.get("compare") or "") or {}).get("properties") or {}
        spec = props.get(self.compare_target() or "") or {}
        most = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        if self.compare_bad:
            room = 1 if self.compare_bad <= 2 else max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        else:
            room = max(COMPARE_START, self.compare_ok)
        return min(room, most) if most else room

    def note_view(self, name: str, observation: Any) -> None:
        """Which fields each view shows, and which a listing has shown since the last change (to skip repeats)."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.ep.version, (have if version == self.ep.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.ep.version and keys <= have and ident not in self.gone

    # ----- result shapes and paging -----
    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """The keys of a row-list result (rows, match count, next-page value and the parameter taking it, a row's price
        and its can-buy flag), named once per tool by the light model and checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + f"\n\n## TOOL\n{name} | parameters: " + to_json(self.ep.schemas.get(name) or {})
                  + "\n\n## RESULT (first rows only)\n" + to_json(cut(sample, 120))[:4000])
        got = self.ep.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        params = (self.ep.schemas.get(name) or {}).get("properties") or {}
        homes = {"count": observation, "next": observation, "next_arg": params, "price": first, "stock": first}
        shape: dict[str, Any] = {"rows": rows}
        for part, home in homes.items():
            key = got.get(part)
            shape[part] = key if isinstance(key, str) and key in home else None
        total = observation[shape["count"]] if shape["count"] else None
        if shape["count"] and (isinstance(total, bool) or not isinstance(total, int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.ep.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the next-page parameter, its value or None on the last page), by the tool's
        result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def span_label(self, ident: str, need: str) -> bool | None:
        """For a requirement wanting a span ("a to b unit"): whether an attribute naming its property on this listing
        restates a wanted span; None when the requirement has no span or no such attribute was read."""
        prop, _, value = need.partition(":")
        wanted = spans_of(value)
        if not wanted:
            return None
        block = self.facts_of(self.full.get(ident) or {}) or {}
        named = [v for k, v in block.items() if key_names(stems(prop), stems(k))]
        return any(restates_span(wanted, v) for v in named) if named else None

    def span_flags(self, ident: str) -> list[str]:
        """Span requirements this listing misses while other opened listings write that span as the store's label:
        there the label decides, and a figure converted from other units does not meet it."""
        flags = []
        for _, need in self.ep.needs_now():
            if self.span_label(ident, need) is False and any(self.span_label(i, need) for i in self.full if i != ident):
                flags.append(need.split(":", 1)[0].strip())
        return flags

    def evidence(self, observation: Any) -> str:
        """[evidence] lines: for each listing with attributes in a result, the keys whose names match a requirement."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and self.facts_of(node) and ident not in ids:
                    ids.append(ident)
                    return
                for inner in node.values():
                    walk(inner, depth + 1)
            elif isinstance(node, list):
                for inner in node:
                    walk(inner, depth + 1)

        walk(observation, 0)
        needs = self.ep.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.ep.items) > 1
        lines = []
        for ident in ids[:12]:
            block = self.facts_of(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if key_names(stems(prop), stems(k))]
                if hits:
                    tag = ""
                    if self.span_label(ident, need):
                        tag = " [an entry restates the wanted span: met]"
                    elif prop in self.span_flags(ident):
                        tag = " [no entry restates the wanted span, which other listings write as a label: not met]"
                    # a value that only implies the wanted one ("none" for "low") is no match: say when no deciding
                    # value carries every word of a wanted value
                    wanted = word_values(need)
                    said = [str(v) for k, v in block.items() if key_names(stems(prop), stems(k))]
                    lack = ""
                    if wanted and not any(says_phrase(w, s) for w in wanted for s in said):
                        lack = (f" (none of these states the words of {' or '.join(wanted)} side by side: no match unless it is"
                                " the same word spelled otherwise)")
                    parts.append((f"(item {number}) " if many else "") + f"{prop} -> " + ", ".join(hits[:3]) + tag + lack)
                    continue
                # no key names the property: a wanted word value written under another key is the evidence that counts
                wanted = word_values(need)
                spots = [f"{k}={str(v)[:60]}" for k, v in block.items() if not isinstance(v, dict)
                         and any(re.search(r"\b" + re.escape(w) + r"(e?s)?\b", str(v).lower()) for w in wanted)]
                if spots:
                    parts.append((f"(item {number}) " if many else "") + f"{prop} -> no key names it; the wanted value is"
                                 " written under " + ", ".join(spots[:2]) + ", which counts")
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return ("[evidence - attribute keys whose name contains a requirement's property; a key about another property"
                " (another part or aspect) settles nothing, and a key named differently may still carry the same property] " + " | ".join(lines))

    # ----- the cart record -----
    def cart_ids(self) -> list[str]:
        if self.ep.fresh_cart():
            text = to_json(self.ep.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.ep.cart_lines)

    def cart_text(self) -> str:
        if self.ep.fresh_cart():
            return to_json(cut(self.ep.cart_view, 300))[:3000]
        return "Kyo's tally from the cart calls (listing: units): " + to_json(self.ep.cart_lines)

    def cart_sum(self) -> str:
        """The cart's lines at the prices of their newest listing records, with the sum: empty when a price is unknown."""
        lines = {i: q for i, q in self.ep.cart_lines.items() if isinstance(q, int) and q > 0}
        prices = {i: self.price_of(i) for i in lines}
        if not lines or any(p is None for p in prices.values()):
            return ""
        unit = f" {self.ep.brief.get('budget_unit') or ''}".rstrip()
        total = sum(prices[i] * q for i, q in lines.items())
        # a cart view newer than the last cart change gives the store's own total; the sum stands in when it is stale
        source = "the cart view above shows its own total, which counts first" if self.ep.fresh_cart() else "the cart has not been viewed since it last changed"
        return ("\nKyo's sum at the newest listing records' prices (" + source + "): "
                + "; ".join(f"{i} x{q} at {prices[i]:g}" for i, q in lines.items()) + f"; total {total:g}{unit}")

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity."""
        props = self.ep.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in kinds_of(s) and s.get("minimum") == 0 for s in props.values())

    def cart_count(self) -> str | None:
        """The cart tool's whole-number argument beside the listing it names (a cart taking a list of lines has none):
        a count of units, whatever its name says."""
        props = self.ep.schemas.get(self.ep.roles.get("cart") or "", {}).get("properties") or {}
        if any(isinstance(s, dict) and "array" in kinds_of(s) for s in props.values()):
            return None
        return next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds_of(s)), None)

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when the
        schema does not say how."""
        name = self.ep.roles.get("cart") or ""
        schema = self.ep.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            inner = spec.get("items") if isinstance(spec, dict) and "array" in kinds_of(spec) else None
            if isinstance(inner, dict) and isinstance(inner.get("properties"), dict):
                fields = inner["properties"]
                id_key = next((k for k, s in fields.items() if isinstance(s, dict) and "string" in kinds_of(s) and "enum" not in s), None)
                count_key = next((k for k, s in fields.items() if isinstance(s, dict) and "integer" in kinds_of(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds_of(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds_of(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        sets = self.cart_sets(name)
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.ep.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and sets:
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        current = self.cart_ids()
        for ident, want in lines.items():
            have = self.ep.cart_lines.get(ident, 0) if ident in current else 0
            if have == want:
                continue
            if have and not sets:
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update the cart tally from a cart call's arguments."""
        ep = self.ep
        ep.cart_touched = True
        ep.cart_at = ep.step
        spec = ep.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            removal = removal or (isinstance(options, list) and len(options) >= 2 and value in options[1:])
        loose = next((as_int(v) for v in args.values() if as_int(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in self.rows:
                    lines.append((value, loose))
                continue
            if not isinstance(value, list):
                continue
            entry = (spec.get(key) or {}).get("items") or {}
            if not value and isinstance(entry, dict) and (entry.get("type") == "object" or "properties" in entry):
                whole = True
            for line in value:
                if isinstance(line, dict):
                    whole = True
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    count = next((as_int(v) for v in line.values() if as_int(v) is not None and not isinstance(v, str)), 1)
                    if ident:
                        lines.append((ident, count))
                elif isinstance(line, str) and line in self.rows:
                    lines.append((line, loose))
        if whole:
            # a list of line records describes the whole cart
            ep.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = ep.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have += count if count is not None else 1
            if have > 0:
                ep.cart_lines[ident] = have
            else:
                ep.cart_lines.pop(ident, None)


class Ledger:
    """The task as Kyo understands it: the reading of the request, the ledger text, and every change a note makes."""

    def __init__(self, ep: Episode, shelf: Shelf):
        self.ep = ep
        self.shelf = shelf
        self.heard: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False

    # ----- reading the request -----
    def read_request(self) -> None:
        ep = self.ep
        tools = "\n".join(f"* {t['function']['name']} -- {t['function'].get('description', '')} -- parameters "
                          + to_json(t["function"].get("parameters") or {}) for t in ep.tools)
        prompt = BRIEF + "\n\n## TOOLS\n" + tools + "\n\n## REQUEST\n" + ep.request
        if ep.view.get("opening") is not None:
            prompt += "\n\n## OPENING NOTE FROM THE STORE\n" + to_json(cut(ep.view.get("opening"), 1500))
        # the light model reads the request (with room to think: a reply cut at its limit once sent a tenth of the
        # briefs to the heavy model); the heavy one reads it only when that reading names no item or no checkout tool
        brief = ep.ask_json(prompt, [LIGHT], 4000) or {}
        roles = brief.get("roles")
        if not ep.no_credit and (not as_list(brief.get("items")) or not isinstance(roles, dict) or not roles.get("buy")):
            brief = ep.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        ep.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        ep.roles = {job: tool for job, tool in roles.items() if isinstance(tool, str) and tool in ep.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = ep.roles.get("buy"), ep.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            fixed = ep.ask_json(MODES.replace("<<TOOL>>", buy) + to_json(ep.schemas.get(buy)), [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in as_list(brief.get("items")):
            if isinstance(entry, dict):
                quantity = as_int(entry.get("quantity"))
                ep.items.append(Want(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""),
                                     quantity if quantity and quantity > 0 else 1,
                                     [keep_excluded(str(m), ep.request) for m in as_list(entry.get("musts")) if m]))
        ep.items = ep.items or [Want("the requested product", "", 1, [])]
        for lookup in as_list(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in ep.items):
                # a lookup copied loosely: the must on the same property stands for it
                prop = plain(text.split(":")[0])
                text = next((m for item in ep.items for m in item.musts if plain(m.split(":")[0]) == prop), text)
            number = next((n for n, item in enumerate(ep.items, 1) if text in item.musts), 1)
            if text and (number, text) not in ep.refs:
                ep.refs.append((number, text))
        self.unmix()
        budget = as_budget(brief.get("budget"))
        if budget:
            ep.budget = budget
        limit = as_int(brief.get("messages"))
        if limit is not None and limit >= 0:
            ep.msg_limit = limit
        elif ep.roles.get("ask"):
            ep.msg_limit = 3
        steps = as_int(brief.get("steps"))
        if steps is not None and 10 <= steps < ep.max_steps:
            ep.max_steps = steps
        for entry in as_list(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                ep.process.append({"step": str(entry["step"]).strip(), "when": when if when in STAGES else "order",
                                   "messages": max(0, as_int(entry.get("messages")) or 0)})
        if ep.roles.get("ask"):
            # messages kept back for the steps at the order and at the ending: the larger need of the two. A request that
            # takes the closest listing when nothing fits ends with a checkout, so its ending steps keep none back
            stages = ("order",) if brief.get("closest_if_none") is True else ("order", "end")
            talk = [p["messages"] for p in ep.process if p["when"] in stages and p["messages"]]
            ep.reserve = min(max(talk), ep.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (misread as musts of the product) join that lookup."""
        ep = self.ep
        for number, item in enumerate(ep.items, 1):
            pointers = [need for n, need in ep.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\n## REQUEST\n" + ep.request + f"\n\n## PRODUCT\n{item.name}"
                      + "\n\n## REQUIREMENTS POINTING AT ANOTHER LISTING\n" + "\n".join(pointers)
                      + "\n\n## OTHER REQUIREMENTS\n" + "\n".join(rest))
            got = ep.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
            named = [str(x).strip() for x in as_list(got.get("describe_other"))]
            moved = [m for m in rest if m in named]
            if not moved:
                continue
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts if m not in moved]
            ep.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in ep.refs]
            ep.trace.append({"kind": "unmix", "item": number, "moved": moved})

    # ----- what the model is shown -----
    def text(self, stage: str | None = None) -> str:
        """The requirements in force; given a stage, only the request's steps due at that stage are listed."""
        ep, brief = self.ep, self.ep.brief
        many = len(ep.items) > 1
        parts = []
        for number, item in enumerate(ep.items, 1):
            silent = self.unstated(number, item)
            needs = [m + (f" [value read on the linked page: {ep.ref_values[m]}]" if m in ep.ref_values else "")
                     + (f" [no opened listing states this ({len(self.shelf.full)} read): it neither passes nor fails a listing]" if m in silent else "")
                     for m in item.live()]
            line = (f"item {number}: " if many else "item: ") + item.name
            if item.shelf:
                line += f' (store category "{item.shelf}")'
            line += f", {item.qty} unit(s); must meet: " + ("; ".join(needs) or "nothing beyond being this kind of product")
            if item.free:
                line += "; the shopper does not mind about: " + ", ".join(item.free)
            parts.append(line)
        if ep.budget is not None:
            scope = "per item" if brief.get("budget_scope") == "each" else "for the whole order"
            parts.append(" ".join(w for w in ("spending limit:", str(ep.budget), str(brief.get("budget_unit") or ""), scope) if w))
        pick = brief.get("pick")
        if pick == "cheapest":
            parts.append("pick rule: the lowest-priced listing that fits")
        elif pick in ("most", "least"):
            parts.append(" ".join(w for w in ("pick rule: the", pick, str(brief.get("pick_measure") or ""), "among the fitting listings") if w))
        else:
            parts.append("pick rule: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            parts.append("only listings in stock")
        parts += [f"nice to have, only to break ties: {wish}" for wish in as_list(brief.get("wishes"))]
        parts += [f"instruction in the request: {rule}" for rule in as_list(brief.get("rules"))]
        if brief.get("closest_if_none") is True:
            parts.append("if nothing fits: check out the listing that comes closest, as the request describes, instead of ending without a purchase")
        elif brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        when = {"choose": "before choosing", "order": "at checkout, done by Kyo when you check out",
                "end": "when ending without a purchase, done by Kyo then"}
        parts += [f"step the request requires ({when[p['when']]}): {p['step']}" for p in ep.process if stage in (None, p["when"])]
        facts = self.shelf.facts_tool()
        if facts and (self.shelf.facts_from.get(facts) is True or ep.roles.get("offer")):
            views = {t for t in (ep.offer_tool(), ep.roles.get("compare")) if t}
            bare = sorted(t for t in views | {ep.roles.get("offer") or ""} if t and self.shelf.facts_from.get(t) is False)
            pages = sorted(t for t in views if t != facts)
            parts.append(f"listing attributes are shown by {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check refers to) is shown by {' or '.join(pages)}" if pages else ""))
        count = self.shelf.cart_count()
        if count:
            # the count's name may suggest a line position or an item number; the schema leaves it a count of units
            parts.append(f"{ep.roles.get('cart')}'s {count} is the number of units of the listing named, never a line"
                         " position or item number: " + ("the line then holds exactly that many, so send the item's full"
                                                         " quantity in one call (0 takes the line out)" if self.shelf.cart_sets(ep.roles.get("cart") or "")
                                                         else "that many units go into the cart"))
        return "\n".join("- " + p for p in parts)

    def unstated(self, number: int, item: Any) -> set[str]:
        """The request's own wordy requirements of an item that no opened listing says anything about, neither the
        property's name among its keys (by the first four letters of each of its words) nor a word of the value anywhere,
        once six or more are opened. A shopper's
        answer, a pointing requirement or one with a figure is never among them."""
        full = list(self.shelf.full.values())
        if len(full) < 6:
            return set()
        pointing = {need for n, need in self.ep.refs if n == number}
        keys = " ".join(" " + plain(k) + " " for record in full for k in walk_keys(record))
        texts = " ".join(plain(to_json(record)) for record in full)
        out = set()
        for need in item.live():
            if need not in item.musts or need in pointing or ":" not in need or re.search(r"\d", need):
                continue
            prop, value = (plain(part) for part in need.split(":", 1))
            named = any(re.search(rf"\b{re.escape(w[:4])}", keys) for w in prop.split() if len(w) >= 4)
            said = any(re.search(rf"\b{re.escape(w)}", texts) for w in value.split() if len(w) >= 4)
            if not named and not said and any(len(w) >= 4 for w in value.split()):
                out.add(need)
        return out

    def status(self) -> str:
        ep = self.ep
        left = ep.left()
        text = f"[Kyo] Steps: {ep.step} used of {ep.max_steps}, {left} remaining."
        if left <= ep.tail() + 1:
            text += (" LAST STEPS: check out now if the cart has a line for every item. End without a purchase only if your search"
                     " showed that some item has no fitting listing; running out of steps does not show that.")
        elif left <= 6 or (left <= len(ep.items) + 5 and ep.buys == 0):
            text += " Only a few steps remain: set the cart to your chosen lines and check out now."
        elif ep.buys == 0 and left <= 12:
            text += " Steps are getting short: settle your picks, fill the cart and check out within the next few steps."
        elif ep.buys == 0 and not ep.cart_touched and ep.step >= ep.max_steps // 2:
            text += " Half the steps are gone and the cart is still empty: cart each item's fitting listing now (the store may yet sell one out)."
        live = {need for _, need in ep.needs_now()}
        lookups = [need for _, need in ep.refs if need in live and need not in ep.ref_values]
        if lookups and ep.buys == 0:
            text += ("\n[look up first: open the listing each of these points at, plus every listing its page refers to; the last page"
                     " gives the value] " + " | ".join(lookups[:3]))
        if ep.leads:
            text += "\n[candidates from the review] " + ", ".join(ep.leads)
        if ep.notes:
            text += "\n[changes so far] " + " | ".join(ep.notes[-4:])
        return text + "\n[ledger]\n" + self.text()

    # ----- changes -----
    def take_heard(self, heard: dict[str, Any], told: list[str], said: str = "") -> None:
        """Put what the shopper's answers state into the ledger; `told` gathers the new requirements for a summary."""
        ep = self.ep
        many = len(ep.items) > 1
        for entry in as_list(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = ep.item_no(entry.get("item"))
            need = str(entry["need"]).strip()
            if entry.get("points") is True and said:
                need = keep_comparing(need, said)
            item = ep.items[number - 1]
            if need not in item.told:
                item.told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in ep.refs:
                ep.refs.append((number, need))
        for entry in as_list(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = ep.items[ep.item_no(entry.get("item")) - 1]
                prop = plain(str(entry["property"]).split(":")[0])
                hit = [m for m in item.live() if plain(m.split(":")[0]) == prop]
                item.dropped.extend(hit)
                told.extend(f"given up: {m}" for m in hit)
        for entry in as_list(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                ep.items[ep.item_no(entry.get("item")) - 1].free.append(str(entry["property"]))
        budget = ep.fresh_budget(heard.get("budget"), said)
        if budget and budget != ep.budget:
            ep.budget = budget
            told.append(f"new budget {budget:g}")

    def loose_lines(self, observation: Any) -> list[str]:
        """Sentences in a result that are not part of a listing record: notes from the shopper or the store, at any
        depth; a short one counts under a note-like key or in a message with a sender."""
        found: list[str] = []
        stack: list[tuple[Any, int, bool, bool]] = [(observation, 0, False, False)] if isinstance(observation, (dict, list)) else []
        while stack:
            node, depth, in_list, short = stack.pop()
            if isinstance(node, str):
                words = len(node.split())
                if words >= (5 if depth <= 1 or in_list else 6) or (short and words >= 2):
                    found.append(node)
            elif isinstance(node, list) and depth <= 4:
                stack += [(inner, depth + 1, True, short) for inner in reversed(node)]
            elif isinstance(node, dict) and depth <= 4 and not self.shelf.ident(node, in_list, depth):
                senders = {k for k in node if re.fullmatch(r"from|role|sender|author|speaker", str(k).lower())}
                # the sender's own name is not a note
                stack += [(inner, depth + 1, False, (bool(senders) and k not in senders)
                           or bool(re.search(r"note|notice|alert|warn|event|news", str(k).lower())))
                          for k, inner in reversed(list(node.items()))]
        return [line for line in dict.fromkeys(found) if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        ep, shelf = self.ep, self.shelf
        self.heard.update(lines)
        prompt = NOTE + "\n\n## ITEMS AND REQUIREMENTS IN FORCE\n" + self.text() + "\n\n## THE NOTE\n" + "\n".join(lines)
        got = ep.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in as_list(got.get("gone")) if isinstance(g, (str, int))]
        shelf.unavailable.update(g for g in gone if g in shelf.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = ep.fresh_budget(got.get("budget"), " ".join(lines))
        if budget and budget != ep.budget:
            ep.budget = budget
            changed.append(f"new budget {budget}")
        drops = [e for e in as_list(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in as_list(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = ep.items[ep.item_no(entry.get("item")) - 1]
            target = str(entry["need"]).strip()
            live = item.live()
            # the exact requirement, else the first one on the same property
            hit = [m for m in live if m == target] or [m for m in live if plain(m.split(":")[0]) == plain(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"no longer required: {target}")
        for entry in adds:
            item = ep.items[ep.item_no(entry.get("item")) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f"now required: {need}")
        if gone:
            changed.append("can no longer be bought: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price moved")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            ep.version += 1
            ep.recent = []
            ep.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def observe(self, name: str, observation: Any) -> str:
        """Take in a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.shelf.absorb(observation)
        lines = self.loose_lines(store_note(self.ep.roles, name, observation))
        meaning = self.take_note(lines) if lines else ""
        if meaning:
            self.ep.queued.append(f"(on {name}) {meaning}")
        return meaning


class Scout:
    """Store reads Kyo makes without a model turn: attributes, the rest of a search, fresh offers."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        self.cheaper_marks: set[str] = set()

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        ep, shelf = self.ep, self.shelf
        tool = shelf.facts_tool()
        target = shelf.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in shelf.rows and i not in shelf.full and not shelf.sold_out(i)]:
            if ep.left() <= keep or ep.finished or shelf.facts_from.get(tool) is False:
                break
            args = ep.fit_args(tool, {target: ident})
            reply = ep.act(tool, args) if args is not None else None
            if reply is None:
                break
            seen = reply.get("observation")
            self.ledger.observe(tool, seen)
            shelf.note_view(tool, seen)
            if isinstance(seen, dict) and shelf.ident(seen):
                shelf.facts_from[tool] = shelf.facts_from.get(tool, False) or bool(shelf.facts_of(seen))
            texts.append(f"{ident}: " + render(seen)[:1500])
        if texts:
            ep.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: unread buyable rows of a cart line's own kind priced below it get their
        attributes read (a few, cheapest first), so the last check can weigh them against the cart's choice."""
        ep, shelf = self.ep, self.shelf
        if ep.brief.get("pick") != "cheapest":
            return []
        mark = f"{ep.cart_mark()}#{ep.version}"
        cat_key = shelf.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = shelf.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = shelf.price_of(line)
            kind = str((shelf.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in shelf.seen_order if i not in cart and i not in shelf.full and shelf.buyable(i)
                    and str((shelf.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind and shelf.price_of(i, _INF) < price]
            pool.sort(key=lambda i: shelf.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], ep.tail() + 4)

    def widen(self) -> None:
        """Open more listings side by side (one step) so the planner's key counts rest on more than two or three listings."""
        ep, shelf = self.ep, self.shelf
        tool = ep.roles.get("compare")
        target = shelf.compare_target()
        # a side-by-side view without attributes shows the planner no keys: read attributes one listing at a time
        if not tool or not target or shelf.compare_bare():
            self.widen_singly()
            return
        if ep.left() <= 12:
            return
        cap = shelf.compare_room()
        groups = shelf.item_groups()
        share = max(1, cap // max(1, len(groups)))
        picks = [i for ids in groups for i in [x for x in ids if x not in shelf.full][:share]][:cap]
        if len(picks) < 2:
            return
        args = ep.fit_args(tool, {target: picks})
        if args is None:
            return
        before = len(shelf.full)
        reply = ep.act(tool, args)
        if reply is None:
            return
        seen = reply.get("observation")
        self.ledger.observe(tool, seen)
        shelf.note_view(tool, seen)
        rows = shelf.records_in(seen)
        if rows:
            shelf.facts_from[tool] = shelf.facts_from.get(tool, False) or any(shelf.facts_of(r) for _, r in rows)
        ep.trace.append({"kind": "widen", "ids": picks, "opened": len(shelf.full) - before})
        if rows and shelf.facts_from.get(tool) is False:
            # this store's side-by-side view carries no attributes: read them singly instead
            self.widen_singly()
            return
        if len(shelf.full) == before and len(picks) > 2 and ep.left() > 12:
            args = ep.fit_args(tool, {target: picks[:2]})
            reply = ep.act(tool, args) if args is not None else None
            if reply is not None:
                self.ledger.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        """Without a side-by-side view that shows attributes: read a few more listings' attributes one by one."""
        shelf = self.shelf
        tool = shelf.facts_tool()
        if not tool or not shelf.single_target(tool):
            return
        lanes = []
        for ids in shelf.item_groups():
            opened = sum(1 for i in ids if i in shelf.full)
            pool = [i for i in ids if i not in shelf.full and shelf.buyable(i)]
            if self.ep.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: shelf.price_of(i, _INF))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        # turns between the item groups, two each at most; 22 steps stay for the rest of the episode
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:5], 22)

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else two singly."""
        ep, shelf = self.ep, self.shelf
        many, joiner = shelf.compare_target(), ep.roles.get("compare")
        tool = shelf.facts_tool()
        target = shelf.single_target(tool)
        if many and joiner and not shelf.compare_bare():
            plan = [(joiner, {many: rows[: shelf.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if ep.left() < len(plan) + 5:
            return False
        before = len(shelf.full)
        for name, args in plan:
            shaped = ep.fit_args(name, args)
            if shaped is None:
                return False
            reply = ep.act(name, shaped)
            if reply is None or ep.finished:
                break
            self.ledger.observe(name, reply.get("observation"))
        ep.trace.append({"kind": "audit_open", "rows": rows, "opened": len(shelf.full) - before})
        return len(shelf.full) > before

    def short_name(self, item: Want) -> str:
        """An item's name for a search box: the store's label, else the name up to its first describing clause."""
        return item.shelf or " ".join(re.split(r"\s+(?:in|with|that|which|for|of|from)\s+", item.name, maxsplit=1)[0].split()[:4])

    def stating(self, number: int, need: str) -> list[str]:
        """Opened, buyable listings of an item whose own attribute for the need's property carries its value."""
        shelf = self.shelf
        prop, value = stems(need.rsplit(":", 1)[0]), need.rsplit(":", 1)[-1].strip()
        found = []
        for ident in reversed(shelf.seen_order):
            facts = shelf.facts_of(shelf.full.get(ident) or {}) or {}
            said = [v for k, v in facts.items() if key_names(prop, stems(k))]
            if said and shelf.buyable(ident) and shelf.item_of(ident) in (number, None) and any(says_phrase(value, v) for v in said):
                found.append(ident)
        return found[:4]

    def chase(self, needs: list[tuple[int, str]]) -> str:
        """Right after the interview: store searches for what the shopper just said, joined to the item's name, then the
        most promising new rows opened, and the listings whose records state it named. Model turns often go on with the
        listings already in view; these searches cost steps and no inference."""
        ep, shelf = self.ep, self.shelf
        if not shelf.text_search:
            return ""
        tool, word, fixed = shelf.text_search
        lines = []
        for number, need in needs[:CHASE_NEEDS]:
            value = need.rsplit(":", 1)[-1].strip()
            plain_value = ":" in need and 0 < len(value.split()) <= 3 and not _NOT_PLAIN & set(re.findall(r"[a-z]+", value.lower()))
            if ep.finished or ep.left() < CHASE_FLOOR or not plain_value or any(need == ref for _, ref in ep.refs):
                continue
            rows: list[str] = []
            for query in (f"{value} {self.short_name(ep.items[number - 1])}", value):
                args = ep.fit_args(tool, {word: query[:80], **fixed})
                reply = ep.act(tool, args) if args is not None else None
                if reply is None or ep.finished:
                    break
                seen = reply.get("observation")
                self.ledger.observe(tool, seen)
                rows = [i for i, _ in shelf.records_in(seen)]
                if rows or ep.left() < CHASE_FLOOR:
                    break
            fresh = [i for i in rows if i not in shelf.full and shelf.buyable(i) and shelf.item_of(i) in (number, None)]
            # a row that says the value in its own words goes first
            fresh.sort(key=lambda i: not says_phrase(value, to_json(shelf.rows.get(i))))
            if fresh:
                self.open_rows(fresh)
            named = self.stating(number, need)
            ep.trace.append({"kind": "chase", "need": need, "rows": len(rows), "stated": named})
            lines.append(f"{need}: " + (f"records stating it: {', '.join(named)}" if named else "no opened record states it yet"))
        return ("Kyo searched for these itself. " + " | ".join(lines)) if lines else ""

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more than it showed, under a cheapest, most or least pick: Kyo fetches a short rest
        itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        ep, shelf = self.ep, self.shelf
        self.paged_obs = []
        pick = ep.brief.get("pick")
        plan = shelf.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        if isinstance(given, int) and not isinstance(given, bool):
            later = given not in (None, 0, 1)
        else:
            later = bool(str(given or "").strip())
        # the last page already, or the agent pages by itself
        if value is None or later or count <= shown:
            return ""
        if pick == "cheapest":
            prices = [shelf.price_of(i) for i, _ in shelf.records_in(observation)]
            if len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:])):
                return ""
        pages = -(-(count - shown) // shown)
        # a keyword search led by a wanted value ranks the listings carrying it first: under a cheapest pick its next
        # two pages are read even when the whole rest is too long, since a cheaper fit often sits just past the first
        fetch = pages if pages <= 2 else (2 if pick == "cheapest" and self.value_query(name, args) else 0)
        if not fetch or self.pages_read + fetch > AUTO_PAGES or ep.left() - ep.tail() - fetch < 12:
            return (f"\n[Kyo] This search reports {count} matches but showed only {shown}, not sorted by price. The pick rule weighs"
                    f" every fitting listing: narrow the search with more specific words (the requirement's value) or fetch the next"
                    f" page with {param}={to_json(value)}.")
        texts = []
        for _ in range(fetch):
            call = ep.fit_args(name, {**args, param: value})
            if call is None:
                break
            reply = ep.act(name, call)
            if reply is None or ep.finished:
                break
            self.pages_read += 1
            seen = reply.get("observation")
            self.ledger.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = render(seen)
            shelf.read_cache[name + to_json(call)] = (ep.version, ep.step, shown_text)
            texts.append(f"{param}={to_json(value)}: {shown_text}")
            after = shelf.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        ep.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return ""
        if fetch < pages and value is not None:
            return (f"\n[Kyo fetched the next {len(texts)} pages of this search itself, one step per page; it ranks listings carrying"
                    f" the query's words first. Further pages: {param}={to_json(value)}]\n" + "\n".join(texts))
        return "\n[Kyo fetched the remaining pages of this search itself, one step per page]\n" + "\n".join(texts)

    def value_query(self, name: str, args: dict[str, Any]) -> bool:
        """A keyword search (not a filtered listing) whose text carries a word of a wanted value."""
        if name != self.ep.roles.get("search"):
            return False
        text = " ".join(str(v) for v in args.values() if isinstance(v, str)).lower()
        wanted = [w for _, need in self.ep.needs_now() for w in word_values(need)]
        return any(re.search(r"\b" + re.escape(w) + r"(e?s)?\b", text) for w in wanted)

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages fetched with it) by price."""
        shelf = self.shelf
        if self.ep.brief.get("pick") != "cheapest" or not shelf.price_key:
            return ""
        found = {i for seen in observations for i, _ in shelf.records_in(seen)}
        rows = [(shelf.price_of(i), i) for i in found if shelf.buyable(i) and shelf.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows seen here, lowest price first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def reread_offers(self, since: int | None = None) -> str:
        """Re-read each cart line with the offer view (those not read at or after `since`; by default the last two
        steps or the last refusal) just before checkout or after a note: stock and price change unannounced."""
        ep, shelf = self.ep, self.shelf
        tool = ep.offer_tool()
        target = shelf.single_target(tool)
        if not tool or not target:
            return ""
        since = max(ep.step - 2, ep.refused_at) if since is None else since
        stale = [i for i in shelf.cart_ids() if shelf.offer_seen.get(i, -99) < since and i not in shelf.gone]
        if ep.left() < len(stale) + 3:
            # short of steps: a re-read can go, but a page never read is part of buying that line
            stale = [i for i in stale if i not in shelf.offer_seen]
        if not stale or ep.left() < len(stale) + 1:
            return ""
        lines = []
        for ident in stale:
            args = ep.fit_args(tool, {target: ident})
            if args is None:
                break
            before = dict(shelf.rows.get(ident) or {})
            reply = ep.act(tool, args)
            if reply is None or ep.finished:
                break
            observation = reply.get("observation")
            shelf.offer_seen[ident] = ep.step
            self.ledger.observe(tool, observation)
            # compare stored records, so a re-read wrapped together with a note still shows what moved
            after = shelf.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} now shows different {', '.join(moved[:4])}: {to_json(cut(observation, 300))[:900]}")
        ep.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            ep.version += 1
        return "\n".join(lines)


class Talk:
    """Everything said to the shopper: the interview about unwritten preferences, and the steps the request asks
    for right before the order or the ending (the process gate)."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger, scout: Scout):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.scout = scout
        self.asked = False
        self.done_log: list[str] = []
        self.done_steps: list[int] = []

    def ready(self, forced: bool = False) -> bool:
        """Whether to interview now: some messages are free and steps remain; unforced, once enough listings were read."""
        ep = self.ep
        if self.asked or not ep.roles.get("ask") or ep.msgs_free() <= 0 or ep.left() <= 8:
            return False
        return forced or len(self.shelf.full) >= 3 * len(ep.items) + 1 or ep.step >= max(8, ep.max_steps // 4)

    def interview(self) -> str:
        ep = self.ep
        self.asked = True
        count = ep.msgs_free()
        if count <= 0:
            return ""
        self.scout.widen()
        if self.shelf.item_unseen():
            # the first listings opened were all of other categories (often the pages a request points to): one more
            # side-by-side view so the planner sees the keys of the products being bought
            self.scout.widen()
        many = len(ep.items) > 1
        # on several products the plan carries spares, so a product whose need came out can give its turn to another
        ordered = self.fill_plan(self.plan(count + (len(ep.items) if many else 0)), count)
        told: list[str] = []
        # Kyo's questions name one property of one product each, so a reply asking for one question at a time cannot be
        # split further and asking the same property again reveals nothing: every message goes to a fresh pick
        batch = self.send_aimed(ordered, count)
        if not batch:
            return ""
        sent = list(batch)
        known = ep.needs_now()
        self.ledger.take_heard(self.hear(batch), told, " ".join(a for _, _, a in batch))
        if told:
            ep.version += 1
        summary = ("Kyo's interviewer asked the shopper the following (one step per question):\n"
                   + "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
                   + "\nFirm requirements these answers add: " + ("; ".join(told) if told else "none")
                   + ". Apply them to every pick, searching again with the new values where needed.")
        pointed = [need for _, need in ep.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Locate that listing, open its"
                        " page in full and open every listing that page refers to; the last page gives the value, and the order"
                        " counts only once all of them are opened.")
        chased = self.scout.chase([pair for pair in ep.needs_now() if pair not in known])
        if chased:
            summary += "\n" + chased
        if ep.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(ep.queued)
            ep.queued = []
        return summary

    def plan(self, count: int) -> list[tuple[int, str]]:
        """The planner's picks as (item number, property label): one property each, no repeat within an item."""
        ep = self.ep
        settled = "\n".join(f"item {n}: {it.name} | already settled by the request: " + ("; ".join(it.musts) or "nothing")
                            for n, it in enumerate(ep.items, 1))
        keys, last = [], None
        for index, (label, name, carried, values) in enumerate(self.shelf.key_table(), 1):
            if label != last:
                keys.append(f"category {label!r}:")
                last = label
            keys.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        prompt = PLAN.replace("<<COUNT>>", str(count)) + "\n\n## REQUEST\n" + ep.request + "\n\n## ITEMS\n" + settled
        first = [str(x) for x in as_list(ep.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\n## ASK FIRST (the request wants these checked with the shopper)\n" + "\n".join(first)
        prompt += "\n\n## ATTRIBUTE KEYS SEEN\n" + ("\n".join(keys) or "(no full listing records yet)")
        got = ep.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in as_list(got.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = ep.item_no(pick.get("item"))
            # one property per question: a label naming two keeps the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(plain(label) != plain(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        ep.trace.append({"kind": "plan", "ask": cut(got.get("ask"), 120)})
        return ordered

    def fill_plan(self, ordered: list[tuple[int, str]], want: int) -> list[tuple[int, str]]:
        """A plan shorter than the messages free, topped up from the keys listings differ on: an unsent message finds
        nothing, and a failed planner call must not cost the whole interview."""
        ep = self.ep
        settled = [stems(need.rsplit(":", 1)[0]) for _, need in ep.needs_now()]
        for label, name, _, values in self.shelf.key_table():
            if len(ordered) >= want:
                break
            number = next((n for n, it in enumerate(ep.items, 1) if stems(f"{it.name} {it.shelf or ''}") & stems(label)), 1)
            if len(values) < 2 or len(name.split()) > 3 or any(key_names(p, stems(name)) for p in settled):
                continue
            if all(plain(name) != plain(other) for n, other in ordered if n == number):
                ordered.append((number, name))
        return ordered

    def send_aimed(self, picks: list[tuple[int, str]], count: int) -> list[tuple[int, str, str]]:
        """The planned questions one message at a time; the next goes to a product no answer has stated a need for yet.
        A product with one may hide a second, so it waits for its turn rather than dropping out."""
        sent: list[tuple[int, str, str]] = []
        queue = list(picks)
        stated: set[int] = set()
        while queue and len(sent) < count:
            number, label = next((p for p in queue if p[0] not in stated), queue[0])
            queue.remove((number, label))
            got = self.send([(number, label, self.question(number, label))])
            if not got:
                break
            sent += got
            if answers(label, got[0][2]):
                stated.add(number)
        return sent

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.ep.items[number - 1].name}?"

    def message_args(self, text: str) -> dict[str, Any] | None:
        ep = self.ep
        name = ep.roles.get("ask")
        schema = ep.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in kinds_of(v) and "enum" not in v]
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
        return ep.fit_args(name, args)

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        ep = self.ep
        tool = ep.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if ep.finished or ep.left() <= floor or ep.msg_used >= ep.msg_limit:
                break
            args = self.message_args(text)
            reply = ep.act(tool, args) if args else None
            if reply is None:
                break
            ep.msg_used += 1
            answer = reply.get("observation")
            words = words_of(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {ep.step}: asked the shopper {text!r}, reply {words[:300]!r} (cart at that point: {ep.cart_mark()})")
            self.done_steps.append(ep.step)
            ep.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if failed(answer):
                break
        return sent

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        ep = self.ep
        limit = f"\n\n## LIMIT IN FORCE\n{ep.budget:g} {ep.brief.get('budget_unit') or ''}".rstrip() if ep.budget else ""
        prompt = (HEAR + limit + "\n\n## ITEMS\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.ep.items, 1))
                  + "\n\n## QUESTIONS AND ANSWERS\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.ep.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the request's steps for that
        moment, done by store calls the model chooses (a message, a page, any tool) until it finds none missing for
        the cart as it is now. Returns whether to go on, and what to tell the agent."""
        ep = self.ep
        steps = [p for p in ep.process if p["when"] == stage]
        if not steps or ep.finished or not ep.gate_due(stage):
            return True, ""
        ending = {ep.roles.get("buy"), ep.roles.get("refuse")} - {None}
        tools = "\n".join(f"* {fn['name']} -- {fn.get('description', '')} -- parameters {to_json(fn.get('parameters') or {})}"
                          for fn in (t["function"] for t in ep.tools) if fn["name"] not in ending)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            if said and ep.late(CLOCK_GATE):
                # past the clock's gate a step already carried out stands: another model round could outlast the problem
                break
            prompt = (GATE.replace("<<STAGE>>", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\n## REQUEST\n" + ep.request + "\n\n## STEPS DUE NOW\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\n## LEDGER\n" + self.ledger.text(stage) + "\n\n## CART NOW\n" + self.shelf.cart_text()
                      + self.shelf.cart_sum()
                      + "\n\n## RECORD (messages and step calls so far)\n"
                      + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\n## STORE NOTES SEEN\n" + "\n".join(ep.notes[-4:]) if ep.notes else "")
                      + f"\n\n## LEFT: {max(0, ep.msg_limit - ep.msg_used)} messages, {ep.left()} steps"
                      + f" (the {'order' if stage == 'order' else 'ending'} itself takes one step)"
                      + "\n\n## TOOLS\n" + tools)
            # the light model decides; after a nudge, the heavy one
            got = ep.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                # only what a reply to the gate's own call asks for (earlier answers were the interviewer's to read)
                self.ledger.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            ep.trace.append({"kind": "gate", "stage": stage, "call": cut(call, 200), "steps": cut(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                ep.version += 1
                done = ("Kyo first carried out the steps the request asks for:\n" + "\n".join(said) + "\n") if said else ""
                return False, (done + "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else []))
                               + ". Adjust the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a message
            # concerns the final cart, so its line must come after the cart last changed
            shown = [e for e in as_list(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            proof = {i for i in range(1, len(self.done_log) + 1)
                     if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > ep.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((as_int(n) or 0) in proof for n in as_list(e.get("lines")))]
            if not call and unproven and not said and not nudged and ep.roles.get("ask") and ep.msg_used < ep.msg_limit and ep.left() >= 2:
                nudged = True
                ep.trace.append({"kind": "gate_unproven", "steps": cut(unproven, 120)})
                self.done_log.append(f"(Kyo: no RECORD line since the cart last changed at step {ep.cart_at} shows "
                                     + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, while {ep.msg_limit - ep.msg_used} message(s) and {ep.left()} steps remain: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in ep.schemas or name in ending:
                if not call:
                    ep.gate_ok.add((stage, f"{ep.cart_mark()}#{ep.version}"))
                break
            if ep.left() < 2 or (name == ep.roles.get("ask") and ep.msg_used >= ep.msg_limit):
                break
            args = ep.fit_args(name, call.get("args") if isinstance(call.get("args"), dict) else {})
            if args is None:
                break
            if name == ep.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                ep.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = ep.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.shelf.absorb(seen)
            self.done_log.append(f"step {ep.step}: {name} {to_json(args)} -> {render(seen)[:300]} (cart at that point: {ep.cart_mark()})")
            self.done_steps.append(ep.step)
            said.append(f"{name} {to_json(args)} -> {render(seen)[:400]}")
        return True, ("Kyo first carried out the steps the request asks for:\n" + "\n".join(said)) if said else ""


class Checks:
    """Second opinions before a finish: the cart check, the page chains behind pointing requirements, the review of
    a decline, and replacements for a sold-out cart line."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger, scout: Scout):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.scout = scout
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.seller_searched: set[str] = set()
        self.audited = False
        self.lead_items: dict[str, list[str]] = {}
        self.confirmed: str | None = None

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.ep.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = REVIEW_SPEND) -> str:
        """The heavy model's verdict on the cart (or on planned lines), once per lines and version; text when it fails."""
        ep, shelf = self.ep, self.shelf
        planned = ids is not None
        ids = ids if planned else shelf.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or ep.cost > limit or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\n## REQUEST\n" + ep.request + "\n\n## REQUIREMENTS IN FORCE\n" + self.ledger.text()
        if planned:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(ep.items, 1))
            prompt += "\n\n## CART (planned: judge these listings as though carted at the requested quantities - " + quantity + ")\n" + ", ".join(ids)
        else:
            prompt += "\n\n## CART\n" + self.shelf.cart_text()
        prompt += "\n\n## RECORDS OF THE CART'S LISTINGS\n" + "\n".join(
            f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if shelf.sold_out(i) else "") + shelf.record_of(i)
            + "".join(f" [Kyo: {p} - no entry restates the wanted span, which other listings write as the store's label]"
                      for p in shelf.span_flags(i)) for i in ids)
        if ep.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\n## OTHER LISTINGS SEEN (newest first)\n" + shelf.others(ids)
        got = ep.ask_json(prompt, [HEAVY, LIGHT], 1500)
        ep.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        ep.leads = [i for i in ep.leads if i not in ids]
        problems = [str(p) for p in as_list(got.get("problems"))][:5]
        better = [str(b) for b in as_list(got.get("better")) if str(b) in shelf.rows and not shelf.sold_out(str(b))][:4]
        text = "An independent review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings it rates better: " + ", ".join(better)
        return text + ". Correct the cart (or re-read the records to show the review is mistaken), then check out."

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: every requirement whose value sits on another listing has each page of its chain opened.
        Kyo opens the links it can name; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        ep, shelf = self.ep, self.shelf
        live = {need for _, need in ep.needs_now()}
        pending = [need for _, need in ep.refs if need in live and need not in ep.ref_values]
        # every link is a page: the page view (the offer view when it is apart from the attributes view)
        tool = ep.offer_tool()
        target = shelf.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and ep.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(shelf.seen_order):
                # an opened page is the page view's record (with the attributes when those were read too)
                record = (shelf.full.get(ident) or shelf.rows.get(ident)) if ident in ep.opened else None
                records.append("OPENED " + to_json(cut(record, 300))[:1400] if record else "ROW " + to_json(cut(shelf.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = (TRAIL + "\n\n## REQUIREMENTS WITH A VALUE ON ANOTHER LISTING\n" + "\n".join(pending)
                      + "\n\n## RECORDS (newest first)\n" + "\n".join(records))
            got = ep.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            ep.trace.append({"kind": "trail", "pending": pending, "got": cut(got, 150)})
            reach: list[str] = []
            lost = []
            sellers: list[str] = []
            for chain in as_list(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [link for link in as_list(chain.get("links")) if isinstance(link, dict)]
                ids = [str(link.get("id") or "").strip() for link in links]
                value = str(chain.get("value") or "").strip()
                if chain.get("done") is True and links and all(i in ep.opened for i in ids) and value:
                    partner = str(chain.get("partner") or "").strip()
                    if chain.get("relation") != "pair" or plain(partner) == plain(value):
                        ep.ref_values[need] = value
                    elif partner:
                        ep.ref_values[need] = f"{value}, which the store's pairing rule matches with {partner}: the wanted value is {partner}"
                    else:
                        # a pairing is never answered by the page's own value: the store's rule is still to be found
                        lost.append(f"{need}: the store's rule saying which value goes together with {value} (a note on that page"
                                    " or on other listings of its category)")
                        continue
                    ep.version += 1
                    continue
                closed = [i for i in ids if i in shelf.rows and i not in ep.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(link.get("about") or "") for link, i in zip(links, ids) if i not in shelf.rows), "") or "the linked listing"
                    lost.append(f"{need}: find {about}")
                    seller = next((str(link.get("seller") or "").strip() for link, i in zip(links, ids) if i not in shelf.rows), "")
                    if seller and seller.lower() not in self.seller_searched and seller not in sellers:
                        sellers.append(seller)
            pending = [p for p in pending if p not in ep.ref_values]
            if not reach and sellers and self.seller_rows(sellers[0]):
                # the seller's own listings are now among the records: the next round can name the link among them
                continue
            if not reach or not tool or not target or ep.left() < len(reach) + 4 or ep.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                args = ep.fit_args(tool, {target: ident})
                if args is None:
                    break
                reply = ep.act(tool, args)
                if reply is None or ep.finished:
                    break
                self.ledger.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + render(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages these requirements point at]\n" + "\n".join(seen) + "\n"
        if ep.ref_values:
            text += "[values read off the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in ep.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and ep.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("An order counts only once every page a requirement points at has been opened in full (a search row does not"
                     " count), including each further listing a page refers to. Still to find and open: " + " | ".join(lost[:3])
                     + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def seller_rows(self, seller: str) -> bool:
        """A link nobody has found yet: search its seller's name alone (the way the store's own search was used before,
        without filters) so all of that seller's listings show up as rows. Once per seller; whether new rows came."""
        ep, shelf = self.ep, self.shelf
        self.seller_searched.add(seller.lower())
        if not shelf.text_search or ep.left() < 6 or ep.finished:
            return False
        tool, word, fixed = shelf.text_search
        args = ep.fit_args(tool, {**fixed, word: seller})
        before = len(shelf.rows)
        reply = ep.act(tool, args) if args is not None else None
        if reply is None:
            return False
        self.ledger.observe(tool, reply.get("observation"))
        ep.trace.append({"kind": "seller_rows", "seller": seller, "new": len(shelf.rows) - before})
        return len(shelf.rows) > before

    def swap_hint(self) -> str:
        """A cart line whose newest record says sold out: name replacements among listings already read, once per line."""
        ep, shelf = self.ep, self.shelf
        lost = [i for i in shelf.cart_ids() if i not in shelf.gone and shelf.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            shelf.gone[ident] = ep.step
        ep.leads = [i for i in ep.leads if i not in lost]
        ep.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out: {', '.join(lost)} (the store shows it can no longer be bought). It will not return, so do not open it again."
        if picks and ep.left() <= 6:
            return text + (f" Steps are nearly gone: this turn, add {picks[0]} at the requested quantity and remove the sold-out line"
                           f" (two calls), then check out. Other already-read listings that appear to fit: {', '.join(picks[1:]) or 'none'}.")
        if picks and ep.brief.get("pick") == "cheapest":
            return text + (f" Already-read listings that appear to fit that item instead, lowest price first: {', '.join(picks)}. The pick"
                           f" rule is the cheapest, so unless you have already seen the whole fitting set, run one search below"
                           f" {picks[0]}'s price (with the search's maximum-price filter, if it has one) before swapping the sold-out"
                           " line for the cheapest fitting listing.")
        live = {need for _, need in ep.needs_now()}
        if picks and any(need in live and need not in ep.ref_values for _, need in ep.refs):
            # a value still to be read off another listing's page cannot be checked by the review: the agent confirms it
            return text + (f" Already-read listings that may fit that item instead: {', '.join(picks)}. Take the sold-out line out"
                           " of the cart and add the first of them whose page meets the value each requirement takes from another"
                           " listing's page (as you traced it), then check out.")
        if picks:
            return text + (f" Already-read listings that appear to fit that item instead: {', '.join(picks)}. This turn, take the"
                           f" sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out.")
        return text + " Take that line out of the cart and add the best other listing that fits; search for one if none you have read fits."

    def replacements(self, lost: list[str]) -> list[str]:
        ep, shelf = self.ep, self.shelf
        if ep.cost > SPEND_SOFT:
            return []
        records = []
        for ident in shelf.kin_first():
            if ident in lost or ident in shelf.gone or shelf.sold_out(ident):
                continue
            records.append(to_json(cut(shelf.full.get(ident) or shelf.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = (SWAP + "\n\n## REQUEST\n" + ep.request + "\n\n## REQUIREMENTS IN FORCE\n" + self.ledger.text()
                  + "\n\n## SOLD-OUT CART LINE\n" + "\n".join(f"{i}: {shelf.record_of(i)}" for i in lost)
                  + "\n\n## CART NOW\n" + self.shelf.cart_text() + "\n\n## OTHER LISTING RECORDS SEEN (requested kinds and opened pages first, newest first within each)\n" + "\n".join(records))
        got = ep.ask_json(prompt, [HEAVY, LIGHT], 2500)
        cart = shelf.cart_ids()
        fits: list[tuple[float, str]] = []
        for entry in as_list((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in shelf.rows and ident not in lost and ident not in shelf.gone and ident not in cart and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else _INF, ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = sorted(fits, key=lambda fit: fit[0]) if ep.brief.get("pick") == "cheapest" else fits
        ep.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return [i for _, i in ranked][:4]

    def lead_plan(self) -> list[str]:
        """One reviewed lead per requested item (its first still buyable), when the review named one for every item."""
        picks = []
        for number in range(1, len(self.ep.items) + 1):
            pick = next((i for i in self.lead_items.get(str(number), []) if not self.shelf.sold_out(i)), None)
            if not pick or pick in picks:
                return []
            picks.append(pick)
        return picks

    def free_clash(self) -> str:
        """The cart against the records with no model: a key naming a plain requirement's property, under every spelling,
        whose values all say something else. Numbers, limits and exclusions stay with the reviews."""
        ep, shelf = self.ep, self.shelf
        found = []
        for ident in shelf.cart_ids():
            facts = shelf.facts_of(shelf.full.get(ident) or {}) or {}
            number = shelf.item_of(ident) or (1 if len(ep.items) == 1 else None)
            for need in [need for n, need in ep.needs_now() if n == number]:
                wanted = word_values(need)
                prop = stems(need.rsplit(":", 1)[0])
                said = [v for k, v in facts.items() if key_names(prop, stems(k))]
                if wanted and said and not any(says_phrase(w, v) for w in wanted for v in said):
                    found.append(f"{ident} needs {need!r} but its record says {str(said[0])[:60]!r}")
        return "; ".join(found[:4])

    def audit(self) -> str:
        """Once, before a decline: the heavy model looks for fitting listings among the records seen (opening a few
        promising unread rows first when it finds none). Returns the leads as text."""
        if self.audited or self.ep.cost > REVIEW_SPEND:
            return ""
        self.audited = True
        got, named = self.audit_call()
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            shelf = self.shelf
            rows = [str(i) for i in as_list((got or {}).get("open")) if str(i) in shelf.rows and str(i) not in self.ep.opened and not shelf.sold_out(str(i))]
            if rows and self.scout.open_rows(rows):
                got, named = self.audit_call()
        if not named:
            return ""
        self.lead_items = named
        self.ep.leads = [i for ids in named.values() for i in ids]
        return ("An independent review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
                + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase"
                " only if every one of them fails.")

    def audit_call(self) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
        """One review of the records seen: the reply, and per item the buyable listings it names as fits."""
        ep, shelf = self.ep, self.shelf
        records = []
        for ident in shelf.kin_first():
            if shelf.sold_out(ident):
                continue
            record = shelf.full.get(ident) if ident in shelf.full else None
            records.append(("OPENED " + to_json(cut(record, 200))[:1200]) if record else ("ROW " + to_json(cut(shelf.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None, {}
        prompt = (AUDIT + "\n\n## REQUEST\n" + ep.request + "\n\n## REQUIREMENTS IN FORCE\n" + self.ledger.text()
                  + "\n\n## LISTING RECORDS SEEN (requested kinds and opened pages first, newest first within each)\n" + "\n".join(records))
        got = ep.ask_json(prompt, [HEAVY, LIGHT], 3000)
        ep.trace.append({"kind": "audit", "got": got})
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return got, {}
        named = {str(k): [str(i) for i in as_list(v) if str(i) in shelf.rows and not shelf.sold_out(str(i))][:4] for k, v in fits.items()}
        return got, {k: v for k, v in named.items() if v}


class Closer:
    """Finishes Kyo makes itself: a checkout of a confirmed lead, and the forced finish when the loop stops."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger, scout: Scout, talk: Talk, checks: Checks):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.scout = scout
        self.talk = talk
        self.checks = checks
        self.completed = False
        self.closest_done = False

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The decline's free-text arguments: the fixed decline unless the request wants alternatives and the text
        names no listing."""
        schema = self.ep.schemas.get(name) or {}
        required = schema.get("required") or []
        out = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            if not isinstance(spec, dict) or "string" not in kinds_of(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in required:
                continue
            text = str(value or "")
            if not text or any(ident in text for ident in self.shelf.rows) or self.ep.brief.get("quiet_decline") is not False:
                out[key] = DECLINE[: spec.get("maxLength", 200)]
        return out

    def complete_cart(self) -> str:
        """The agent declines although the independent check confirmed a reviewed listing: Kyo carts the reviewed lines
        (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        ep, shelf, checks = self.ep, self.shelf, self.checks
        cart, buy = ep.roles.get("cart"), ep.roles.get("buy")
        confirmed = checks.confirmed
        if self.completed or not confirmed or shelf.sold_out(confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = shelf.category_key()

        def kind(ident: str) -> str:
            return str((shelf.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in checks.lead_items.items():
            index = as_int(number)
            pick = next((i for i in ids if not shelf.sold_out(i)), None)
            if pick and index and 1 <= index <= len(ep.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = ep.items[index - 1].qty
        if confirmed not in plan:
            return ""
        now = self.shelf.cart_ids()
        planned_kinds = {kind(p) for p in plan}
        keep = {i: ep.cart_lines.get(i, 1) for i in now if i not in plan and not shelf.sold_out(i) and kind(i) not in planned_kinds}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(ep.items):
            return ""
        lines = {**keep, **plan}
        calls = self.shelf.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > ep.left():
            return ""
        # acting takes no model call; a cart the checks have not passed gets one check, allowed up to the hard spend cap
        self.scout.fill_facts(list(lines), len(calls) + 3)
        checks.check_cart(list(lines), SPEND_HARD)
        if not checks.verdicts.get(checks.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            shaped = ep.fit_args(cart, args)
            reply = ep.act(cart, shaped) if shaped is not None else None
            if reply is None or ep.finished:
                return ""
            self.shelf.note_cart(cart, shaped)
            self.ledger.observe(cart, reply.get("observation"))
            done.append(to_json(shaped))
        args = ep.buy_args(buy)
        if args is None:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one more try after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.scout.reread_offers()
                checks.check_cart(list(lines), SPEND_HARD)
                if (not checks.verdicts.get(checks.check_key(list(lines))) or ep.left() < 1 or ep.finished
                        or any(shelf.sold_out(i) for i in lines)):
                    break
            go, said = self.talk.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = ep.act(buy, args)
            ep.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": cut(reply, 200)})
            if reply is None:
                break
            ep.buys += 1
            if not reply.get("finished"):
                ep.refused_at = ep.step
            self.ledger.observe(buy, reply.get("observation"))
            replies.append(render(reply.get("observation")))
            if ep.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not executed; Kyo acted instead: the independent check confirmed " + confirmed + ", so Kyo set the cart ("
                + "; ".join(done) + ") and checked out. Store reply: " + " | then: ".join(replies))

    def closest_cart(self) -> str:
        """A repeated decline although the request names the closest listing as its finish: Kyo sets the cart to one line
        per item at the requested quantity (the agent's own cart lines first, then the review's leads, then the fullest
        cart built) and checks out. No full fit is needed: the closest listing is what such a request takes."""
        ep, shelf, checks = self.ep, self.shelf, self.checks
        cart, buy = ep.roles.get("cart"), ep.roles.get("buy")
        if self.closest_done or ep.brief.get("closest_if_none") is not True or not cart or not buy:
            return ""
        self.closest_done = True
        now = shelf.cart_ids()
        plan: dict[str, int] = {}
        for number, item in enumerate(ep.items, 1):
            pool = ([i for i in now if shelf.item_of(i) == number] + checks.lead_items.get(str(number), [])
                    + [i for i in shelf.built if shelf.item_of(i) == number])
            pick = next((i for i in pool if i not in plan and not shelf.sold_out(i)), None)
            if not pick:
                ep.trace.append({"kind": "closest_cart_none", "item": number})
                return ""
            plan[pick] = item.qty
        spend = [(shelf.price_of(i) or 0.0) * n for i, n in plan.items()]
        if ep.budget and (max(spend) if ep.brief.get("budget_scope") == "each" else sum(spend)) > ep.budget:
            return ""
        drop = [i for i in now if i not in plan]
        calls = shelf.cart_calls(plan, drop)
        if calls is None or len(calls) + 1 > ep.left():
            return ""
        done = []
        for args in calls:
            shaped = ep.fit_args(cart, args)
            reply = ep.act(cart, shaped) if shaped is not None else None
            if reply is None or ep.finished:
                return ""
            shelf.note_cart(cart, shaped)
            self.ledger.observe(cart, reply.get("observation"))
            done.append(to_json(shaped))
        args = ep.buy_args(buy)
        if args is None:
            return ""
        go, said = self.talk.gate("order") if ep.left() >= 2 and not ep.late(CLOCK_GATE) else (True, "")
        if not go:
            return "Not executed; Kyo set the cart (" + "; ".join(done) + ") for the closest-listing checkout, but: " + said
        reply = ep.act(buy, args)
        ep.trace.append({"kind": "closest_cart", "lines": plan, "drop": drop, "reply": cut(reply, 200)})
        if reply is None:
            return ""
        ep.buys += 1
        if not reply.get("finished"):
            ep.refused_at = ep.step
        self.ledger.observe(buy, reply.get("observation"))
        return ("Not executed; Kyo acted instead: the request takes the closest listing when nothing fits every requirement,"
                " so Kyo set the cart (" + ("; ".join(done) or "as it stood") + ") and checked out. Store reply: "
                + render(reply.get("observation")))

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        handed back (as the store's reply) for the agent to deal with instead of being followed by a decline."""
        ep = self.ep
        if ep.finished or ep.broke or ep.left() <= 0:
            return ""
        buy, refuse = ep.roles.get("buy"), ep.roles.get("refuse")
        # a request that takes the closest listing never ends empty while one line per item can be set: Kyo sets them
        # (at the requested quantities) and checks out
        done = self.closest_cart() if ep.brief.get("closest_if_none") is True and ep.left() >= 2 else ""
        if done and (ep.finished or ep.left() <= 0):
            return ""
        if done and again:
            return "the store refused the checkout: " + done.rsplit("Store reply: ", 1)[-1]
        fresh = ep.fresh_cart()
        holding = bool(self.shelf.cart_ids()) if fresh else bool(ep.cart_lines)
        if buy and (holding or (ep.cart_touched and not fresh and ep.left() >= 2)):
            args = ep.buy_args(buy)
            if args is not None:
                # a reply that adds a requirement cannot stop a forced finish: no steps are left to act on it
                if ep.left() >= 2 and not ep.late(CLOCK_GATE):
                    self.talk.gate("order")
                if ep.finished or ep.left() <= 0:
                    return ""
                self.scout.reread_offers(since=-1)
                if ep.finished or ep.left() <= 0:
                    return ""
                reply = ep.act(buy, args)
                ep.trace.append({"kind": "forced_buy", "reply": cut(reply, 200)})
                if ep.finished or ep.left() <= 0:
                    return ""
                if reply is not None and again:
                    ep.buys += 1
                    ep.refused_at = ep.step
                    self.ledger.observe(buy, reply.get("observation"))
                    return "the store refused the checkout: " + render(reply.get("observation"))
        if refuse:
            args = ep.fit_args(refuse, self.quiet(refuse, dict(ep.brief.get("refuse_args") or {})))
            if args is not None:
                if ep.left() >= 2 and not ep.late(CLOCK_GATE):
                    self.talk.gate("end")
                if ep.finished or ep.left() <= 0:
                    return ""
                reply = ep.act(refuse, args)
                ep.trace.append({"kind": "forced_refuse", "reply": cut(reply, 200)})
                if reply is not None and not ep.finished:
                    self.ledger.observe(refuse, reply.get("observation"))
                    return "the store refused the ending without a purchase: " + render(reply.get("observation"))
        return ""


class Call:
    """One model tool call on its way through the pipeline, with what the stages learn about it."""

    def __init__(self, raw: dict[str, Any], alone: bool):
        fn = raw.get("function") or {}
        self.cid: str = raw["id"]
        self.name: str = fn.get("name")
        self.raw: Any = fn.get("arguments")
        self.alone = alone                  # the only call of its turn
        self.args: dict[str, Any] = {}
        self.mark = ""                      # tool name plus arguments: the key of repeats and the read cache
        self.many: str | None = None        # the list argument, for the side-by-side tool
        self.ending = False                 # a checkout or an ending without a purchase
        self.readers: set[str] = set()
        self.plain: set[str] = set()
        self.extra = ""                     # Kyo's notes added to the store's reply
        self.paged = ""
        self.reply: dict[str, Any] | None = None


class Pipeline:
    """Runs the model's tool calls. Each call passes the stages in order; a stage either answers the call (with no
    store step, or with Kyo's own steps) or lets it on. A call that passes them all goes to the store; its reply is
    then recorded and annotated."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger, scout: Scout, talk: Talk, checks: Checks, closer: Closer):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.scout = scout
        self.talk = talk
        self.checks = checks
        self.closer = closer
        self.stages = (self.screen, self.guard, self.hold_walk, self.hold_buy, self.hold_refuse, self.hold_decline, self.gatekeep)
        self.held = 0
        self.walks: dict[str, dict[str, Any]] = {}   # each search (its arguments less the page one): ids by page, count, next
        self.id_walk: set[str] = set()     # search tools whose pages were seen to run on in id order
        self.walk_holds = 0
        self.refuse_blocks = 0
        self.pushed = False
        self.late_held = False
        self.closest_held = False
        self.wider_held = False
        self.free_holds = 0
        self.empty_held = False

    def run(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        calls = self.join_opens(calls)
        replies = []
        for raw in calls:
            if raw.get("joined"):
                replies.append((raw["id"], "Covered by the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            call = Call(raw, len(calls) == 1)
            answer = None
            for stage in self.stages:
                answer = stage(call)
                if answer is not None:
                    break
            if answer is None:
                call.reply = self.ep.act(call.name, call.args)
                if call.reply is None:
                    answer = "The call failed (network or server). You may try it again."
                else:
                    self.record(call)
                    answer = self.annotate(call)
            replies.append((call.cid, answer))
        return replies

    def offered(self) -> list[dict[str, Any]]:
        """The tools the model may call now: never the messaging tool, and only finishing tools near the end."""
        ep = self.ep
        left = ep.left()
        ends = {ep.roles.get("buy"), ep.roles.get("refuse")} - {None}
        carts = {ep.roles.get("cart"), ep.roles.get("cart_view")}
        keep = None
        if left <= ep.tail():
            keep = set(ends)
        elif left <= ep.tail() + 3:
            keep = ends | carts
        elif left <= len(ep.items) + 5 and ep.buys == 0:
            keep = ends | carts | {ep.roles.get("offer")}
        tools = [t for t in ep.tools if t["function"]["name"] != ep.roles.get("ask")]
        if keep is not None and ends:
            tools = [t for t in tools if t["function"]["name"] in keep] or tools
        return tools

    # ----- stages that may answer a call without a store step -----
    def screen(self, call: Call) -> str | None:
        """Calls that cannot run as sent: wrong tool, unusable arguments."""
        ep = self.ep
        allowed = {t["function"]["name"] for t in self.offered()}
        if ep.finished:
            return "Not executed: the episode is over."
        if call.name == ep.roles.get("ask"):
            return SKIP + "Kyo's interviewer sends every message to the shopper."
        if call.name not in ep.schemas:
            return SKIP + f"there is no tool by that name. Tools: {', '.join(sorted(allowed))}."
        if call.name not in allowed:
            return SKIP + "too few steps remain for that; use one of: " + ", ".join(sorted(allowed)) + "."
        raw = call.raw
        args = raw if isinstance(raw, dict) else (parse_object(raw) if (raw or "").strip() else {})
        if args is None:
            return SKIP + "the arguments were not a single JSON object."
        faults: list[str] = []
        args = conform(args, ep.schemas[call.name], "args", faults)
        if faults:
            return SKIP + "; ".join(faults[:6])
        call.args, wrong = self.resolve_ids(call.name, args)
        return SKIP + wrong + "." if wrong else None

    def guard(self, call: Call) -> str | None:
        """Trim the arguments, and stop calls that would spend a step for nothing."""
        ep, shelf = self.ep, self.shelf
        name, args = call.name, call.args
        many = call.many = shelf.compare_target() if name == ep.roles.get("compare") else None
        if many and isinstance(args.get(many), list) and len(args[many]) > shelf.compare_room():
            room = shelf.compare_room()
            call.extra = (f"\n[Kyo] Only the first {room} listings were sent: "
                          + (f"this store shows at most {room} side by side." if shelf.compare_bad else f"Kyo sends {room} side by side at a time.")
                          + f" Still to show: {', '.join(map(str, args[many][room:]))}.")
            args = {**args, many: args[many][:room]}
        target = shelf.single_target(name)
        if target and name in (ep.roles.get("open"), ep.roles.get("offer")) and str(args.get(target)) in shelf.gone:
            return (SKIP + f"{args[target]} sold out at step {shelf.gone[str(args[target])]} and will not come back."
                    " Replace its cart line with another fitting listing.")
        if name == ep.roles.get("cart") and any((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            view = to_json(ep.cart_view) if ep.cart_seen_at >= ep.cart_at else ""
            absent = [v for v in args.values() if isinstance(v, str) and v in shelf.rows and v not in ep.cart_lines
                      and v not in view and v not in self.shelf.cart_said]
            if absent:
                return SKIP + f"{absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.shelf.cart_text()
        call.ending = ep.ends("buy", name, args) or ep.ends("refuse", name, args)
        if call.ending and not call.alone:
            return SKIP + "send the finishing call by itself, after you have read the results of your other calls."
        if not call.ending and ep.left() <= ep.tail() and (ep.roles.get("buy") or ep.roles.get("refuse")):
            return (SKIP + "the last steps are reserved for checking out (with the shopper's approval of the basket when it is due)"
                    " or for ending without a purchase.")
        call.mark = name + to_json(args)
        if not call.ending and name != ep.roles.get("cart") and call.mark in ep.recent:
            return SKIP + "you just made this exact call and nothing has changed since; its result is above."
        call.readers = {ep.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
        cached = shelf.read_cache.get(call.mark)
        if name in call.readers and cached and cached[0] == ep.version:
            return SKIP + f"the identical call at step {cached[1]} returned the following, and nothing has changed since:\n" + cached[2][:3000]
        if target and name in call.readers and name != ep.roles.get("compare") and shelf.shown_already(name, str(args.get(target))):
            return SKIP + f"{args[target]} has already shown everything this view offers since the last change. Its record: " + shelf.record_of(str(args[target]))
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not shelf.shown_already(name, str(i))]
            if not keep:
                return (SKIP + "these listings have already shown everything this view offers since the last change. Their records:\n"
                        + "\n".join(f"{i}: {shelf.record_of(str(i))}" for i in args[many])[:4000])
            if len(keep) < len(args[many]):
                call.extra += f"\n[Kyo] Skipped {', '.join(str(i) for i in args[many] if i not in keep)}: already shown in full, unchanged since."
                args = {**args, many: keep}
                call.mark = name + to_json(args)
        call.args = args
        return None

    def walk_key(self, name: str, args: dict[str, Any]) -> tuple[str, str | None]:
        """A search's key without its page argument, and that argument's name when the call carries one."""
        page = next((k for k, v in args.items() if PAGE_ARG.search(k) and v not in (None, "", 0, 1)), None)
        return name + to_json({k: v for k, v in args.items() if not PAGE_ARG.search(k)}), page

    def hold_walk(self, call: Call) -> str | None:
        """A further page of a long list that runs in id order: it cannot be walked to the end in the steps left, so its
        next page is a blind draw, while a keyword search ranks the closest matches first."""
        ep = self.ep
        key, page = self.walk_key(call.name, call.args)
        walk = self.walks.get(key)
        if not page or not walk or call.name not in self.id_walk or self.walk_holds >= 3:
            return None
        ids, count = walk["ids"], walk["count"]
        size = max(walk["size"], 1)
        if not count or ids != sorted(ids) or len(ids) != len(set(ids)) or -(-(count - len(ids)) // size) <= ep.left() // 2:
            return None
        self.walk_holds += 1
        open_pages = [f"{w['name']} {w['base']} -> next page with {w['param']}={to_json(w['next'])}"
                      for k, w in reversed(list(self.walks.items())) if k != key and w["name"] not in self.id_walk
                      and w["next"] is not None and w["param"] and w["words"]][:3]
        ep.trace.append({"kind": "walk_held", "call": key, "count": count, "seen": len(ids)})
        return (HOLD + f"This list runs in id order, not by relevance or price: {count} matches at {size} a page would take"
                f" about {-(-(count - len(ids)) // size)} more steps, far more than the {ep.left()} left, so its next page is a"
                " blind draw. A keyword search ranks the closest matches first, and a fitting listing often sits on its second"
                " or third page: run a short query with the required value first (one to three words), and fetch the next"
                " pages of a short keyword query that came close." + (" Keyword searches with pages still unread: "
                + "; ".join(open_pages) + "." if open_pages else ""))

    def note_walk(self, call: Call, observation: Any) -> None:
        """Record a search page: its ids in order, the reported total and the value that fetches the next page."""
        name, args = call.name, call.args
        if not isinstance(observation, dict):
            return
        key, page = self.walk_key(name, args)
        ids = [i for i, _ in self.shelf.records_in(observation)]
        count = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                      and re.search(r"count|total|found|match|hits|results", str(k), re.I)), None)
        nxt = next((v for k, v in observation.items() if isinstance(v, (str, int)) and not isinstance(v, bool)
                    and PAGE_ARG.search(str(k))), None)
        props = (self.ep.schemas.get(name) or {}).get("properties") or {}
        walk = self.walks.get(key) if page else None
        if walk is None:
            words = any(isinstance(v, str) and re.search(r"[a-zA-Z]{3}", v) and "enum" not in (props.get(k) or {})
                        and (name, k) not in self.shelf.id_args for k, v in args.items() if not PAGE_ARG.search(k))
            walk = self.walks[key] = {"name": name, "base": to_json({k: v for k, v in args.items() if not PAGE_ARG.search(k)}),
                                      "ids": [], "pages": 0, "size": len(ids), "count": None, "next": None, "words": words,
                                      "param": next((k for k in props if PAGE_ARG.search(k)), None)}
        walk["ids"] = walk["ids"] + ids
        walk["pages"] += 1
        walk["count"] = count if count is not None else walk["count"]
        walk["next"] = nxt
        # two pages that run on in id order: the tool lists in id order, whatever it is asked
        if walk["pages"] >= 2 and len(walk["ids"]) >= 6 and walk["ids"] == sorted(walk["ids"]) and len(set(walk["ids"])) == len(walk["ids"]):
            self.id_walk.add(name)

    def hold_buy(self, call: Call) -> str | None:
        """Holds before a checkout with steps to spare, in this order: a wider search, fresh offers and the cart view,
        sold-out lines, page chains, the cart check, a re-read that showed changes."""
        ep, shelf, checks = self.ep, self.shelf, self.checks
        if (call.ending and ep.ends("buy", call.name, call.args) and not self.empty_held and not shelf.cart_ids()
                and (ep.fresh_cart() or not ep.cart_touched)):
            # a checkout of nothing is refused and spends a step, which at the very end leaves the episode unfinished
            self.empty_held = True
            return HOLD + "The cart is empty, so a checkout cannot go through: cart each item's listing first, or end without a purchase."
        # past the clock's stop the checkout goes through unreviewed: a review that runs past the limit scores nothing
        if not call.ending or ep.left() <= 2 or not ep.ends("buy", call.name, call.args) or ep.late():
            return None
        if not self.wider_held and ep.brief.get("pick") == "cheapest" and ep.left() >= 14 + 2 * len(ep.items):
            # once: store searches read titles, so a requirement's value finds few of the fitting listings
            self.wider_held = True
            priced = [f"{i} at {self.shelf.price_of(i):g}" for i in shelf.cart_ids() if self.shelf.price_of(i) is not None]
            # words of an item's own name never searched alone: titles carry the product's name, not its attributes
            unused = sorted({w for item in ep.items for w in plain(f"{item.name} {item.shelf or ''}").split()
                             if len(w) >= 4 and not stems(w) <= shelf.queried})
            return (HOLD + f"{ep.left()} steps remain and the pick rule is the cheapest. Store searches match listing titles, so"
                    " searching for a requirement's value (often present only in the attributes) misses many fitting listings, and"
                    " rows arrive in no price order. Before checking out, run a few more searches per item using other names for the"
                    " product (its plain one-word name, synonyms, words from fitting titles you have seen) and consider only buyable"
                    " rows priced below the cart's line" + (f" ({'; '.join(priced)})" if priced else "")
                    + (f"; name words not searched yet: {', '.join(unused)}" if unused else "")
                    + "; read the attributes of the cheapest of those, swap in any that fits, then check out.\nCart now: " + shelf.cart_text())
        reread = self.scout.reread_offers()
        view = ep.roles.get("cart_view")
        if ep.cart_touched and ep.cart_seen_at < ep.cart_at and view and ep.left() > 3:
            args = ep.fit_args(view, {})
            reply = ep.act(view, args) if args is not None else None
            if reply is not None:
                ep.cart_view = reply.get("observation")
                ep.cart_seen_at = ep.step
                change = self.ledger.observe(view, ep.cart_view)
                if change:
                    ep.queued = []
                    return (HOLD + "Viewing the cart revealed a change: " + change + "\nCart now: " + shelf.cart_text()
                            + "\nApply the change, correct the cart if needed, then check out.")
        swap = checks.swap_hint()
        # a sold-out line holds the checkout back twice at most, so a wrong cart record cannot stall the episode
        held = [i for i in shelf.cart_ids() if i in self.shelf.gone] if self.held < 2 else []
        if swap or held:
            self.held += 0 if swap else 1
            return HOLD + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + shelf.cart_text()
        chain, hold = checks.trail()
        if hold:
            return HOLD + chain + "\nCart now: " + shelf.cart_text()
        # the check judges attributes: read the cart lines' own (and, under a cheapest pick, a few cheaper rows')
        read = self.scout.fill_facts(shelf.cart_ids(), ep.tail() + 3) + self.scout.cheaper_rows()
        # free, so it goes first: a cart the records already contradict needs no paid review
        clash = checks.free_clash() if self.free_holds < FREE_HOLDS and ep.left() > 3 else ""
        if clash:
            self.free_holds += 1
            ep.trace.append({"kind": "free_hold", "clash": clash})
            better = [i for n, need in ep.needs_now() for i in self.scout.stating(n, need) if i not in shelf.cart_ids()][:4]
            return (HOLD + "The records disagree with the requirements: " + clash + "."
                    + (f" Opened listings that state them: {', '.join(better)}." if better else "")
                    + " Replace that line, or check out again if the value is only worded differently.\nCart now: " + shelf.cart_text())
        verdict = checks.check_cart()
        if verdict:
            seen = ("Kyo read these listings' attributes for the review (one step each):\n" + "\n".join(read) + "\n") if read else ""
            return "Held back. " + chain + seen + verdict + "\nCart now: " + shelf.cart_text()
        if reread and ep.cost > REVIEW_SPEND:
            return (HOLD + "Kyo just re-read the cart's listings:\n" + reread + "\nCheck out again once every line still shows stock"
                    " and a price within the budget; otherwise replace that line first.")
        return None

    def hold_refuse(self, call: Call) -> str | None:
        """Holds before a decline with steps to spare: the review, then its leads (named once, then checked once)."""
        ep, checks = self.ep, self.checks
        if not call.ending or ep.left() <= 5 or ep.ends("buy", call.name, call.args) or ep.late():
            return None
        review = checks.audit()
        if review:
            return HOLD + review
        plan = checks.lead_plan() if ep.leads and self.refuse_blocks == 0 and not ep.late(CLOCK_GATE) else []
        if plan and ep.left() > 2 * len(plan) + 2:
            # a second decline while the review names a fit for every item: Kyo checks that cart and, when it passes, sets
            # the cart to it and checks out (the agent's view that a cart cannot be repaired is no reason to end empty)
            checks.check_cart(plan, SPEND_HARD)
            if checks.verdicts.get(checks.check_key(plan)):
                checks.confirmed = plan[0]
                done = self.closer.complete_cart()
                if done:
                    self.refuse_blocks = 2
                    ep.trace.append({"kind": "order_instead", "plan": plan})
                    return done + (DONE if ep.finished else "")
        if ep.leads and ep.left() > 4 and self.refuse_blocks == 0:
            self.refuse_blocks = 1
            ep.trace.append({"kind": "refuse_blocked", "leads": list(ep.leads)})
            return (HOLD + "The review found listings that appear to fit: " + ", ".join(ep.leads) + ". Open them if needed, cart the"
                    " best-fitting one (respecting the pick rule) and check out. End without a purchase only if each of them clearly"
                    " fails a requirement; in that case name which one in your text and send the ending call again.")
        if ep.leads and ep.left() > 4 and self.refuse_blocks == 1:
            self.refuse_blocks = 2
            first = ep.leads[0]
            checks.check_cart([first], SPEND_HARD)
            if checks.verdicts.get(checks.check_key([first])) and first in ep.leads:
                checks.confirmed = first
                ep.trace.append({"kind": "refuse_blocked", "confirmed": first})
                return (HOLD + f"An independent check of {first}'s records finds that it meets every requirement. Add it to the cart"
                        " at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement.")
        return None

    def hold_decline(self, call: Call) -> str | None:
        """A decline late in the episode or before the search is done; Kyo's own checkout of a confirmed lead; the
        decline's wording."""
        ep, shelf = self.ep, self.shelf
        if not ep.ends("refuse", call.name, call.args):
            return None
        if ep.brief.get("closest_if_none") is True and not self.closest_held and ep.left() > 2:
            # the request names its own finish for a store without a full fit: the closest listing, bought
            self.closest_held = True
            return (HOLD + "The request asks for the listing that comes closest when nothing meets every requirement, so this case"
                    " ends with a checkout, not without a purchase. For each item put the closest listing in the cart (the one"
                    " missing the fewest requirements, within any price margin the request sets, in stock and inside the"
                    " budget), then check out; no confirmation from the shopper is needed.\nCart now: " + shelf.cart_text())
        if ep.brief.get("closest_if_none") is True and self.closest_held:
            # declined again after that hold: Kyo checks out the closest cart itself (a decline scores nothing here)
            done = self.closer.closest_cart()
            if done:
                return done + (DONE if ep.finished else "")
        if ep.left() <= 5 and not self.late_held:
            # too late for the review: a decline now mostly throws away a built cart because time ran short
            lines = [i for i in shelf.built if not self.shelf.sold_out(i)]
            fresh = ep.fresh_cart()
            current = shelf.cart_ids()
            missing = [i for i in lines if i not in current or (i in shelf.dropped and not fresh)]
            if len(lines) >= len(ep.items) and len(missing) + 1 <= ep.left():
                self.late_held = True
                ep.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                redo = (f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing
                        else "Check out the cart as it stands.")
                return (HOLD + "Ending without a purchase scores nothing when the store has a fitting listing, and running short of"
                        " steps does not prove it has none. " + redo + " End without a purchase only if your search showed that some"
                        " requested item has no listing meeting its requirements; then send the ending call again.")
        if not ep.leads and not self.pushed and ep.left() > ep.max_steps // 3:
            self.pushed = True
            return (HOLD + f"{ep.left()} steps still remain, so before ending try searches you have not tried yet: fetch the next"
                    " pages of your short keyword searches with the token or offset the store returns (they rank the closest"
                    " matches first, so a fitting listing often sits on page two or three), try single-word value queries and"
                    " other names for the property, page through a category only when it is short, and open the listings that"
                    " might fit. Only then end without a purchase.")
        done = self.closer.complete_cart()
        if done:
            return done + (DONE if ep.finished else "")
        call.args = self.closer.quiet(call.name, call.args)
        return None

    def gatekeep(self, call: Call) -> str | None:
        """The steps the request asks for right before this checkout or ending."""
        ep = self.ep
        if not call.ending or ep.left() < 2 or ep.late(CLOCK_GATE):
            return None
        go, said = self.talk.gate("order" if ep.ends("buy", call.name, call.args) else "end")
        if not go:
            return "Held back. " + said + "\nCart now: " + self.shelf.cart_text()
        if said:
            call.extra = "\n[Kyo] " + said + call.extra
        return None

    # ----- after the store answered -----
    def record(self, call: Call) -> None:
        """What the reply teaches: the side-by-side limit, which args take ids, records, views, paging, which tool shows
        attributes or offers, the cart."""
        ep, shelf = self.ep, self.shelf
        name, args, many = call.name, call.args, call.many
        observation = call.reply.get("observation")
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            if not failed(observation) and any(str(i) in to_json(observation) for i in args[many]):
                shelf.compare_ok = max(shelf.compare_ok, len(args[many]))
            else:
                shelf.compare_bad = min(shelf.compare_bad or 999, len(args[many]))
                call.extra += f"\n[Kyo] The side-by-side view returned nothing for {len(args[many])} listings: send no more than {shelf.compare_room()} per call."
        ep.recent = (ep.recent + [call.mark])[-3:]
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(x, (str, int)) and str(x) in shelf.rows for x in values):
                shelf.id_args.add((name, key))
        shelf.absorb(observation)
        if name == ep.roles.get("compare"):
            rows = shelf.records_in(observation)
            if rows:
                shelf.facts_from[name] = shelf.facts_from.get(name, False) or any(shelf.facts_of(r) for _, r in rows)
        if name in call.readers - {ep.roles.get("search")}:
            shelf.note_view(name, observation)
        # only a search result is paged: the tools of every other job are left out
        call.plain = {tool for job, tool in ep.roles.items() if job != "search"}
        if name not in call.plain and not call.ending:
            self.note_walk(call, observation)
            call.paged = self.scout.more_pages(name, args, observation)
            props = (ep.schemas.get(name) or {}).get("properties") or {}
            words = [k for k, v in args.items() if isinstance(v, str) and re.search(r"[a-zA-Z]{3}", v) and (name, k) not in shelf.id_args
                     and "enum" not in (props.get(k) or {}) and not re.search(r"token|cursor|page|next|offset|after|start|begin|from", k, re.I)]
            for key in words:
                shelf.queried |= stems(args[key])
            if len(words) == 1 and len(shelf.records_in(observation)) >= 2:
                shelf.text_search = (name, words[0], {k: v for k, v in args.items() if "enum" in (props.get(k) or {})})
        fixed_jobs = (ep.roles.get("cart"), ep.roles.get("buy"), ep.roles.get("refuse"), ep.roles.get("ask"))
        listing = shelf.ident(observation) if isinstance(observation, dict) else None
        if listing and shelf.single_target(name) and name not in fixed_jobs:
            shelf.facts_from[name] = shelf.facts_from.get(name, False) or bool(shelf.facts_of(observation))
            # the attributes view showed none: another single-listing tool not yet seen bare takes the job
            if name == ep.roles.get("open") and not shelf.facts_from[name]:
                other = next((t for t in ep.schemas if t != name and shelf.single_target(t) and shelf.facts_from.get(t) is not False
                              and t not in fixed_jobs + (ep.roles.get("search"),)), None)
                if other:
                    ep.roles["open"] = other
            if not shelf.facts_from[name] and name != ep.roles.get("open") and not ep.roles.get("offer"):
                ep.roles["offer"] = name
            if name == ep.offer_tool():
                shelf.offer_seen[listing] = ep.step
        if name == ep.roles.get("cart"):
            shelf.note_cart(name, args)
            shelf.cart_said = to_json(observation)
            ep.recent = []
            # lines a zero or negative count took off (unless the reply still lists them), and the fullest cart built
            # so far, which is offered back when a late decline would throw it away
            counts = [as_int(v) for v in args.values() if not isinstance(v, str) and as_int(v) is not None]
            for value in args.values():
                if counts and isinstance(value, str) and value in shelf.rows:
                    if counts[0] <= 0 and not holds_id(observation, value):
                        shelf.dropped.add(value)
                    elif counts[0] > 0:
                        shelf.dropped.discard(value)
            lines = [i for i in shelf.cart_ids() if not shelf.sold_out(i) and i not in shelf.dropped]
            if len(lines) >= len(shelf.built):
                shelf.built = lines
            if ep.left() <= 14 and any(i not in shelf.offer_seen for i in lines):
                # the end is near enough that this line probably stays: read its page now, while a step is still cheap
                self.scout.reread_offers(since=-1)
        if name == ep.roles.get("cart_view"):
            ep.cart_view = observation
            ep.cart_seen_at = ep.step
        if ep.ends("buy", name, args):
            ep.buys += 1
            if not call.reply.get("finished"):
                ep.refused_at = ep.step

    def annotate(self, call: Call) -> str:
        """The reply the model sees: the result, then what Kyo adds (pages, cheapest rows, evidence, notes, swaps)."""
        ep, shelf, ledger = self.ep, self.shelf, self.ledger
        name, args = call.name, call.args
        observation = call.reply.get("observation")
        text = render(observation) + call.paged + call.extra
        if name not in call.plain and not call.ending:
            text += self.scout.cheapest_here([observation] + self.scout.paged_obs)
        cached = text
        proof = shelf.evidence(observation)
        if proof:
            text += "\n" + proof
        if name == ep.roles.get("cart") and any(as_int(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in shelf.rows and holds_id(observation, v) for v in args.values()):
                text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count removes units."
            elif shelf.cart_sets(name):
                text += "\n[Kyo] A count of 0 holds none of this listing: the number is how many units the line holds, so 0 takes it out."
        if name == ep.roles.get("cart"):
            text += self.unit_hint(name, args)
        ledger.note_sold = False
        lines = ledger.loose_lines(store_note(ep.roles, name, observation))
        if lines:
            meaning = ledger.take_note(lines)
            if meaning:
                text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
                # a change to the task can come with an unannounced sell-out or price move of a cart line: each line not
                # read at this step is read again now (a step each), while steps remain to replace it
                seen = self.scout.reread_offers(ep.step) if not ep.ends("buy", name, args) and ep.left() > 8 else ""
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the note]\n" + seen
        if (ep.ends("buy", name, args) and not call.reply.get("finished")
                and (ledger.note_sold or any(line in words_of(observation) for line in ledger.sold_lines))):
            # the refusal names no listing: a lone cart line is the one; several are re-read so their records show which
            lines = [i for i in shelf.cart_ids() if i not in shelf.gone]
            seen = ""
            if lines and not any(shelf.sold_out(i) for i in lines):
                if len(lines) == 1:
                    shelf.unavailable.add(lines[0])
                else:
                    seen = self.scout.reread_offers()
            if seen:
                text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
        swap = self.checks.swap_hint()
        if swap:
            text += "\n" + swap
        if ep.queued:
            text += "\n[notes seen on Kyo's own calls] " + " | ".join(ep.queued)
            ep.queued = []
        if name in call.readers:
            shelf.read_cache[call.mark] = (ep.version, ep.step, cached)
        return text + (DONE if call.reply.get("finished") else "")

    def unit_hint(self, name: str, args: dict[str, Any]) -> str:
        """A cart line whose units (Kyo's tally) differ from its item's requested quantity: what the number argument
        means and the call that sets the line right (a count argument named like a position is easily misread)."""
        ep, shelf = self.ep, self.shelf
        props = (ep.schemas.get(name) or {}).get("properties") or {}
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds_of(s)), None)
        lines = getattr(ep, "cart_lines", None) or {}
        for value in args.values():
            if not count or not isinstance(value, str) or value not in lines:
                continue
            wants = {item.qty for item in ep.items}
            have = lines[value]
            number = shelf.item_of(value)
            want = ep.items[number - 1].qty if number else (next(iter(wants)) if len(wants) == 1 else None)
            if want is None or have == want:
                continue
            if shelf.cart_sets(name):
                fix = f"send {name} again for {value} with {count} {want} (the number of units the line is to hold)"
            elif isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= want - have:
                fix = f"send {name} again for {value} with {count} {want - have} (this tool adds the number to the line)"
            else:
                fix = f"take the line out and add {value} again with {count} {want}"
            return (f"\n[Kyo] {value} now holds {have} unit(s), but " + (f"item {number}" if number else "the request")
                    + f" asks for {want}: {count} is how many units go on the line, not a line position. To correct it, {fix};"
                    " a wrong count is fixed this way, never a reason to end without a purchase.")
        return ""

    # ----- argument fixes -----
    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes ids: the id when exactly one seen row starts with that text,
        otherwise what is wrong, so no step goes to a call the store cannot match."""
        ep, shelf = self.ep, self.shelf
        if name not in {ep.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not shelf.id_key:
            return args, ""
        props = (ep.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in kinds_of(v) or "array" in kinds_of(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen with one)
        if len(free) != 1 or (name != ep.roles.get("cart") and (name, free[0]) not in shelf.id_args):
            return args, ""
        key = free[0]
        if key not in args:
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in shelf.rows:
                return value.strip(), ""
            if not isinstance(value, str) or id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in shelf.rows.items()
                    if any(k != shelf.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings start with the text {value!r} ({', '.join(hits[:6])}); pass a single listing id"
            return value, f"{value!r} is not a listing id; use the id from the result rows (their {shelf.id_key} field)"

        value = args[key]
        fixed = [fix(v) for v in value] if isinstance(value, list) else [fix(value)]
        problem = next((p for _, p in fixed if p), "")
        if problem:
            return args, problem
        return {**args, key: [v for v, _ in fixed] if isinstance(value, list) else fixed[0][0]}, ""

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        ep, shelf = self.ep, self.shelf
        opener, joiner = ep.roles.get("open"), ep.roles.get("compare")
        target, many = shelf.single_target(opener), shelf.compare_target()
        # only into a side-by-side view that shows what the opens would (attributes)
        if not opener or not joiner or not target or not many or opener == joiner or shelf.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = parse_object(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in shelf.rows and ident not in shelf.gone:
                picks.append((index, ident))
        picks = picks[: shelf.compare_room()]
        if len(picks) < 2:
            return calls
        merged = {index for index, _ in picks}
        ids = [ident for _, ident in picks]
        out = []
        for index, call in enumerate(calls):
            if index == picks[0][0]:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": to_json({many: ids})}})
            elif index in merged:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        ep.trace.append({"kind": "joined", "ids": ids})
        return out


class Runner:
    """The episode as a sequence of stages: read the request; explore with the model; finish, with a short second
    round when the store turns the finish down; finish for good."""

    def __init__(self, ep: Episode, shelf: Shelf, ledger: Ledger, talk: Talk, closer: Closer, pipeline: Pipeline):
        self.ep = ep
        self.shelf = shelf
        self.ledger = ledger
        self.talk = talk
        self.closer = closer
        self.pipeline = pipeline
        self.chat: list[dict[str, Any]] = []
        self.misses = 0
        self.loop_calls = 0

    def run(self) -> list[dict[str, Any]]:
        ep = self.ep
        if ep.finished:
            return ep.trace
        # stage 1: read the request and open the conversation
        self.ledger.read_request()
        ep.trace.append({"kind": "brief", "brief": ep.brief})
        opening = unfold(ep.view.get("opening"))
        intro = "SHOPPER'S REQUEST:\n" + ep.request
        if opening is not None:
            self.shelf.absorb(opening)
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + render(opening)
        intro += ("\n\nBegin by searching each item's category and opening several listings in full, so the interviewer can see"
                  " which properties these listings carry.\n\n" + self.ledger.status())
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        # stage 2: the model explores and shops
        self.work(LOOP_CALLS, SPEND_HARD)
        if not ep.finished:
            # stage 3: finish; the store may turn it down (a change from the shopper, a sold-out line) and a short
            # second round deals with that
            turned = self.closer.finish(again=True)
            if turned and not ep.finished and not ep.broke and not ep.no_credit and ep.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode, but " + turned
                                  + "\nApply what the store says, then finish.\n" + self.ledger.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            # stage 4: the last finish
            if not ep.finished:
                self.closer.finish()
        ep.trace.append({"kind": "end", "finished": ep.finished, "steps": ep.step, "cost": round(ep.cost, 5), "calls": ep.calls})
        return ep.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        ep = self.ep
        idle = 0
        while not ep.finished and not ep.broke and ep.left() > 0 and self.loop_calls < loop_cap and ep.cost < spend_cap and not ep.late():
            if self.talk.ready():
                summary = self.talk.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.ledger.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = ep.think(self.chat, [LIGHT, SPARE], 3000, self.pipeline.offered())
            if message is None:
                self.misses += 1
                if ep.no_credit or ep.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (parse_object(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": to_json(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = to_json(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Every reply needs a tool call: call one to continue.\n" + self.ledger.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            committing = {ep.roles.get("cart"), ep.roles.get("buy"), ep.roles.get("refuse")} - {None}
            if any(c["function"]["name"] in committing for c in calls) and self.talk.ready(forced=True):
                # a cart change or a finish before the interview: the shopper is asked first
                self.chat += [{"role": "tool", "tool_call_id": c["id"], "content": SKIP + "the shopper gets asked first."} for c in calls]
                summary = self.talk.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer found nothing to ask.") + "\n" + self.ledger.status()})
                continue
            ep.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.pipeline.run(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.ledger.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})

    def trim(self) -> None:
        """Shorten the older half of the tool results once the conversation grows past the history limit."""
        if sum(len(to_json(m.get("content"))) for m in self.chat) <= HISTORY_LIMIT:
            return
        results = [m for m in self.chat if m.get("role") == "tool"]
        for message in results[: max(1, len(results) // 2)]:
            content = message.get("content") or ""
            if len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.shelf.rows][:20]
                message["content"] = content[:400] + f"...(earlier result cut short; listings it named: {', '.join(ids)})"


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        ep = Episode(problem_data)
        shelf = Shelf(ep)
        ledger = Ledger(ep, shelf)
        scout = Scout(ep, shelf, ledger)
        talk = Talk(ep, shelf, ledger, scout)
        checks = Checks(ep, shelf, ledger, scout)
        closer = Closer(ep, shelf, ledger, scout, talk, checks)
        pipeline = Pipeline(ep, shelf, ledger, scout, talk, checks, closer)
        runner = Runner(ep, shelf, ledger, talk, closer, pipeline)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return runner.run()
    except Exception as exc:
        ep.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not ep.finished:
                closer.finish()
        except Exception as again:
            ep.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return ep.trace


__all__ = ["agent_main"]
